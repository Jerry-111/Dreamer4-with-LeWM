import argparse
import math
import os
from pathlib import Path

import hdf5plugin  # noqa: F401 - registers HDF5 compression filters used by pixels
import h5py
import torch
import torch.nn.functional as F


REQUIRED_KEYS = ("pixels", "action", "episode_idx", "step_idx", "ep_len", "ep_offset")
OPTIONAL_KEYS = ("state", "proprio")


def parse_args():
    p = argparse.ArgumentParser(description="Convert LeWM PushT HDF5 data to Dreamer4 shards.")
    p.add_argument("--input", type=str, default="/root/lewm-pusht/pusht_expert_train.h5")
    p.add_argument("--raw-out", type=str, default="/root/data/pusht-raw")
    p.add_argument("--frames-out", type=str, default="/root/data/pusht-shards")
    p.add_argument("--task", type=str, default="pusht")
    p.add_argument("--target-size", type=int, default=128)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--pixel-batch-size", type=int, default=512)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def safe_torch_save(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def validate_h5(f):
    missing = [key for key in REQUIRED_KEYS if key not in f]
    if missing:
        raise KeyError(f"Missing required HDF5 key(s): {missing}")

    pixels = f["pixels"]
    action = f["action"]
    episode_idx = f["episode_idx"]
    step_idx = f["step_idx"]

    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected pixels shape (N,H,W,3), got {pixels.shape}")
    if pixels.dtype != "uint8":
        raise ValueError(f"Expected uint8 pixels, got {pixels.dtype}")
    if action.ndim != 2:
        raise ValueError(f"Expected action shape (N,A), got {action.shape}")

    n = pixels.shape[0]
    for key, dataset in (("action", action), ("episode_idx", episode_idx), ("step_idx", step_idx)):
        if dataset.shape[0] != n:
            raise ValueError(f"{key} length {dataset.shape[0]} does not match pixels length {n}")

    if f["ep_len"].shape[0] != f["ep_offset"].shape[0]:
        raise ValueError("ep_len and ep_offset must have the same length")


def get_conversion_span(f, max_episodes):
    ep_len = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    total_episodes = int(ep_len.shape[0])
    episodes = total_episodes if max_episodes is None else min(int(max_episodes), total_episodes)
    if episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided")

    start = int(ep_offset[0])
    if start != 0:
        raise ValueError(f"Expected first ep_offset to be 0, got {start}")
    end = int(ep_offset[episodes - 1] + ep_len[episodes - 1])
    return episodes, end, ep_len[:episodes], ep_offset[:episodes]


def prepare_outputs(raw_path: Path, task_frames_dir: Path, *, overwrite: bool):
    existing_shards = sorted(task_frames_dir.glob("*shard*.pt")) if task_frames_dir.exists() else []
    if not overwrite:
        conflicts = []
        if raw_path.exists():
            conflicts.append(str(raw_path))
        conflicts.extend(str(p) for p in existing_shards[:5])
        if conflicts:
            extra = "" if len(existing_shards) <= 5 else f" (+{len(existing_shards) - 5} more shard files)"
            raise FileExistsError(
                "Output already exists. Pass --overwrite to replace:\n  "
                + "\n  ".join(conflicts)
                + extra
            )
        return

    if raw_path.exists():
        raw_path.unlink()
    for path in existing_shards:
        path.unlink()


def shift_actions_by_episode(action: torch.Tensor, ep_len: torch.Tensor, ep_offset: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(action)
    for offset, length in zip(ep_offset.tolist(), ep_len.tolist()):
        offset = int(offset)
        length = int(length)
        if length > 1:
            shifted[offset + 1 : offset + length] = action[offset : offset + length - 1]
    return shifted


def load_raw_metadata(f, n: int, ep_len, ep_offset):
    action = torch.from_numpy(f["action"][:n]).to(torch.float32)
    ep_len_t = torch.from_numpy(ep_len).to(torch.int32)
    ep_offset_t = torch.from_numpy(ep_offset).to(torch.int64)

    raw = {
        "episode": torch.from_numpy(f["episode_idx"][:n]).to(torch.int64),
        "action": shift_actions_by_episode(action, ep_len_t, ep_offset_t),
        "reward": torch.zeros(n, dtype=torch.float32),
        "step_idx": torch.from_numpy(f["step_idx"][:n]).to(torch.int64),
        "ep_len": ep_len_t,
        "ep_offset": ep_offset_t,
    }

    for key in OPTIONAL_KEYS:
        if key in f:
            raw[key] = torch.from_numpy(f[key][:n]).to(torch.float32)

    return raw


def resize_pixels_to_chw_u8(pixels_np, target_size: int) -> torch.Tensor:
    frames = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
    if frames.shape[-2:] == (target_size, target_size):
        return frames

    frames_f = frames.to(torch.float32) / 255.0
    resized = F.interpolate(
        frames_f,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return (resized.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


class FrameShardWriter:
    def __init__(self, out_dir: Path, task: str, shard_size: int):
        self.out_dir = out_dir
        self.task = task
        self.shard_size = int(shard_size)
        self.parts = []
        self.total = 0
        self.shard_idx = 0

    def add(self, frames: torch.Tensor):
        if frames.numel() == 0:
            return
        self.parts.append(frames.cpu())
        self.total += int(frames.shape[0])

        while self.total >= self.shard_size:
            concat = torch.cat(self.parts, dim=0)
            shard = concat[: self.shard_size].contiguous()
            remainder = concat[self.shard_size :].contiguous()
            self._write(shard)
            self.parts = [remainder] if remainder.shape[0] > 0 else []
            self.total = int(remainder.shape[0])

    def finish(self):
        if self.total > 0:
            shard = torch.cat(self.parts, dim=0).contiguous()
            self._write(shard)
            self.parts = []
            self.total = 0

    def _write(self, frames: torch.Tensor):
        out_path = self.out_dir / f"{self.task}_shard{self.shard_idx:04d}.pt"
        safe_torch_save({"frames": frames}, out_path)
        print(f"[frames] saved {out_path} shape={tuple(frames.shape)}")
        self.shard_idx += 1


def check_shift(raw):
    episode = raw["episode"]
    action = raw["action"]
    first = torch.ones_like(episode, dtype=torch.bool)
    first[1:] = episode[1:] != episode[:-1]
    first_actions = action[first]
    if first_actions.numel() > 0 and not torch.allclose(first_actions, torch.zeros_like(first_actions)):
        raise RuntimeError("Action shift validation failed: first action in an episode is non-zero")


def main():
    args = parse_args()
    input_path = Path(args.input)
    raw_dir = Path(args.raw_out)
    frames_dir = Path(args.frames_out)
    raw_path = raw_dir / f"{args.task}.pt"
    task_frames_dir = frames_dir / args.task

    with h5py.File(input_path, "r") as f:
        validate_h5(f)
        episodes, n, ep_len, ep_offset = get_conversion_span(f, args.max_episodes)
        expected_shards = int(math.ceil(n / args.shard_size))
        final_shard = n - (expected_shards - 1) * args.shard_size

        print(f"[schema] input={input_path}")
        print(f"[schema] episodes={episodes:,} frames={n:,}")
        print(f"[schema] pixels={f['pixels'].shape} action={f['action'].shape}")
        print(f"[schema] output raw={raw_path}")
        print(f"[schema] output frames={task_frames_dir}")
        print(f"[schema] expected_shards={expected_shards:,} final_shard_frames={final_shard:,}")

        if args.dry_run:
            return

        prepare_outputs(raw_path, task_frames_dir, overwrite=args.overwrite)
        raw_dir.mkdir(parents=True, exist_ok=True)
        task_frames_dir.mkdir(parents=True, exist_ok=True)

        raw = load_raw_metadata(f, n, ep_len, ep_offset)
        check_shift(raw)

        writer = FrameShardWriter(task_frames_dir, args.task, args.shard_size)
        for start in range(0, n, args.pixel_batch_size):
            end = min(start + args.pixel_batch_size, n)
            pixels_np = f["pixels"][start:end]
            frames = resize_pixels_to_chw_u8(pixels_np, args.target_size)
            writer.add(frames)
            print(f"[frames] processed {end:,}/{n:,}")
        writer.finish()

        safe_torch_save(raw, raw_path)
        print(f"[raw] saved {raw_path}")


if __name__ == "__main__":
    main()
