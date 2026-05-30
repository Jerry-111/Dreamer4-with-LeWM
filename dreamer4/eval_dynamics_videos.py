import argparse
import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

from interactive import (
    decode_single_packed_frame,
    find_episode_starts,
    load_dynamics_from_ckpt,
    load_task_action_dim,
    load_tokenizer_from_ckpt,
    make_tau_schedule,
    pack_bottleneck_to_spatial,
    sample_one_timestep_packed,
)
from model import temporal_patchify


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ShardCache:
    def __init__(self, max_items: int = 4):
        self.max_items = int(max_items)
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key, value):
        self.cache[key] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


def load_frame_cached(frames_dir: str, task: str, index: int, *, shard_size: int, cache: ShardCache) -> torch.Tensor:
    task_dir = os.path.join(frames_dir, task)
    shard_idx = int(index) // int(shard_size)
    off = int(index) % int(shard_size)
    key = (task_dir, shard_idx)
    frames = cache.get(key)
    if frames is None:
        shard_paths = sorted(Path(task_dir).glob('*shard*.pt'))
        if not shard_paths:
            raise FileNotFoundError(f'No shards found under {task_dir}')
        if shard_idx >= len(shard_paths):
            raise IndexError(f'index={index} -> shard_idx={shard_idx} but only {len(shard_paths)} shards')
        td = torch.load(shard_paths[shard_idx], map_location='cpu', weights_only=False)
        frames = td['frames']
        if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[1] != 3:
            frames = frames.permute(0, 3, 1, 2).contiguous()
        if frames.dtype != torch.uint8:
            frames_f = frames.to(torch.float32)
            mx = float(frames_f.max().item()) if frames_f.numel() > 0 else 0.0
            if mx > 1.5:
                frames = frames_f.clamp(0, 255).to(torch.uint8)
            else:
                frames = (frames_f.clamp(0, 1) * 255.0).to(torch.uint8)
        cache.put(key, frames)
    return frames[off].to(torch.float32) / 255.0


def load_raw_task(data_dir: str, task: str) -> Dict[str, torch.Tensor]:
    path = Path(data_dir) / f'{task}.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location='cpu', weights_only=False)


def valid_episode_starts(raw: Dict[str, torch.Tensor], starts: List[int], rollout_steps: int) -> List[int]:
    ep = raw['episode'].to(torch.int64).cpu()
    valid = []
    for s in starts:
        end = int(s) + int(rollout_steps)
        if end < ep.numel() and int(ep[s].item()) == int(ep[end].item()):
            valid.append(int(s))
    return valid


def pad_actions(actions: torch.Tensor, *, act_dim: int, A: int = 16) -> torch.Tensor:
    out = torch.zeros(actions.shape[0], A, dtype=torch.float32)
    if act_dim > 0:
        out[:, :act_dim] = torch.nan_to_num(actions[:, :act_dim].to(torch.float32), nan=0.0)
    return out


def to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().float().clamp(0, 1) * 255.0).to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def draw_panel(gt: np.ndarray, pred: np.ndarray, floor: np.ndarray, *, t: int, sample_idx: int, start_idx: int) -> np.ndarray:
    h, w, _ = gt.shape
    gap = 6
    header = 28
    footer = 18
    out_w = 3 * w + 2 * gap
    out_h = h + header + footer
    canvas = Image.new('RGB', (out_w, out_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    labels = ['GT frame', 'interactive rollout', 'frozen start baseline']
    xs = [0, w + gap, 2 * (w + gap)]
    for label, x0, img in zip(labels, xs, [gt, pred, floor]):
        canvas.paste(Image.fromarray(img), (x0, header))
        draw.text((x0 + 4, 6), label, fill=(240, 240, 240))
    phase = 'initial latent' if t == 0 else f'rollout +{t}'
    draw.text((4, header + h + 2), f'sample {sample_idx:03d} | start={start_idx} | t={t:02d} | {phase}', fill=(230, 230, 230))
    if t == 0:
        draw.rectangle((0, header, out_w - 1, out_h - 1), outline=(102, 194, 255), width=3)
    return np.asarray(canvas)


def write_mp4(frames: List[np.ndarray], path: Path, *, fps: float) -> None:
    if not frames:
        raise ValueError('No frames provided for video.')
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', str(float(fps)),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f'ffmpeg failed for {path}\nstdout={stdout.decode(errors="ignore")}\nstderr={stderr.decode(errors="ignore")}'
        )


@torch.inference_mode()
def encode_initial_latents(encoder, frames0: torch.Tensor, *, device: torch.device, C: int, H: int, W: int, patch: int,
                           n_spatial: int, packing_factor: int, use_amp: bool) -> torch.Tensor:
    frames0 = frames0.to(device)
    patches0 = temporal_patchify(frames0.view(frames0.shape[0], 1, C, H, W), patch)
    z0_btLd, _ = encoder(patches0)
    z0_packed = pack_bottleneck_to_spatial(z0_btLd, n_spatial=n_spatial, k=packing_factor)[:, 0]
    if use_amp and device.type == 'cuda':
        z0_packed = z0_packed.to(torch.float16)
    else:
        z0_packed = z0_packed.to(torch.float32)
    return z0_packed.detach()


@torch.inference_mode()
def decode_batch_like_interactive(decoder, z_batch: torch.Tensor, *, H: int, W: int, C: int, patch: int,
                                  packing_factor: int, d_bottleneck: int) -> torch.Tensor:
    frames = []
    for i in range(z_batch.shape[0]):
        fr = decode_single_packed_frame(
            decoder,
            z_packed=z_batch[i],
            H=H,
            W=W,
            C=C,
            patch=patch,
            packing_factor=packing_factor,
            d_bottleneck=d_bottleneck,
        )
        frames.append(fr.detach().cpu())
    return torch.stack(frames, dim=0)


@torch.inference_mode()
def rollout_interactive_replay(
    *,
    encoder,
    decoder,
    dyn,
    gt_frames: torch.Tensor,
    actions: torch.Tensor,
    act_mask_1d: torch.Tensor,
    device: torch.device,
    H: int,
    W: int,
    C: int,
    patch: int,
    d_bottleneck: int,
    n_spatial: int,
    d_spatial: int,
    packing_factor: int,
    k_max: int,
    sched: Dict[str, Any],
    ctx_window: int,
    use_amp: bool,
    action_smooth_beta: float,
) -> torch.Tensor:
    B = gt_frames.shape[0]
    rollout_steps = actions.shape[1]
    z0 = encode_initial_latents(
        encoder,
        gt_frames[:, 0],
        device=device,
        C=C,
        H=H,
        W=W,
        patch=patch,
        n_spatial=n_spatial,
        packing_factor=packing_factor,
        use_amp=use_amp,
    )

    z_hist = [z0]
    a_hist = [torch.zeros(B, 16, device=device, dtype=torch.float32)]
    a_smooth = torch.zeros(B, 16, device=device, dtype=torch.float32)
    act_mask_1d = act_mask_1d.to(device).view(1, 16).expand(B, 16)
    actions = actions.to(device).clamp(-1, 1) * act_mask_1d[:, None, :]

    decoded = [decode_batch_like_interactive(
        decoder,
        z0,
        H=H,
        W=W,
        C=C,
        patch=patch,
        packing_factor=packing_factor,
        d_bottleneck=d_bottleneck,
    )]

    beta = min(max(float(action_smooth_beta), 0.0), 0.999)
    for step in range(rollout_steps):
        a_raw = actions[:, step].to(torch.float32)
        if beta > 0.0:
            a_smooth = (beta * a_smooth + (1.0 - beta) * a_raw).to(torch.float32)
            a = a_smooth
        else:
            a = a_raw
        a_hist.append(a)

        g = len(z_hist)
        start = max(0, g - int(ctx_window))
        past = torch.stack(z_hist[start:g], dim=1)
        t = past.shape[1]

        actions_local = torch.zeros((B, t + 1, 16), device=device, dtype=torch.float32)
        if t >= 1:
            actions_local[:, 1:t + 1] = torch.stack(a_hist[start + 1:start + t + 1], dim=1)
        actmask_local = act_mask_1d.view(B, 1, 16).expand(B, t + 1, 16).contiguous()

        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            actions=actions_local,
            act_mask=actmask_local,
            use_amp=use_amp,
        )
        z_next = z_next.detach().view(B, n_spatial, d_spatial)
        z_hist.append(z_next)
        decoded.append(decode_batch_like_interactive(
            decoder,
            z_next,
            H=H,
            W=W,
            C=C,
            patch=patch,
            packing_factor=packing_factor,
            d_bottleneck=d_bottleneck,
        ))

    return torch.stack(decoded, dim=1).cpu()


def main() -> None:
    p = argparse.ArgumentParser(description='Generate MP4 qualitative rollouts using the same inference path as interactive.py.')
    p.add_argument('--tokenizer_ckpt', required=True)
    p.add_argument('--dynamics_ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--frames_dir', required=True)
    p.add_argument('--tasks_json', type=str, default='../tasks.json')
    p.add_argument('--task', type=str, default='pusht')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--start_idx', type=int, nargs='*', default=None)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--rollout_steps', type=int, default=32)
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--packing_factor', type=int, default=2)
    p.add_argument('--ctx_window', type=int, default=24)
    p.add_argument('--schedule', type=str, default='shortcut', choices=['finest', 'shortcut'])
    p.add_argument('--eval_d', type=float, default=0.5)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--action_smooth_beta', type=float, default=0.0)
    p.add_argument('--shard_size', type=int, default=2048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb_project', type=str, default=None)
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--wandb_entity', type=str, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_task(args.data_dir, args.task)
    starts = find_episode_starts(args.data_dir, args.task)
    if args.start_idx:
        ep = raw['episode'].to(torch.int64).cpu()
        chosen = []
        for start in args.start_idx:
            end = int(start) + int(args.rollout_steps)
            if end >= ep.numel() or int(ep[start].item()) != int(ep[end].item()):
                raise RuntimeError(f'--start_idx {start} crosses an episode boundary for rollout_steps={args.rollout_steps}')
            chosen.append(int(start))
        if len(chosen) < args.num_samples:
            reps = int(np.ceil(args.num_samples / max(1, len(chosen))))
            chosen = (chosen * reps)[:args.num_samples]
        else:
            chosen = chosen[:args.num_samples]
    else:
        starts = valid_episode_starts(raw, starts, args.rollout_steps)
        if not starts:
            raise RuntimeError(f'No episode starts with at least {args.rollout_steps} steps for task={args.task}')
        chosen = rng.choice(starts, size=args.num_samples, replace=(args.num_samples > len(starts))).astype(int).tolist()

    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    encoder = tok.encoder
    decoder = tok.decoder
    H = int(tok_info['H'])
    W = int(tok_info['W'])
    C = int(tok_info['C'])
    patch = int(tok_info['patch'])
    n_latents = int(tok_info['n_latents'])
    d_bottleneck = int(tok_info['d_bottleneck'])

    dyn, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt,
        device=device,
        d_bottleneck=d_bottleneck,
        n_latents=n_latents,
        packing_factor=args.packing_factor,
    )
    k_max = int(dyn_info['k_max'])
    n_spatial = int(dyn_info['n_spatial'])
    d_spatial = int(dyn_info['d_spatial'])
    sched = make_tau_schedule(k_max=k_max, schedule=args.schedule, d=(args.eval_d if args.schedule == 'shortcut' else None))

    act_dim = max(0, min(16, load_task_action_dim(args.tasks_json, args.task, default_dim=16)))
    act_mask_1d = torch.zeros(16, dtype=torch.float32)
    if act_dim > 0:
        act_mask_1d[:act_dim] = 1.0

    raw_actions = raw['action'].to(torch.float32).cpu()
    shard_cache = ShardCache(max_items=max(2, args.batch_size * 2))

    wandb_run = None
    if args.wandb_project:
        import wandb
        ckpt_meta = torch.load(args.dynamics_ckpt, map_location='cpu')
        ckpt_step = int(ckpt_meta.get('step', 0)) if isinstance(ckpt_meta, dict) else 0
        del ckpt_meta
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f'interactive-videos-step-{ckpt_step}',
            config={**vars(args), 'checkpoint_step': ckpt_step, 'k_max': k_max},
        )
    else:
        ckpt_meta = torch.load(args.dynamics_ckpt, map_location='cpu')
        ckpt_step = int(ckpt_meta.get('step', 0)) if isinstance(ckpt_meta, dict) else 0
        del ckpt_meta

    saved = []
    metric_rows = []
    for offset in range(0, len(chosen), args.batch_size):
        batch_starts = chosen[offset:offset + args.batch_size]
        gt_batch = []
        act_batch = []
        for start_idx in batch_starts:
            frames = [
                load_frame_cached(args.frames_dir, args.task, start_idx + t, shard_size=args.shard_size, cache=shard_cache)
                for t in range(args.rollout_steps + 1)
            ]
            gt_batch.append(torch.stack(frames, dim=0))
            acts = raw_actions[start_idx + 1:start_idx + 1 + args.rollout_steps]
            act_batch.append(pad_actions(acts, act_dim=act_dim))

        gt = torch.stack(gt_batch, dim=0)
        actions = torch.stack(act_batch, dim=0)
        pred = rollout_interactive_replay(
            encoder=encoder,
            decoder=decoder,
            dyn=dyn,
            gt_frames=gt,
            actions=actions,
            act_mask_1d=act_mask_1d,
            device=device,
            H=H,
            W=W,
            C=C,
            patch=patch,
            d_bottleneck=d_bottleneck,
            n_spatial=n_spatial,
            d_spatial=d_spatial,
            packing_factor=args.packing_factor,
            k_max=k_max,
            sched=sched,
            ctx_window=args.ctx_window,
            use_amp=bool(args.amp),
            action_smooth_beta=args.action_smooth_beta,
        )

        floor = gt[:, 0:1].expand_as(gt).clone()
        mse_pred = (pred[:, 1:].float() - gt[:, 1:].float()).pow(2).mean(dim=(1, 2, 3, 4))
        mse_floor = (floor[:, 1:].float() - gt[:, 1:].float()).pow(2).mean(dim=(1, 2, 3, 4))
        psnr_pred = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
        psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

        for b, start_idx in enumerate(batch_starts):
            sample_id = offset + b
            frames = []
            for t in range(args.rollout_steps + 1):
                frames.append(draw_panel(
                    to_uint8_hwc(gt[b, t]),
                    to_uint8_hwc(pred[b, t]),
                    to_uint8_hwc(floor[b, t]),
                    t=t,
                    sample_idx=sample_id,
                    start_idx=int(start_idx),
                ))
            path = out_dir / f'interactive_replay_{sample_id:03d}_start_{int(start_idx)}.mp4'
            write_mp4(frames, path, fps=args.fps)
            saved.append(str(path))
            row = {
                'sample': sample_id,
                'start_idx': int(start_idx),
                'mse_pred': float(mse_pred[b].item()),
                'mse_floor': float(mse_floor[b].item()),
                'mse_ratio_pred_over_floor': float((mse_pred[b] / mse_floor[b].clamp_min(1e-12)).item()),
                'psnr_pred': float(psnr_pred[b].item()),
                'psnr_floor': float(psnr_floor[b].item()),
                'psnr_gain_over_floor_db': float((psnr_pred[b] - psnr_floor[b]).item()),
                'video': str(path),
            }
            metric_rows.append(row)
            if wandb_run is not None:
                import wandb
                wandb.log({
                    f'video/sample_{sample_id:03d}': wandb.Video(str(path), fps=args.fps, format='mp4'),
                    f'sample/{sample_id:03d}_psnr_pred': row['psnr_pred'],
                    f'sample/{sample_id:03d}_psnr_gain_over_floor_db': row['psnr_gain_over_floor_db'],
                }, step=ckpt_step)

    summary_metrics = {}
    if metric_rows:
        for key in ['mse_pred', 'mse_floor', 'mse_ratio_pred_over_floor', 'psnr_pred', 'psnr_floor', 'psnr_gain_over_floor_db']:
            summary_metrics[key] = float(np.mean([r[key] for r in metric_rows]))

    summary = {
        'checkpoint': args.dynamics_ckpt,
        'checkpoint_step': ckpt_step,
        'tokenizer': args.tokenizer_ckpt,
        'task': args.task,
        'pipeline': 'interactive.py loaders + schedule + sample_one_timestep_packed + decode_single_packed_frame',
        'num_videos': len(saved),
        'videos': saved,
        'metrics': summary_metrics,
        'samples': metric_rows,
        'config': {
            'rollout_steps': args.rollout_steps,
            'fps': args.fps,
            'packing_factor': args.packing_factor,
            'ctx_window': args.ctx_window,
            'schedule': args.schedule,
            'eval_d': args.eval_d,
            'amp': bool(args.amp),
            'action_smooth_beta': args.action_smooth_beta,
            'k_max': k_max,
            'act_dim': act_dim,
        },
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        import wandb
        wandb.log({f'metrics/{k}': v for k, v in summary_metrics.items()}, step=ckpt_step)
        wandb.save(str(out_dir / 'metrics.json'))
        wandb.finish()

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
