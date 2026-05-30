import argparse
import json
import subprocess
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def resize_h5_pixels(pixels_np, target_size):
    frames = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
    if frames.shape[-2:] == (target_size, target_size):
        return frames.to(torch.uint8)
    frames_f = frames.to(torch.float32) / 255.0
    resized = F.interpolate(frames_f, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return (resized.clamp(0, 1) * 255.0).to(torch.uint8)


def load_converted_frames(frames_dir, task, indices, shard_size):
    task_dir = Path(frames_dir) / task
    shards = sorted(task_dir.glob('*shard*.pt'))
    out = []
    cache = {}
    for idx in indices:
        shard_idx = int(idx) // int(shard_size)
        off = int(idx) % int(shard_size)
        if shard_idx not in cache:
            td = torch.load(shards[shard_idx], map_location='cpu', weights_only=False)
            frames = td['frames']
            if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[1] != 3:
                frames = frames.permute(0, 3, 1, 2).contiguous()
            cache[shard_idx] = frames.to(torch.uint8)
        out.append(cache[shard_idx][off])
    return torch.stack(out, dim=0)


def write_mp4(frames, path, fps):
    if not frames:
        raise ValueError('No frames to write')
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p', '-crf', '18', str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode(errors='ignore'))


def chw_u8_to_hwc(x):
    return x.permute(1, 2, 0).cpu().numpy()


def make_panel(orig_128, conv_128, raw_224, diff, *, idx, step, action):
    gap = 6
    header = 30
    footer = 18
    raw = np.array(Image.fromarray(raw_224).resize((128, 128), Image.BILINEAR))
    orig = chw_u8_to_hwc(orig_128)
    conv = chw_u8_to_hwc(conv_128)
    heat = np.zeros_like(conv)
    heat[..., 0] = np.clip(diff * 16, 0, 255).astype(np.uint8)
    heat[..., 1] = heat[..., 0]
    heat[..., 2] = heat[..., 0]
    imgs = [raw, orig, conv, heat]
    labels = ['HDF5 raw 224->128', 'adapter resize', 'Dreamer shard', 'abs diff x16']
    h, w = 128, 128
    out_w = 4 * w + 3 * gap
    out_h = h + header + footer
    canvas = Image.new('RGB', (out_w, out_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(zip(labels, imgs)):
        x0 = i * (w + gap)
        canvas.paste(Image.fromarray(img), (x0, header))
        draw.text((x0 + 3, 7), label, fill=(240, 240, 240))
    draw.text((4, header + h + 2), f'idx={idx} step={step} action=[{action[0]:+.3f},{action[1]:+.3f}]', fill=(230, 230, 230))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='/root/lewm-pusht/pusht_expert_train.h5')
    p.add_argument('--raw', default='/root/data/pusht-raw/pusht.pt')
    p.add_argument('--frames_dir', default='/root/data/pusht-shards')
    p.add_argument('--task', default='pusht')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--start_idx', type=int, default=1587434)
    p.add_argument('--num_frames', type=int, default=64)
    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--target_size', type=int, default=128)
    p.add_argument('--shard_size', type=int, default=2048)
    p.add_argument('--sample_count', type=int, default=128)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = torch.load(args.raw, map_location='cpu', weights_only=False)
    indices = list(range(args.start_idx, args.start_idx + args.num_frames))
    with h5py.File(args.h5, 'r') as f:
        n = int(f['pixels'].shape[0])
        if indices[-1] >= n:
            raise IndexError('requested render indices exceed HDF5 length')
        # Metadata checks over whole tensors where cheap, sampled pixels where expensive.
        meta = {}
        for h5_key, raw_key in [('episode_idx', 'episode'), ('step_idx', 'step_idx'), ('ep_len', 'ep_len'), ('ep_offset', 'ep_offset')]:
            h5_arr = torch.from_numpy(f[h5_key][:])
            raw_arr = raw[raw_key]
            meta[f'{h5_key}_shape_h5'] = list(h5_arr.shape)
            meta[f'{raw_key}_shape_raw'] = list(raw_arr.shape)
            meta[f'{h5_key}_exact_match'] = bool(torch.equal(h5_arr.to(raw_arr.dtype), raw_arr.cpu()))

        h5_action = torch.from_numpy(f['action'][:]).to(torch.float32)
        shifted = torch.zeros_like(h5_action)
        ep_len = raw['ep_len'].to(torch.int64)
        ep_offset = raw['ep_offset'].to(torch.int64)
        for off, length in zip(ep_offset.tolist(), ep_len.tolist()):
            if length > 1:
                shifted[off + 1:off + length] = h5_action[off:off + length - 1]
        action_abs_diff = (shifted - raw['action'].cpu()).abs()
        meta['action_shift_max_abs_diff'] = float(action_abs_diff.max().item())
        meta['action_shift_mean_abs_diff'] = float(action_abs_diff.mean().item())
        meta['episode_first_actions_zero'] = bool(torch.allclose(raw['action'][ep_offset].cpu(), torch.zeros_like(raw['action'][ep_offset].cpu())))

        for key in ['state', 'proprio']:
            if key in f and key in raw:
                h5_arr = torch.from_numpy(f[key][:]).to(torch.float32)
                diff = (h5_arr - raw[key].cpu()).abs()
                meta[f'{key}_max_abs_diff'] = float(diff.max().item())
                meta[f'{key}_mean_abs_diff'] = float(diff.mean().item())

        rng = np.random.default_rng(0)
        sample_idx = sorted(set(indices + rng.choice(n, size=min(args.sample_count, n), replace=False).astype(int).tolist()))
        h5_pixels = f['pixels'][sample_idx]
        h5_resized = resize_h5_pixels(h5_pixels, args.target_size)
        conv = load_converted_frames(args.frames_dir, args.task, sample_idx, args.shard_size)
        pix_diff = (h5_resized.to(torch.int16) - conv.to(torch.int16)).abs()
        meta['pixel_sample_count'] = len(sample_idx)
        meta['pixel_resize_vs_shard_max_abs_diff'] = int(pix_diff.max().item())
        meta['pixel_resize_vs_shard_mean_abs_diff'] = float(pix_diff.float().mean().item())
        meta['pixel_resize_vs_shard_exact_match'] = bool(torch.equal(h5_resized, conv))

        # Render contiguous requested window.
        h5_window = f['pixels'][indices]
        h5_window_resized = resize_h5_pixels(h5_window, args.target_size)
        conv_window = load_converted_frames(args.frames_dir, args.task, indices, args.shard_size)
        diff_window = (h5_window_resized.to(torch.int16) - conv_window.to(torch.int16)).abs().amax(dim=1).cpu().numpy().astype(np.float32)
        panels = []
        for i, idx in enumerate(indices):
            panels.append(make_panel(
                h5_window_resized[i],
                conv_window[i],
                h5_window[i],
                diff_window[i],
                idx=idx,
                step=int(f['step_idx'][idx]),
                action=raw['action'][idx].cpu().numpy(),
            ))

    video_path = out_dir / f'conversion_check_start_{args.start_idx}.mp4'
    write_mp4(panels, video_path, args.fps)
    meta['video'] = str(video_path)
    meta['window_start_idx'] = int(args.start_idx)
    meta['window_num_frames'] = int(args.num_frames)
    meta['h5'] = args.h5
    meta['raw'] = args.raw
    meta['frames_dir'] = args.frames_dir
    (out_dir / 'conversion_metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
