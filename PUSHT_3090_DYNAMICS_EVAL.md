# PushT 8x3090 Dynamics Evaluation

Records the evaluation workflow and results for dynamics model checkpoints trained on the PushT adapter dataset.

## Standard Eval Command

```bash
cd /root/Dreamer4-with-LeWM/dreamer4

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/dreamer4/bin/python eval_dynamics_videos.py \
  --tokenizer_ckpt /jerry_slow_vol/dreamer4-runs/pusht-3090/tokenizer_ckpts_b8x8_iid_100k_clean/step_0100000.pt \
  --dynamics_ckpt <CKPT_PATH> \
  --data_dir /root/data/pusht-raw \
  --frames_dir /root/data/pusht-shards \
  --tasks_json /root/Dreamer4-with-LeWM/tasks.json \
  --task pusht \
  --out_dir /root/exports/eval_videos_<TAG> \
  --num_samples 10 \
  --rollout_steps 32 \
  --ctx_window 8 \
  --eval_d 0.25 \
  --fps 10.0 \
  --batch_size 4 \
  --amp \
  --seed 42
```

## Parameter Rationale

| Param | Value | Why |
|---|---|---|
| `rollout_steps` | 32 | 3.2 s at 10 fps — enough to see T-block motion. 16 steps is too short (little motion visible). 64 steps degrades heavily at this training budget. |
| `ctx_window` | 8 | Matches `--eval_ctx 8` default in `train_dynamics.py`. Using 24 (the `interactive.py` default) causes teleporting because the model was never trained with that much context. |
| `eval_d` | 0.25 | Matches `--eval_d 0.25` default in `train_dynamics.py`. Using 0.5 (interactive default) produced worse rollouts. |
| `fps` | 10.0 | Native PushT framerate. |
| `seed` | 42 | Fixed for reproducibility across checkpoint comparisons. |
| `num_samples` | 10 | Sufficient for a quick comparison; increase to 50+ for publication-quality estimates. |

## Note on Official Dreamer4 Defaults

The upstream `train_dynamics.py` training-time eval uses `eval_horizon=16, eval_ctx=8, eval_d=0.25`.
The upstream `interactive.py` demo uses `ctx_window=24, eval_d=0.5` but is open-ended (human-in-the-loop), not a fixed-length rollout.

32 steps is a reasonable middle ground: longer than the training eval baseline (16) but within what the model handles at 40k–80k steps.

## Results — PushT 8x3090, 32-step Rollout

Tokenizer: `tokenizer_ckpts_b8x8_iid_100k_clean/step_0100000.pt` (100k steps, batch 8×8 IID)
Checkpoint dir: `dynamics_ckpts_b28x8_cache256_40k` (batch 28×8, continued to 80k)

| Checkpoint | PSNR pred | PSNR floor | Gain vs frozen | MSE ratio |
|---|---|---|---|---|
| step_0040000 | 31.01 dB | 27.13 dB | +3.88 dB | 0.460 |
| step_0070000 | 30.76 dB | 27.13 dB | +3.63 dB | 0.477 |
| **step_0080000** | **31.65 dB** | **27.13 dB** | **+4.52 dB** | **0.410** |

MSE ratio < 1.0 means the model beats the frozen-first-frame baseline on every checkpoint.
80k is the best checkpoint: 2.4× lower MSE than frozen, +4.52 dB PSNR gain.

## Output Location

Videos and `metrics.json` are written to `/root/exports/` (ephemeral, not on PVC).
Move to PVC only if needed for long-term storage.

```
/root/exports/eval_videos_step40k_32step/   # 10 MP4s + metrics.json
/root/exports/eval_videos_step70k_32step/
/root/exports/eval_videos_step80k_32step/
```

Each MP4 shows three panels side-by-side: **GT frame | model rollout | frozen baseline**.
