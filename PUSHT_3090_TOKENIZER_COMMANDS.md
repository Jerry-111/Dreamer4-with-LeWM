# PushT 8x3090 Tokenizer Commands

Brief record of tokenizer probes and the clean production command used on this pod.

## Current Code State

Only the shard loading change is kept in code:

```python
torch.load(path, map_location="cpu", mmap=True)
```

The tokenizer script is restored to its original sampling behavior: `iid_sampling=True` in the dataset and `shuffle=True` in `DistributedSampler`. The temporary CLI flags for sampling/prefetch were removed.

## Safety Notes

- CPU RAM looked safe with original iid/shuffled sampling and `num_workers=1`: about 32-33 GiB used out of 755 GiB during smoke/full startup.
- `batch_size=10` per GPU OOMed on 24 GiB 3090s.
- `batch_size=8` per GPU is the clean production setting on 8x3090s.
- Use `max_steps=100001` with `save_every=5000` if we want `step_0100000.pt`, because checkpointing happens before `step += 1`.
- Since `train_tokenizer.py` was restored, the old outer loop behavior is back: after reaching `max_steps`, the run may need to be stopped manually, but `step_0100000.pt` and `latest.pt` should already be written.

## Clean Smoke Test

This used original iid/shuffled sampling. It reached step 15 quickly with stable GPU and CPU memory.

```bash
cd /root/Dreamer4-with-LeWM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/jerry_slow_vol/dreamer4-runs/pusht-3090/logs/wandb
/opt/conda/envs/dreamer4/bin/torchrun --standalone --nproc_per_node=8 dreamer4/train_tokenizer.py \
  --tasks pusht \
  --data_dirs /root/data/pusht-shards \
  --wandb_entity jec083-uc-san-diego \
  --wandb_run_name pusht-tokenizer-3090-b8x8-iid-smoke-clean \
  --batch_size 8 \
  --num_workers 1 \
  --max_steps 20 \
  --save_every 1000 \
  --log_every 5 \
  --print_every 5 \
  --viz_every 0 \
  --ckpt_dir /jerry_slow_vol/dreamer4-runs/pusht-3090/tokenizer_smoke_b8x8_iid_clean
```

## OOM Probe

Do not use this for training; `batch_size=10` per GPU OOMed.

```bash
/opt/conda/envs/dreamer4/bin/torchrun --standalone --nproc_per_node=8 dreamer4/train_tokenizer.py \
  --tasks pusht \
  --data_dirs /root/data/pusht-shards \
  --wandb_entity jec083-uc-san-diego \
  --wandb_run_name pusht-tokenizer-3090-b10x8-oom-probe \
  --batch_size 10 \
  --num_workers 1 \
  --max_steps 20 \
  --save_every 0 \
  --log_every 5 \
  --print_every 1 \
  --viz_every 0 \
  --ckpt_dir /jerry_slow_vol/dreamer4-runs/pusht-3090/tokenizer_ckpts_b10x8_oom_probe
```

## Production Run

Clean run, no resume, original iid/shuffled sampling, only mmap shard loading changed.

```bash
setsid bash -lc 'cd /root/Dreamer4-with-LeWM && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export WANDB_DIR=/jerry_slow_vol/dreamer4-runs/pusht-3090/logs/wandb && \
  exec /opt/conda/envs/dreamer4/bin/torchrun --standalone --nproc_per_node=8 dreamer4/train_tokenizer.py \
    --tasks pusht \
    --data_dirs /root/data/pusht-shards \
    --wandb_entity jec083-uc-san-diego \
    --wandb_run_name pusht-tokenizer-3090-b8x8-iid-100k-clean \
    --batch_size 8 \
    --num_workers 1 \
    --max_steps 100001 \
    --save_every 5000 \
    --log_every 100 \
    --print_every 100 \
    --viz_every 500 \
    --ckpt_dir /jerry_slow_vol/dreamer4-runs/pusht-3090/tokenizer_ckpts_b8x8_iid_100k_clean \
    > /jerry_slow_vol/dreamer4-runs/pusht-3090/logs/tokenizer_b8x8_iid_100k_clean.log 2>&1'
```

Key paths:

```text
checkpoints: /jerry_slow_vol/dreamer4-runs/pusht-3090/tokenizer_ckpts_b8x8_iid_100k_clean
log:         /jerry_slow_vol/dreamer4-runs/pusht-3090/logs/tokenizer_b8x8_iid_100k_clean.log
data:        /root/data/pusht-shards
```
