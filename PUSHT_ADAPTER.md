# PushT Adapter Implementation Report

## Objective

The goal was to train the Dreamer4 PyTorch implementation on the LeWM PushT
dataset while preserving Dreamer4's existing tokenizer and dynamics training
pipeline. The main engineering task was to bridge the dataset-format mismatch
between LeWM PushT and Dreamer4.

Dreamer4 expects data in two pieces:

- frame shards for tokenizer training and frame loading:
  `root/<task>/<task>_shardXXXX.pt`
- raw transition metadata for action-conditioned dynamics training:
  `root/<task>.pt`

LeWM PushT is distributed as a single HDF5 file:

```text
/root/lewm-pusht/pusht_expert_train.h5
```

with keys:

```text
pixels, action, episode_idx, step_idx, ep_len, ep_offset, state, proprio
```

Instead of rewriting Dreamer4's dataloaders to consume HDF5 directly, we added a
standalone conversion adapter that emits Dreamer4-native data.

## Design Rationale

The adapter approach was chosen to minimize disruption to Dreamer4.

- `ShardedFrameDataset` can continue to load image sequences for tokenizer
  training.
- `WMDataset` can continue to load aligned observation/action/reward sequences
  for dynamics training.
- Evaluation and checkpointing code remain unchanged.
- Future upstream Dreamer4 changes are easier to merge because the core
  dataloaders were not replaced.

The tradeoff is storage: PushT's compressed HDF5 expands into approximately
107 GiB of uncompressed `uint8` frame shards after conversion to Dreamer4's
format.

## Implemented Changes

Added:

```text
dreamer4/convert_lewm_pusht.py
```

The converter reads the LeWM PushT HDF5 file and writes:

```text
/root/data/pusht-raw/pusht.pt
/root/data/pusht-shards/pusht/pusht_shard0000.pt
...
```

Each frame shard contains:

```python
{"frames": UInt8Tensor[N, 3, 128, 128]}
```

The raw metadata file contains:

```python
{
    "episode": LongTensor[N],
    "action": FloatTensor[N, 2],
    "reward": FloatTensor[N],
    "step_idx": LongTensor[N],
    "ep_len": IntTensor[num_episodes],
    "ep_offset": LongTensor[num_episodes],
    "state": FloatTensor[N, 7],
    "proprio": FloatTensor[N, 4],
}
```

Additional integration changes:

- Added `--tasks` to `train_tokenizer.py` and `train_dynamics.py` so PushT can
  be selected without modifying the global `TASK_SET`.
- Added PushT metadata to `tasks.json`:

```json
"pusht": {
  "action_dim": 2
}
```

- Added `h5py` and `hdf5plugin` to `environment.yaml`. `hdf5plugin` is needed
  because the HDF5 `pixels` dataset uses a compression filter that plain `h5py`
  cannot read.

## Alignment Decisions

Pixel alignment:

- LeWM PushT stores pixels as `(N, 224, 224, 3)` HWC `uint8`.
- Dreamer4 expects `(N, 3, 128, 128)` CHW `uint8`.
- The adapter resizes with bilinear interpolation and saves fixed 2048-frame
  shards, matching `WMDataset`'s default `shard_size`.

Action alignment:

- LeWM PushT actions are stored as `(N, 2)`.
- Dreamer4's `WMDataset` assumes `action[i]` is the action that produced
  observation `i`.
- The adapter shifts actions within each episode:

```python
dreamer_action[t + 1] = h5_action[t]
```

- The first action in each episode is set to zero.
- `WMDataset` pads actions to 16 dimensions internally and uses the PushT
  `action_dim=2` metadata to create the action mask.

Reward handling:

- LeWM PushT does not provide rewards.
- The adapter writes zero rewards because the current Dreamer4 dynamics loss
  does not train on rewards.
- This keeps `WMDataset` compatible without introducing task-specific reward
  reconstruction logic.

Extra state:

- `state` and `proprio` are saved in `pusht.pt` for debugging and possible
  future use.
- Dreamer4 currently ignores these extra keys.

## Validation

The full conversion produced:

```text
raw rows:          2,336,736
frame shards:      1,141
final shard size:  2,016 frames
raw metadata size: 161 MiB
frame shard size:  107 GiB
```

Loader checks passed:

- `ShardedFrameDataset` instantiated on `/root/data/pusht-shards`.
- `WMDataset` instantiated on `/root/data/pusht-raw` plus
  `/root/data/pusht-shards`.
- `WMDataset` reported:

```text
valid sequences: 1,738,816
action_dim:      2
action mask:     [1, 1, 0, ..., 0]
```

Tokenizer training was started on a single RTX 3090. Early metrics looked
healthy:

```text
step 0:    loss=0.611, mse=0.422, lpips=0.944, psnr=3.74
step 100:  loss=0.043, mse=0.009, lpips=0.169, psnr=20.47
step 1000: loss≈0.040, mse≈0.0088, psnr≈20.55
```

W&B system metrics show high GPU utilization and stable memory usage. The latent
standard deviation metric stayed near 1.0, suggesting the tokenizer latents have
not collapsed.

## Current Training Plan

Tokenizer training should be run with an explicit step budget. The default
Dreamer4 value is `10,000,000` steps, which is effectively open-ended for this
experiment.

Recommended tokenizer target:

```bash
cd /root/dreamer4/dreamer4

torchrun --nproc_per_node=1 train_tokenizer.py \
  --tasks pusht \
  --data_dirs /root/data/pusht-shards \
  --wandb_entity jerrychsh-ucsd \
  --wandb_run_name pusht-tokenizer-90k \
  --batch_size 8 \
  --num_workers 8 \
  --save_every 5000 \
  --max_steps 90000
```

After tokenizer training produces:

```text
./logs/tokenizer_ckpts/latest.pt
```

run action-conditioned dynamics training:

```bash
torchrun --nproc_per_node=1 train_dynamics.py \
  --use_actions \
  --tasks pusht \
  --data_dirs /root/data/pusht-raw \
  --frame_dirs /root/data/pusht-shards \
  --tasks_json ../tasks.json \
  --tokenizer_ckpt ./logs/tokenizer_ckpts/latest.pt \
  --wandb_entity jerrychsh-ucsd \
  --wandb_run_name pusht-dynamics \
  --batch_size 24 \
  --num_workers 8 \
  --save_every 10000
```

## Open Questions / Next Checks

- Inspect tokenizer reconstruction visualizations, not just scalar metrics,
  because PushT has a large white background and MSE/PSNR may overstate quality.
- Decide whether to keep converted data on `/root/data` or move the canonical
  copy to a PVC such as `/jerry_slow_vol` for durability.
- After dynamics training starts, monitor action-conditioning usefulness via
  Dreamer4's action-shuffle loss ratio.
- If GPU memory becomes unstable on the 3090, reduce tokenizer `batch_size` from
  8 to 4, or dynamics `batch_size` from 24 to a smaller value.
