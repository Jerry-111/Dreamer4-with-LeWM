# PushT Adapter Notes

This commit adds a LeWM PushT conversion path without replacing Dreamer4's
existing dataloaders.

The original problem was a data-format mismatch. Dreamer4's training pipeline
expects preprocessed frame shards plus a separate raw `.pt` file containing
episode/action/reward tensors. The LeWM PushT dataset is distributed as a single
HDF5 file with `pixels`, `action`, `episode_idx`, `step_idx`, `ep_len`,
`ep_offset`, `state`, and `proprio`. Feeding that HDF5 file directly would have
required changing Dreamer4's dataloaders and risked disrupting tokenizer,
dynamics, and evaluation paths.

We solved this by writing a standalone adapter script. It converts the PushT
HDF5 file into the same filesystem layout and tensor schema that Dreamer4
already knows how to load. This keeps the core pipeline intact: tokenizer
training still uses `ShardedFrameDataset`, and action-conditioned dynamics
training still uses `WMDataset`.

## Output Contract

The converter reads:

```text
/root/lewm-pusht/pusht_expert_train.h5
```

and writes:

```text
/root/data/pusht-raw/pusht.pt
/root/data/pusht-shards/pusht/pusht_shard0000.pt
...
```

Frame shards are saved as:

```python
{"frames": UInt8Tensor[N, 3, 128, 128]}
```

Raw metadata is saved as:

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

## Decisions

- Keep Dreamer4's existing loaders intact. PushT is adapted into the existing
  `ShardedFrameDataset` and `WMDataset` formats instead of adding an HDF5
  training loader.
- Resize PushT pixels from `(N, 224, 224, 3)` HWC to `(N, 3, 128, 128)` CHW
  uint8 shards.
- Use fixed 2048-frame shards, matching `WMDataset`'s default `shard_size`.
- Shift actions per episode so `dreamer_action[t + 1] = h5_action[t]`.
  The first action of each episode is zero. This matches Dreamer4's convention
  that the action at index `i` produced observation `i`.
- Fill rewards with zeros because LeWM PushT does not include rewards and the
  current dynamics loss does not train on rewards.
- Save `state` and `proprio` for debugging, although Dreamer4 currently ignores
  extra keys in `pusht.pt`.
- Import `hdf5plugin` in the converter because the HDF5 `pixels` dataset uses a
  compression filter that plain `h5py` cannot read.

## Training Integration

`train_tokenizer.py` and `train_dynamics.py` now support:

```bash
--tasks pusht
```

If `--tasks` is omitted, both scripts still use the original `TASK_SET`.
`tasks.json` includes:

```json
"pusht": {
  "action_dim": 2
}
```

Tokenizer training uses only frame shards:

```bash
torchrun --nproc_per_node=1 train_tokenizer.py \
  --tasks pusht \
  --data_dirs /root/data/pusht-shards
```

Dynamics training uses frame shards plus raw action metadata:

```bash
torchrun --nproc_per_node=1 train_dynamics.py \
  --use_actions \
  --tasks pusht \
  --data_dirs /root/data/pusht-raw \
  --frame_dirs /root/data/pusht-shards \
  --tasks_json ../tasks.json \
  --tokenizer_ckpt ./logs/tokenizer_ckpts/latest.pt
```

## Validation

The full conversion produced:

```text
2,336,736 raw rows
1,141 frame shards
final shard size: 2,016 frames
```

Both Dreamer4 datasets instantiate on the converted output. `WMDataset` reports
`action_dim=2` and builds an action mask of `[1, 1, 0, ..., 0]` for PushT.
