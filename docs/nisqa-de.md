# NISQA_DE JAX port

This branch adds a native JAX implementation of upstream double-ended `NISQA_DE` without changing the artifact schema or inference path used by the three already-qualified single-ended checkpoints.

## Source contract

Source architecture is pinned to `gabrielmittag/NISQA` revision `fe84f0f252abec382b24367d5b22498a7ce34dbb`. The checkpoint converter accepts the canonical graph documented by upstream `config/train_nisqa_double_ended.yaml`:

- adaptive shared CNN
- shared first self-attention network: 64 dimensions, one head, two layers
- cosine alignment with hard application
- `x/y/-` fusion without learned projection
- second self-attention network: 64 dimensions, one head, two layers
- attention pooling
- fullband frontend profile: 48 mel bands, 15-bin segments, hop 4, maximum 1300 segments

The runtime alignment/fusion primitives also implement upstream Bahdanau, Luong, dot, cosine, distance, soft/hard application, all three fusion formulas, and optional fusion projection. The canonical checkpoint converter intentionally rejects graph drift instead of guessing how to map an arbitrary custom training configuration.

## API

```python
import jax
from nisqa_jax import NisqaDeJaxModel, convert_double_ended_checkpoint

cfg, params, source_sha256 = convert_double_ended_checkpoint("my_nisqa_de.tar")
model = NisqaDeJaxModel(cfg, params, jax.devices("cpu")[0])

# segments: [batch, steps, 2, 48, 15]
# n_wins:   [batch, 2] degraded/reference valid segment counts
mos = model(segments, n_wins)
```

Conversion requires optional PyTorch dependency only to safely read original `.tar` and extract tensors. Returned model graph itself is JAX-native.

## Evidence

### Observed

- A bounded isolated PyTorch/JAX operator suite executed during development: 16/16 passed.
- It covered five alignment score mechanisms, hard/soft masked application for parameter-free alignments, three fusion formulas with/without projection, and gradient parity for differentiable soft dot alignment.
- Repository tests check exact source-state key coverage, target shape coverage, architecture drift rejection, masked alignment behavior, complete default graph finite forward, and a nonzero finite gradient through the composed graph.
- Existing single-ended artifacts remain on their original v4 conversion path.

### Not established

Upstream publishes a NISQA_DE training recipe but no pretrained double-ended checkpoint. Therefore this work does **not** claim:

- parity against a real pretrained NISQA_DE checkpoint
- MOS/task metric equivalence on a double-ended evaluation corpus
- long-training trajectory or optimizer-state parity
- CUDA/TPU performance or numerical parity for NISQA_DE
- support for arbitrary NISQA_DE training configurations

A trusted canonical user-trained checkpoint can close real-checkpoint inference parity without changing the architecture implementation. Full training/fine-tuning is a separate porting milestone because it also requires optimizer, scheduler, dropout RNG, data pipeline, checkpoint-resume, loss, and trajectory parity.
