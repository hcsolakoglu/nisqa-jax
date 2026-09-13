# Architecture

## Scope

NISQA-JAX is an inference-only port for the three shipped upstream checkpoints
plus the canonical double-ended `NISQA_DE` graph documented by upstream
`config/train_nisqa_double_ended.yaml`:

| Artifact / graph | Validated architecture | Semantic outputs |
|---|---|---|
| `nisqa.npz` | adaptive CNN, self-attention, five attention-pooling heads | `mos`, `noi`, `dis`, `col`, `loud` |
| `nisqa_mos_only.npz` | adaptive CNN, self-attention, one attention-pooling head | `mos` |
| `nisqa_tts.npz` | standard CNN, bidirectional LSTM, last-step-bi pooling | `naturalness` |
| canonical `NISQA_DE` source checkpoint | shared adaptive CNN + self-attention, cosine hard alignment, `x/y/-` fusion, second self-attention, attention pool | MOS array |

Training, fine-tuning, dataset evaluation, arbitrary double-ended variants,
multi-head variants outside the qualified profiles, and arbitrary checkpoint
architectures remain outside this contract and fail validation when encountered.
Upstream ships no pretrained `NISQA_DE` checkpoint, so double-ended support is
architecture/operator/conversion tested rather than a real-pretrained-checkpoint
or perceptual-task parity claim.

## Data flow

Single-ended shipped models:

```text
audio path
  -> Librosa/SoundFile decode and cached-mel-basis spectrogram on host
  -> checkpoint-specific overlapping segment extraction
  -> length estimation, stable sorting, chunking, and bucket padding
  -> explicit device placement
  -> one jitted JAX model forward
  -> one intentional device-to-host result transfer
  -> semantic dictionary or PyTorch-compatible DataFrame/CSV formatting
```

Canonical double-ended model:

```text
degraded path + reference path
  -> independent upstream-compatible mel extraction and segmentation
  -> pad both segment sequences to their pair-local maximum
  -> concatenate degraded/reference on CNN channel axis
  -> shared CNN and first time-dependency network on each side
  -> mask reference tail and align it to degraded queries
  -> fuse degraded + aligned-reference features
  -> second time-dependency network
  -> attention pool -> MOS
```

Preprocessing remains on the host to preserve upstream Librosa behavior.
`predict_batch` parallelizes single-ended preprocessing, keeps row identity,
groups similar lengths, and pads only at batch assembly. Single-ended model
inputs are `[batch, steps, 1, n_mels, segment_length]` plus one `int32`
valid-window count per sample.

`preprocess_pair` segments degraded and reference audio independently and
`pack_double_ended_segments` produces
`[batch, steps, 2, n_mels, segment_length]` plus two independent counts in
`[batch, 2]`, degraded first and reference second. Static JAX padding is masked
so invalid reference tail values cannot influence valid alignment or final MOS.

## Components

| Module | Responsibility |
|---|---|
| `nisqa_jax/config.py` | Immutable single-ended model/feature configuration and exact supported-architecture validation |
| `nisqa_jax/features.py` | Audio loading, channel validation, mel extraction, segmentation, and cheap length estimation |
| `nisqa_jax/checkpoint.py` | Shipped single-ended checkpoint conversion, artifact/metadata validation, model loading, cache configuration, and prewarming |
| `nisqa_jax/model.py` | Shared pure-JAX CNN, attention, pooling, and LSTM primitives plus single-ended forward graph |
| `nisqa_jax/double_ended.py` | Native JAX alignment, fusion, complete `NISQA_DE` forward graph, and JIT wrapper |
| `nisqa_jax/double_ended_checkpoint.py` | Fail-closed canonical `NISQA_DE` source-state conversion and exhaustive shape/key validation |
| `nisqa_jax/double_ended_features.py` | Canonical DE frontend contract, independent pair preprocessing, and pair packing |
| `nisqa_jax/predict.py` | Single-ended file/batch APIs, CLI modes, error collection, length-aware scheduling, and OOM retry |
| `nisqa_jax/bench.py` | JAX-only synthetic or end-to-end benchmark |
| `nisqa_jax/bench_compare.py` | Hash-bound JAX/PyTorch model-forward comparison |
| `scripts/benchmark_hf_real.py` | Bounded Hugging Face streaming, real-audio framework benchmark, parity, and profiling |
| `nisqa_jax/weights/` | Bundled single-ended artifacts, sidecar metadata, checksums, and model-weight license |

The package avoids a mandatory Flax or PyTorch dependency for normal runtime.
Parameters are functional pytrees owned by thin `NisqaJaxModel` or
`NisqaDeJaxModel` wrappers. Source `.tar` conversion is optional and uses
PyTorch only as a restricted tensor reader.

## Artifact contract

Normal shipped-model inference loads a converted `.npz` plus JSON metadata. The
runtime loader validates:

- the artifact SHA-256 against the sidecar's embedded hash;
- the metadata checksum;
- the complete tensor-name, shape, and dtype manifest;
- source provenance fields;
- the exact model architecture and output-name contract.

The separate `scripts/verify_artifacts.py --strict` release/CI gate checks the
bundled `CHECKSUMS.sha256` catalog and rejects unknown artifacts. This second
layer is not consulted by `load_converted_checkpoint`.

Canonical `NISQA_DE` conversion is deliberately separate from the v4
single-ended `.npz` artifact schema. `convert_double_ended_checkpoint` accepts
only the pinned canonical upstream graph, accounts for every required source
tensor, permits only BatchNorm `num_batches_tracked` as non-model state,
transposes convolution/linear layouts, folds evaluation BatchNorm into the CNN,
and verifies every target leaf shape. Graph drift fails closed rather than being
guessed. Upstream provides no pretrained DE artifact, so user-trained source
checkpoints remain trusted external inputs rather than bundled release assets.

## JIT shapes and batching

JAX compilation keys include the effective batch and padded sequence shapes.
To limit shape proliferation for shipped single-ended models:

- preprocessing returns only real segments;
- each chunk is padded to its real maximum rounded to a configurable bucket;
- self-attention defaults to a 32-step grid and TTS to a 64-step grid;
- `prewarm` can compile a bounded geometric grid into a persistent cache;
- `--auto_batch` halves the batch only for recognized device OOM errors.

Double-ended pair preprocessing pads only to the local maximum of degraded and
reference segment counts. The current DE API is explicit rather than integrated
into the single-ended batch scheduler, so no DE throughput or bucket-size claim
is made.

The persistent cache is process-global and trusted executable state. A process
must use one consistent, access-controlled cache directory.

## Numerical behavior

Strict conformance uses float32 inputs and reductions. The model wraps forward
computation in `jax.default_matmul_precision("float32")`; numerically sensitive
LayerNorm, attention, and pooling reductions accumulate in float32. `bf16` is
an opt-in compute mode with float32 reductions and separate drift tests for the
shipped single-ended paths.

The PyTorch CPU implementation is the live parity reference. The bundled golden
fixtures contain independently hashed PyTorch outputs so CI can enforce shipped
single-ended parity without installing PyTorch. Development of the DE slice also
used bounded direct PyTorch/JAX operator and gradient comparisons; because no
upstream pretrained DE checkpoint exists, DE does not inherit the shipped
checkpoint golden-evidence level.

## Public surfaces

Single-ended library entry points include:

- `load_model`
- `predict_file`
- `predict_batch`
- `prewarm`
- `convert_checkpoint`
- `load_converted_checkpoint`

Canonical double-ended entry points include:

- `DoubleEndedConfig`
- `NisqaDeJaxModel`
- `forward_double_ended`
- `convert_double_ended_checkpoint`
- `canonical_double_ended_feature_config`
- `pack_double_ended_segments`
- `preprocess_pair`

The installed `nisqa-jax` command remains focused on shipped single-ended
`predict_file`, `predict_dir`, and `predict_csv` modes. Dictionary results use
semantic names; DataFrame/CSV output uses upstream-compatible `*_pred` columns
and a `model` column. The DE path is currently an explicit Python API rather
than an implied CLI compatibility promise.

NISQA-JAX is pre-1.0. New public APIs should be added only with a demonstrated
consumer, tests, documentation, and a changelog entry.
