# Architecture

## Scope

NISQA-JAX is an inference-only port for exactly three upstream checkpoints:

| Artifact | Validated architecture | Semantic outputs |
|---|---|---|
| `nisqa.npz` | adaptive CNN, self-attention, five attention-pooling heads | `mos`, `noi`, `dis`, `col`, `loud` |
| `nisqa_mos_only.npz` | adaptive CNN, self-attention, one attention-pooling head | `mos` |
| `nisqa_tts.npz` | standard CNN, bidirectional LSTM, last-step-bi pooling | `naturalness` |

Training, fine-tuning, dataset evaluation, double-ended `NISQA_DE`,
multi-head attention, and arbitrary checkpoint architectures are outside this
contract and fail validation when encountered.

## Data flow

```text
audio path
  -> Librosa/SoundFile decode and mel spectrogram on host
  -> checkpoint-specific overlapping segment extraction
  -> length estimation, stable sorting, chunking, and bucket padding
  -> explicit device placement
  -> one jitted JAX model forward
  -> one intentional device-to-host result transfer
  -> semantic dictionary or PyTorch-compatible DataFrame/CSV formatting
```

Preprocessing remains on the host to preserve upstream Librosa behavior.
`predict_batch` parallelizes preprocessing, keeps row identity, groups similar
lengths, and pads only at batch assembly. The model receives
`[batch, steps, 1, n_mels, segment_length]` plus an `int32` valid-window count
for each sample.

## Components

| Module | Responsibility |
|---|---|
| `nisqa_jax/config.py` | Immutable model/feature configuration and exact supported-architecture validation |
| `nisqa_jax/features.py` | Audio loading, channel validation, mel extraction, segmentation, and cheap length estimation |
| `nisqa_jax/checkpoint.py` | Trusted checkpoint conversion, artifact/metadata validation, model loading, cache configuration, and prewarming |
| `nisqa_jax/model.py` | Pure-JAX CNN, attention, pooling, and LSTM forward graph |
| `nisqa_jax/predict.py` | File/batch APIs, CLI modes, error collection, length-aware scheduling, and OOM retry |
| `nisqa_jax/bench.py` | JAX-only synthetic or end-to-end benchmark |
| `nisqa_jax/bench_compare.py` | Hash-bound JAX/PyTorch model-forward comparison |
| `nisqa_jax/weights/` | Bundled artifacts, sidecar metadata, checksums, and model-weight license |

The package avoids a mandatory Flax or PyTorch dependency. Parameters are
functional pytrees owned by a thin `NisqaJaxModel` wrapper.

## Artifact contract

Normal inference loads a converted `.npz` plus JSON metadata. The runtime
loader validates:

- the artifact SHA-256 against the sidecar's embedded hash;
- the metadata checksum;
- the complete tensor-name, shape, and dtype manifest;
- source provenance fields;
- the exact model architecture and output-name contract.

The separate `scripts/verify_artifacts.py --strict` release/CI gate checks the
bundled `CHECKSUMS.sha256` catalog and rejects unknown artifacts. This second
layer is not consulted by `load_converted_checkpoint`.

Source `.tar` conversion is optional and requires the `convert` extra. It is a
trusted-data operation, not the normal inference path.

## JIT shapes and batching

JAX compilation keys include the effective batch and padded sequence shapes.
To limit shape proliferation:

- preprocessing returns only real segments;
- each chunk is padded to its real maximum rounded to a configurable bucket;
- self-attention defaults to a 32-step grid and TTS to a 64-step grid;
- `prewarm` can compile a bounded geometric grid into a persistent cache;
- `--auto_batch` halves the batch only for recognized device OOM errors.

The persistent cache is process-global and trusted executable state. A process
must use one consistent, access-controlled cache directory.

## Numerical behavior

Strict conformance uses float32 inputs and reductions. The model wraps forward
computation in `jax.default_matmul_precision("float32")`; numerically sensitive
LayerNorm, attention, and pooling reductions accumulate in float32. `bf16` is
an opt-in compute mode with float32 reductions and separate drift tests.

The PyTorch CPU implementation is the live parity reference. The bundled golden
fixtures contain independently hashed PyTorch outputs so CI can enforce parity
without installing PyTorch.

## Public surfaces

The supported library entry points are:

- `load_model`
- `predict_file`
- `predict_batch`
- `prewarm`
- `convert_checkpoint`
- `load_converted_checkpoint`

The installed `nisqa-jax` command supports `predict_file`, `predict_dir`, and
`predict_csv`. Dictionary results use semantic names; DataFrame/CSV output uses
upstream-compatible `*_pred` columns and a `model` column.

NISQA-JAX is pre-1.0. New public APIs should be added only with a demonstrated
consumer, tests, documentation, and a changelog entry.
