# NISQA-JAX

Standalone [JAX](https://github.com/google/jax) inference port for the three shipped [NISQA](https://github.com/gabrielmittag/NISQA) speech quality assessment checkpoints. No PyTorch dependency at inference time — runs on CPU, GPU, and TPU.

**~3× faster than eager PyTorch** on the self-attention model (model-forward, RTX 3070); the BiLSTM (TTS) model is roughly on par with optimized PyTorch. Numerical parity at ~1e-7 (100× tighter than the 1e-3 acceptance threshold). See the [Benchmark](#benchmark-jax-gpu-vs-pytorch-gpu-rtx-3070) section for the full optimized-PyTorch comparison.

## Checkpoints

| Artifact | Architecture | Outputs | Source |
|---|---|---|---|
| `nisqa_mos_only.npz` | adapt-CNN + self-attention + att-pool | `mos` | `nisqa_mos_only.tar` |
| `nisqa.npz` | adapt-CNN + self-attention + att-pool | `mos, noi, dis, col, loud` | `nisqa.tar` (NISQA_DIM) |
| `nisqa_tts.npz` | standard-CNN + BiLSTM + last-step-bi | `naturalness` | `nisqa_tts.tar` |

Pre-converted `.npz` weights ship in `weights/` — zero-config inference after clone.

## Benchmark (JAX-GPU vs PyTorch-GPU, RTX 3070)

Measured against **optimized PyTorch** (real `.tar` weights from
[gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA), `cudnn.benchmark=True`,
eval + `no_grad`, TF32 off) — not just eager. Warm model-forward latency, median
of 60 timed iterations after warmup (compile excluded), inputs pre-staged on-GPU
(compute-only; host↔device transfer is framework-independent overhead). Strict
float32 for both frameworks, matching the port's `default_matmul_precision("float32")`.
Grid: batch ∈ {1, 8, 16} × steps ∈ {64, 256, 512}. Reproducible harness:
`bench_jax.py`, `bench_pt.py`, `bench_pt_graphs.py`, `bench_combine.py`.

| Model | PT eager | PT cuda-graphs | PT compile (dynamic / reduce-overhead / max-autotune) | Parity vs eager |
|---|---|---|---|---|
| mos (self-att) | **3.11×** (2.84–4.13) | **2.27×** (1.19–2.83) | BLOCKED in this env¹ | 2.4e-7 |
| tts (BiLSTM) | 0.96× (0.49–1.51) | n/a² | 0.97× / 0.97× / 0.97× (0.49–1.57) | 5.5e-6 |

Speedup = PT_latency / JAX_latency (>1 = JAX faster), geomean over the 9-shape grid.

**Headline correction.** The previous "3–7× faster than PyTorch" was measured
against *eager* PyTorch and held only for the self-attention model. Against
**optimized** PyTorch the self-attention speedup is ~3.1× (eager) / ~2.3× (CUDA
graphs), and the BiLSTM (TTS) model is **not** faster — it is roughly on par with
PyTorch's cuDNN LSTM (faster at large batch, up to ~2× slower on long single
sequences where `jax.lax.scan` cannot match cuDNN's fused LSTM). `torch.compile`
does not materially change TTS latency (cuDNN LSTM is already optimized).

¹ `torch.compile` (inductor / triton 3.6.0) fails on the self-attention model in
this environment with `RuntimeError: CUDA driver error: invalid argument`
launching a fused triton softmax kernel (RTX 3070, driver 595.84, sm_86); verified
across `dynamic=True`, `reduce-overhead`, `max-autotune`, and several inductor
config workarounds (`split_reductions=False`, `autotune_fallback_to_aten=True`,
mask-skip bypass). A triton-free manual CUDA-graph mode is reported instead. The
TTS/LSTM model compiles fine (no fused softmax kernel).
² cuDNN LSTM cannot be captured into a CUDA graph; TTS uses the `torch.compile` modes instead.

JAX bf16 vs PyTorch fp16 autocast (eager): **5.8×–6.5×** on self-attention. Full
benchmark results and the adversarial review in [`adversarial_review/`](adversarial_review/).

## GPU Memory & Batch Size

Peak memory scales with `batch_size × seq_len`. The table below lists the **largest
batch size that runs without a fatal OOM at the checkpoint's full sequence length**
(`max_segments=1300` for self-attention, `6000` for TTS), measured on an 8 GB RTX 3070
with JAX as the sole GPU consumer (see
[`adversarial_review/probes/probe_oom_boundary2.py`](adversarial_review/probes/probe_oom_boundary2.py)):

| Checkpoint | Full seq len | Max bs on 8 GB (measured) | Fails at |
|---|---|---|---|
| `nisqa_mos_only` (self-att, 1-out) | 1300 | **32** | 48 |
| `nisqa` (self-att, 5-out / DIM) | 1300 | **16** | 24 |
| `nisqa_tts` (BiLSTM) | 6000 | **8** | 12 |

Those are ceilings with the whole GPU available to JAX. **Recommended** batch sizes
leave headroom for the OS / display and for length-padding waste in mixed-length
batches, and scale linearly with GPU memory:

| GPU memory | self-att (`--bs`) | TTS (`--bs`) |
|---|---|---|
| ≤ 8 GB | 4–8 | 2–4 |
| 12 GB | 8–16 | 4–8 |
| 24 GB | 16–32 | 8–16 |

TTS is heavier (10× longer sequence → ~10× the LSTM activations). Real-world audio is
usually shorter than the full sequence length, so larger batches fit in practice; the
numbers above are worst-case full-length bounds.

If a batch is too large for the available memory, pass `--auto_batch` to recover
automatically (see below).

## Install

**CPU** (validation, development, CPU inference):
```bash
pip install -e .
# or pin the exact tested versions:
pip install -r requirements-jax.txt
```

**NVIDIA GPU** (CUDA 12) — install the CUDA JAX meta-package instead of plain `jax`:
```bash
pip install -e .
pip install "jax[cuda12_pip]==0.4.30"  # see https://docs.jax.dev/en/latest/installation.html
```

> **JAX version pin.** `jax` is pinned to the tested `0.4.x` minor (`jax>=0.4.30,<0.5`).
> JAX's compilation-cache config API churns across minors (e.g.
> `initialize_cache`/`is_initialized` were removed after 0.4.30); this repo uses
> the version-tolerant `jax_compilation_cache_dir` knob, but staying within 0.4.x
> avoids surprise breakage of the persistent compilation cache and JIT behavior.
> Bump only after re-running the full parity suite. A full lockfile is intentionally
> out of scope; `requirements-jax.txt` records the tested CPU versions.

For checkpoint conversion or PyTorch parity tests (optional):
```bash
pip install -e '.[convert]'
```

## Predict

Single file:
```bash
python -m nisqa_jax.predict \
  --mode predict_file \
  --pretrained_model weights/nisqa_mos_only.npz \
  --deg sample.wav \
  --precision float32
```

Batch directory:
```bash
python -m nisqa_jax.predict \
  --mode predict_dir \
  --pretrained_model weights/nisqa.npz \
  --data_dir wavs \
  --bs 8 \
  --preprocess_workers 4 \
  --precision bf16
```

Robust batch options:
```bash
# Skip corrupt/too-short files instead of aborting the whole batch; an `error`
# column is added (NaN for good rows, message for bad ones).
python -m nisqa_jax.predict --mode predict_dir --pretrained_model weights/nisqa.npz \
  --data_dir wavs --bs 8 --on_error collect

# On GPU out-of-memory, halve --bs and retry down to 1 (logs each reduction).
python -m nisqa_jax.predict --mode predict_dir --pretrained_model weights/nisqa.npz \
  --data_dir wavs --bs 16 --auto_batch
```

Python API:
```python
from nisqa_jax import load_model, predict_file

model = load_model("weights/nisqa_mos_only.npz", device="gpu", precision="bf16")
scores = predict_file(model, "sample.wav")
# {'mos': 1.324}
```

## Persistent Compilation Cache

Pass `cache_dir` to `load_model` to persist XLA compilations across processes so
the first inference of a given input shape pays the compile cost only once, ever:

```python
model = load_model("weights/nisqa_mos_only.npz", device="gpu", cache_dir="weights")
```

JAX's default 1-second minimum compile-time threshold would silently skip this
model's sub-second compiles, so `load_model` lowers it to 0 when `cache_dir` is
set. The cache directory is treated as trusted executable code by JAX: it must
be writable only by a trusted user (never a shared/world-writable location), as
a tampered cache entry can execute arbitrary code on cache hit.

### Prewarm

The persistent cache eliminates *repeat* compiles, but the first process to hit
a given `(batch_size, bucket_length)` shape still compiles. `prewarm` runs a
tiny dummy `predict_segments` (zeros, no audio) for each shape so the cache is
hot before real traffic:

```python
from nisqa_jax import load_model, prewarm
from nisqa_jax.predict import default_length_bucket

model = load_model("weights/nisqa_mos_only.npz", device="gpu", cache_dir="weights")
prewarm(model, batch_sizes=[8], bucket_lengths=[default_length_bucket(model.config)],
        cache_dir="weights")
```

CLI: pass `--prewarm` to pre-compile the model's default bucket grid at `--bs`
before the first real batch:

```bash
python -m nisqa_jax.predict --mode predict_dir --pretrained_model weights/nisqa_mos_only.npz \
  --data_dir wavs --bs 8 --cache_dir weights --prewarm
```

## Convert Original Checkpoints

Conversion is deterministic and produces an `.npz` artifact plus JSON metadata (source hash, conversion version, tensor shape manifest):

```python
from nisqa_jax.checkpoint import convert_checkpoint

convert_checkpoint("path/to/nisqa_mos_only.tar", cache_dir="weights")
```

## Validate

```bash
pytest -q                    # core JAX tests (standalone, no PyTorch needed)
pytest -q -m parity          # PyTorch reference parity tests (requires torch)
```

## Benchmark

Model-only (JAX only):
```bash
python -m nisqa_jax.bench_compare \
  --pretrained_model weights/nisqa_mos_only.npz \
  --device gpu --precision bf16 --batch_size 8 --seq_len 128 --no_torch
```

JAX vs PyTorch head-to-head (requires CUDA PyTorch):
```bash
python -m nisqa_jax.bench_compare \
  --pretrained_model weights/nisqa_mos_only.npz \
  --device gpu --batch_size 8 --seq_len 128 --steps 100
```

End-to-end with preprocessing:
```bash
python -m nisqa_jax.bench \
  --pretrained_model weights/nisqa_mos_only.npz \
  --device gpu --precision bf16 --batch_size 8 \
  --data_dir wavs --preprocess_workers 4
```

## Adversarial Review

A full read-only adversarial review (136 probes: correctness, edge cases, parity, GPU benchmark) is documented in [`adversarial_review/`](adversarial_review/). Key findings and the improvement roadmap are in [`adversarial_review/ISSUES_AND_ROADMAP.md`](adversarial_review/ISSUES_AND_ROADMAP.md).

## Architecture

```
nisqa_jax/
├── model.py         — JAX forward graph (CNN, self-attention, LSTM, attention pooling)
├── checkpoint.py    — .tar→.npz conversion + artifact loading
├── config.py        — checkpoint args → validated ModelConfig
├── features.py      — librosa mel-spectrogram + segmentation (matches PyTorch exactly)
├── predict.py       — CLI + batch prediction API
├── bench.py         — end-to-end benchmark
└── bench_compare.py — JAX vs PyTorch head-to-head benchmark
```

Key porting decisions:
- **BatchNorm folded into conv** at conversion time (no BN at runtime)
- **Adaptive max pool** uses exact PyTorch bin-edge algorithm
- **LSTM** via `jax.lax.scan` with invalid-timestep masking (`jnp.where`)
- **Self-attention** keeps PyTorch's single `in_proj`, splits q/k/v in forward
- **Pure JAX** — no Flax dependency, functional pytrees + `NisqaJaxModel` wrapper

## License

- **Source code:** MIT (see [LICENSE](LICENSE))
- **Model weights:** the bundled NISQA weights in `weights/` are derived from the
  original TU Berlin checkpoints and are licensed **CC BY-NC-SA 4.0 (non-commercial)**
  — see [weights/LICENSE_model_weights](weights/LICENSE_model_weights) for the full
  text. **Commercial deployment requires resolving the model-weight license
  separately**; the MIT license above covers this port's source code only.
- **Academic use:** please cite the original NISQA paper (see [CITATION.cff](CITATION.cff))

## Acknowledgements

Original NISQA model by Gabriel Mittag, Babak Naderi, Assmaa Chehadi, and Sebastian Möller (TU Berlin). See [gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) for the original PyTorch implementation, training code, and datasets.
