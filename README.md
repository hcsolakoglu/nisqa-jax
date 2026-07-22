# NISQA-JAX

Standalone [JAX](https://github.com/google/jax) inference port for the three shipped [NISQA](https://github.com/gabrielmittag/NISQA) speech quality assessment checkpoints. No PyTorch dependency at inference time — runs on CPU, CUDA GPU, and TPU (see [Backends](#backends-cpu--cuda--tpu) for the per-backend support/tested matrix).

**~3× faster than eager PyTorch** on the self-attention model and **~1.3–1.9× faster than optimized PyTorch** on the BiLSTM (TTS) model (model-forward, RTX 3070). Numerical parity vs the PyTorch CPU reference at 5e-5. See the [Benchmark](#benchmark-jax-gpu-vs-pytorch-gpu-rtx-3070) section for the full optimized-PyTorch comparison.

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
| tts (BiLSTM) | **1.9×** (1.4–2.6) | n/a² | **1.33×** vs max-autotune (0.97–1.70) | 5e-5 (CPU ref)³ |

Speedup = PT_latency / JAX_latency (>1 = JAX faster), geomean over the 9-shape grid.

**Headline correction.** The previous "3–7× faster than PyTorch" was measured
against *eager* PyTorch and held only for the self-attention model. Against
**optimized** PyTorch the self-attention speedup is ~3.1× (eager) / ~2.3× (CUDA
graphs). The BiLSTM (TTS) model was previously ~0.97× (on par / up to 2× slower
on long single sequences); after the Goal-B LSTM rewrite — precomputed input
projection (one batched GEMM outside the scan, as cuDNN does) plus both
directions fused into a single `lax.scan` over a 2×batch stack — it is now
**~1.9× vs eager and ~1.33× vs `torch.compile(max-autotune)`** geomean. The
formerly-worst cell (bs=1/steps=512) went from 0.49× (2× slower) to ~1.3×; the
heaviest cell (bs=16/steps=512) sits at ~parity (0.97–1.3× across runs, memory-
pressure variance on the 8 GB card). `torch.compile` does not materially change
TTS latency (cuDNN LSTM is already optimized).

¹ `torch.compile` (inductor / triton 3.6.0) fails on the self-attention model in
this environment with `RuntimeError: CUDA driver error: invalid argument`
launching a fused triton softmax kernel (RTX 3070, driver 595.84, sm_86); verified
across `dynamic=True`, `reduce-overhead`, `max-autotune`, and several inductor
config workarounds (`split_reductions=False`, `autotune_fallback_to_aten=True`,
mask-skip bypass). A triton-free manual CUDA-graph mode is reported instead. The
TTS/LSTM model compiles fine (no fused softmax kernel).
² cuDNN LSTM cannot be captured into a CUDA graph; TTS uses the `torch.compile` modes instead.
³ TTS parity is asserted against the PyTorch **CPU** reference at 5e-5 (see the
"Parity methodology" note below); cuDNN-GPU is the f64-truth outlier and is not
the parity reference.

**Parity methodology (TTS/BiLSTM).** The parity suite compares the JAX port
against the PyTorch **CPU** reference at a strict 5e-5 tolerance. A float64
ground-truth LSTM (same weights) confirms this is the mathematically correct
check: the JAX `jax.lax.scan` loop and PyTorch's non-fused CPU LSTM reference
both agree with f64 truth to ~1e-6, while PyTorch's **cuDNN** GPU LSTM is the
outlier — its fused kernel accumulates the four gates in a different order,
drifting to ~3.5e-5 at bs=32/sl=64 and ~7e-3 at bs=8/sl=6000 (10–100× the CPU
paths). The ~1.2e-3 drift once accommodated by a loose 2e-3 tolerance was
therefore cuDNN's accumulation, not a port bug; the widening has been reverted.

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

## Backends: CPU / CUDA / TPU

The forward graph is pure JAX with no backend-conditional code, so the same
model runs on CPU, CUDA, and TPU. The table below is explicit about what is
**empirically tested** in this repo vs **code-audited only** (no TPU hardware in
CI).

| Backend | Status | Evidence |
|---|---|---|
| **CPU** | Tested | Full 62-test suite passes on `jax 0.4.30` CPU (`JAX_PLATFORMS=cpu`); persistent compilation cache verified on CPU. |
| **CUDA** | Tested | Full 62-test suite passes on `jax 0.4.30` CUDA, RTX 3070 (`JAX_PLATFORMS=cuda`); CPU-vs-CUDA parity measured (max abs diff ≤ 2.7e-6 across all 3 models); persistent cache verified on CUDA. |
| **TPU** | Code-audited, expected-supported | No TPU hardware in CI. Portability established by code audit + JAX docs: all matmuls/convs/einsums run under `jax.default_matmul_precision("float32")` (f32 accumulation on TPU per JAX docs), every reduction (LayerNorm mean/var, attention + pooling softmax/einsum) casts to f32, NHWC/HWIO conv layout is TPU-optimal, int32 indices throughout (no int64 downcast), and bf16 compute is native on TPU. Not "tested" — run the suite on TPU hardware before production deployment. |

### Device selection

`load_model(..., device=...)` accepts the JAX platform names passed to
`jax.devices()`:

| `device=` | Resolves to | Notes |
|---|---|---|
| `None` (default) | `jax.devices()` — the default backend | On a TPU host this is TPU; on a CUDA host, CUDA; on CPU-only, CPU. |
| `"cpu"` | CPU backend | Always available (unless `JAX_PLATFORMS` excludes it). |
| `"gpu"` or `"cuda"` | CUDA backend | Both accepted as aliases. |
| `"tpu"` | TPU backend | Raises a clear `RuntimeError` naming `libtpu.so` and listing available backends if no TPU is present. |

`predict_segments` stages inputs on `model.device` via explicit `jax.device_put`
and the jitted forward runs fully on-device. Under `jax.transfer_guard("disallow")`
the on-device compute raises no implicit-transfer errors (verified on CUDA);
the only guarded transfer in the full predict path is the single intentional
device→host retrieval of the output array.

### Precision per backend

| Setting | CPU | CUDA | TPU |
|---|---|---|---|
| `precision="float32"` (default) | f32 throughout | f32 throughout | f32 throughout — `default_matmul_precision("float32")` forces f32 matmul/conv accumulation on TPU, so strict mode is portable and bit-stable. |
| `precision="bf16"` | Works (emulated; slower) | Native on Ampere+ (RTX 3070 verified) | Native — recommended for throughput. bf16 inputs with f32 accumulation (TPU MXU native); ~2× faster than float32. |

Reductions that would lose precision under bf16 accumulation are explicitly cast
to f32: LayerNorm (`_layer_norm`), attention scores/softmax (`_self_attention_layer`),
and attention pooling (`_pool_att_ff`). The `default_matmul_precision("float32")`
context manager wraps the entire forward, so all `jnp.matmul`/`einsum`/`conv`
calls without an explicit `precision` arg accumulate in f32 on every backend.

### Persistent compilation cache

`cache_dir` enables JAX's persistent compilation cache (see
[Persistent Compilation Cache](#persistent-compilation-cache)). Verified working
on **CPU** and **CUDA** (jax 0.4.30) via the prewarm test suite. The cache config
uses the version-tolerant `jax_compilation_cache_dir` knob (not the
`initialize_cache`/`is_initialized` API removed after 0.4.30). On **TPU** the
persistent cache is supported by JAX (code-audited; not CI-tested here).

### Caveats

- The benchmark scripts (`bench_compare.py`, `bench.py`, `bench_batching.py`,
  `adversarial_review/`) compare against CUDA PyTorch and use
  `torch.cuda` — they are **CUDA-only** and require a CUDA PyTorch install. The
  core inference and test suite have no such dependency.
- `--auto_batch` OOM recovery matches `ResourceExhaustedError` and the
  `RESOURCE_EXHAUSTED` / `out of memory` tokens in the error message, which
  covers GPU and TPU OOM surfaces; CPU does not raise OOM (it over-commits), so
  non-OOM CPU errors never trigger a futile retry.
- No `XLA_FLAGS` are set or required by this repo.

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
- **LSTM** via a fused bidirectional `jax.lax.scan` over a 2×batch stack (forward + time-reversed backward) with precomputed input projection (one batched GEMM outside the scan) and invalid-timestep masking (`jnp.where`); per-direction recurrent weights applied via a batched (group=2) GEMM
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
