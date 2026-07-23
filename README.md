# NISQA-JAX

![CI](https://github.com/hcsolakoglu/nisqa-jax/actions/workflows/ci.yml/badge.svg)

Standalone [JAX](https://github.com/google/jax) inference port for the three shipped [NISQA](https://github.com/gabrielmittag/NISQA) speech quality assessment checkpoints. No PyTorch dependency at inference time — runs on CPU, CUDA GPU, and TPU (see [Backends](#backends-cpu--cuda--tpu) for the per-backend support/tested matrix).

**~3× faster than eager PyTorch** on the self-attention model and **~1.3–1.9× faster than optimized PyTorch** on the BiLSTM (TTS) model (model-forward, RTX 3070). Numerical parity vs the PyTorch CPU reference at 5e-5. See the [Benchmark](#benchmark-jax-gpu-vs-pytorch-gpu-rtx-3070) section for the full optimized-PyTorch comparison.

## Checkpoints

| Artifact | Architecture | Outputs | Source |
|---|---|---|---|
| `nisqa_mos_only.npz` | adapt-CNN + self-attention + att-pool | `mos` | `nisqa_mos_only.tar` |
| `nisqa.npz` | adapt-CNN + self-attention + att-pool | `mos, noi, dis, col, loud` | `nisqa.tar` (NISQA_DIM) |
| `nisqa_tts.npz` | standard-CNN + BiLSTM + last-step-bi | `naturalness` | `nisqa_tts.tar` |

Pre-converted `.npz` weights ship inside the `nisqa_jax/weights/` package data — zero-config inference after install (no repo checkout needed).

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
| **CPU** | Tested | Full 107-test suite passes on `jax 0.4.30` and `jax 0.4.38` CPU (`JAX_PLATFORMS=cpu`) across Python 3.10/3.11/3.12; persistent compilation cache verified on CPU. |
| **CUDA** | Tested (developer machine, not CI) | Full test suite passes on `jax 0.4.30` CUDA, RTX 3070 (`JAX_PLATFORMS=cuda`); CPU-vs-CUDA parity measured (max abs diff ≤ 2.7e-6 across all 3 models); persistent cache verified on CUDA. CI runs CPU-only (no GPU runner); CUDA is re-validated manually before each release. |
| **TPU** | Code-audited, expected-supported | No TPU hardware in CI. Portability established by code audit + JAX docs: all matmuls/convs/einsums run under `jax.default_matmul_precision("float32")` (f32 accumulation on TPU per JAX docs), every reduction (LayerNorm mean/var, attention + pooling softmax/einsum) casts to f32, NHWC/HWIO conv layout is TPU-optimal, int32 indices throughout (no int64 downcast), and bf16 compute is native on TPU. Not "tested" — run the suite on TPU hardware before production deployment. |

### Supported matrix (truthful)

| Axis | Supported range | CI-gated |
|---|---|---|
| Python | 3.10, 3.11, 3.12 (`requires-python>=3.10`) | Yes (all three) |
| JAX / jaxlib | `>=0.4.30,<0.5` | Yes (lower `0.4.30` + upper `0.4.38`) |
| Precision | `float32` (default, strict parity), `bf16` (compute, f32 reductions) | Yes (CPU) |
| Architectures | the 3 shipped checkpoints only (see table above) | Yes |

Unsupported (rejected at the config boundary with a clear error): `NISQA_DE` (double-ended), multi-head self-attention (`td_sa_nhead>1`), and any architecture combo outside the 3 shipped checkpoints. See `nisqa_jax/config.py`.

### Security trust boundary

- **Bundled weight artifacts** (`nisqa_jax/weights/*.npz` + `.json`) are verified on every load and in CI: SHA-256 checksum (`CHECKSUMS.sha256`), embedded `npz_sha256`, and `shape_manifest` (tensor keys/shapes/dtypes). The verifier runs in strict mode in CI — an unknown/unlisted artifact fails the gate. Treat the shipped artifacts as trusted data; do not load `.npz`/`.json` from untrusted sources without re-running conversion from a trusted `.tar`.
- **Persistent compilation cache** (`cache_dir`) is treated as **trusted executable code** by JAX: a tampered cache entry can execute arbitrary code on cache hit. The cache directory must be writable only by a trusted user (never a shared/world-writable location).
- **Source `.tar` checkpoints** (optional `convert`/`parity` extras) are loaded via PyTorch `torch.load`; only convert checkpoints from a trusted source. Inference from the shipped `.npz` artifacts does not require PyTorch or touch `.tar` files.

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

- The benchmark scripts (`bench_compare.py`, `bench.py`,
  `adversarial_review/probes/bench_batching.py`, `adversarial_review/`) compare
  against CUDA PyTorch and use `torch.cuda` — they are **CUDA-only** and require
  a CUDA PyTorch install. The core inference and test suite have no such
  dependency.
- `--auto_batch` OOM recovery matches `ResourceExhaustedError` and the
  `RESOURCE_EXHAUSTED` / `out of memory` tokens in the error message, which
  covers GPU and TPU OOM surfaces; CPU does not raise OOM (it over-commits), so
  non-OOM CPU errors never trigger a futile retry.
- No `XLA_FLAGS` are set or required by this repo.

## Predict

A `nisqa-jax` console script is installed alongside the package (equivalent to
`python -m nisqa_jax.predict`). The `--pretrained_model` path may be an
absolute path to any converted `.npz`, or you can resolve the bundled
checkpoints portably via `nisqa_jax.weights.WEIGHTS_DIR` (works inside an
installed wheel/sdist with no repo checkout).

**Installed package (recommended):** resolve the bundled weights directory from
the installed package so the path is correct regardless of install location:

```bash
# Shell: resolve WEIGHTS_DIR from the installed package, then pass it to the CLI.
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"

nisqa-jax --mode predict_file --pretrained_model "$W" --deg sample.wav --precision float32
```

```bash
# Single file (module form):
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.predict --mode predict_file --pretrained_model "$W" \
  --deg sample.wav --precision float32
```

```bash
# Batch directory:
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa.npz")')"
python -m nisqa_jax.predict --mode predict_dir --pretrained_model "$W" \
  --data_dir wavs --bs 8 --preprocess_workers 4 --precision bf16
```

```bash
# Robust batch options:
# Skip corrupt/too-short files instead of aborting the whole batch; an `error`
# column is added (NaN for good rows, message for bad ones).
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa.npz")')"
python -m nisqa_jax.predict --mode predict_dir --pretrained_model "$W" \
  --data_dir wavs --bs 8 --on_error collect

# On GPU out-of-memory, halve --bs and retry down to 1 (logs each reduction).
python -m nisqa_jax.predict --mode predict_dir --pretrained_model "$W" \
  --data_dir wavs --bs 16 --auto_batch
```

**Source checkout:** when running from a clone of the repo (editable install or
not), the in-package weights are also reachable via the repo-relative path
`nisqa_jax/weights/<name>.npz`, e.g.
`--pretrained_model nisqa_jax/weights/nisqa_mos_only.npz`. The
`WEIGHTS_DIR` form above is preferred because it is install-location-agnostic
and works inside an installed wheel.

Python API:
```python
from nisqa_jax import load_model, predict_file
from nisqa_jax.weights import WEIGHTS_DIR

# WEIGHTS_DIR resolves to the installed package's bundled weights directory.
model = load_model(WEIGHTS_DIR / "nisqa_mos_only.npz", device="gpu", precision="bf16")
scores = predict_file(model, "sample.wav")
# {'mos': 1.324}
```

## Persistent Compilation Cache

Pass `cache_dir` to `load_model` to persist XLA compilations across processes so
the first inference of a given input shape pays the compile cost only once, ever:

```python
from nisqa_jax import load_model
from nisqa_jax.weights import WEIGHTS_DIR

model = load_model(WEIGHTS_DIR / "nisqa_mos_only.npz", device="gpu", cache_dir="xla_cache")
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
from nisqa_jax.weights import WEIGHTS_DIR

model = load_model(WEIGHTS_DIR / "nisqa_mos_only.npz", device="gpu", cache_dir="xla_cache")
prewarm(model, batch_sizes=[8], bucket_lengths=[default_length_bucket(model.config)],
        cache_dir="xla_cache")
```

CLI: pass `--prewarm` to pre-compile the model's default bucket grid at `--bs`
before the first real batch:

```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.predict --mode predict_dir --pretrained_model "$W" \
  --data_dir wavs --bs 8 --cache_dir xla_cache --prewarm
```

## Convert Original Checkpoints

Conversion is deterministic and produces an `.npz` artifact plus JSON metadata (source hash, conversion version, tensor shape manifest):

```python
from nisqa_jax.checkpoint import convert_checkpoint

convert_checkpoint("path/to/nisqa_mos_only.tar", cache_dir="xla_cache")
```

## Validate

```bash
pytest -q                    # self-contained CI suite (no PyTorch needed)
pytest -q tests/test_golden_parity.py  # golden-vector parity gate only
pytest -q -m parity          # live PyTorch parity tests (requires torch +
                             #   source .tar checkpoints + PyTorch reference
                             #   source repo; skips cleanly without them)
```

The CI pipeline runs the self-contained suite + golden parity gate only.
Live PyTorch parity tests are optional (they require the source `.tar`
checkpoints and the PyTorch reference source repo, which are not in this
repo); they skip cleanly when those are absent. The golden parity gate
(`tests/test_golden_parity.py`) is the CI correctness proof — it replays
deterministic golden vectors (generated from the trusted converted artifacts
and validated against PyTorch at generation time) with a strict 5e-5
tolerance, no torch install required.

## Development

The CI gates (`.github/workflows/ci.yml`) can all be run locally. Tool versions
are pinned in CI (ruff 0.6.9, mypy 1.11.2, build 1.2.2.post1); match them
locally for reproducible results:

```bash
pip install "ruff==0.6.9" "mypy==1.11.2" "build==1.2.2.post1" pytest

ruff check .                 # lint (config in [tool.ruff] in pyproject.toml)
mypy nisqa_jax/              # lenient typecheck (excludes tests/bench/probes)
pytest -q                    # test suite (torch-dependent tests auto-skip)
python scripts/verify_artifacts.py   # weight checksum + manifest + metadata verification
python -m build              # build sdist + wheel
pytest -q tests/test_build_contents.py  # wheel + sdist carry required files
```

The bundled weight artifacts are verified against
`nisqa_jax/weights/CHECKSUMS.sha256` (SHA-256 of each `.npz`), each `.json`
sidecar's `shape_manifest` (tensor keys/shapes/dtypes), and the metadata gate
(required fields + `npz_sha256` cross-check + no absolute-path leak) on every
CI run, in strict mode (unknown/unlisted artifacts fail the gate).

## Benchmark

Model-only (JAX only):
```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.bench_compare \
  --pretrained_model "$W" \
  --device gpu --precision bf16 --batch_size 8 --seq_len 128 --no_torch
```

JAX vs PyTorch head-to-head (requires CUDA PyTorch):
```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.bench_compare \
  --pretrained_model "$W" \
  --device gpu --batch_size 8 --seq_len 128 --steps 100
```

End-to-end with preprocessing:
```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.bench \
  --pretrained_model "$W" \
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
├── bench_compare.py — JAX vs PyTorch head-to-head benchmark
└── weights/         — bundled .npz artifacts + .json metadata + CHECKSUMS.sha256 (package data)
```

Key porting decisions:
- **BatchNorm folded into conv** at conversion time (no BN at runtime)
- **Adaptive max pool** uses exact PyTorch bin-edge algorithm
- **LSTM** via a fused bidirectional `jax.lax.scan` over a 2×batch stack (forward + time-reversed backward) with precomputed input projection (one batched GEMM outside the scan) and invalid-timestep masking (`jnp.where`); per-direction recurrent weights applied via a batched (group=2) GEMM
- **Self-attention** keeps PyTorch's single `in_proj`, splits q/k/v in forward
- **Pure JAX** — no Flax dependency, functional pytrees + `NisqaJaxModel` wrapper

## License

- **Source code:** MIT (see [LICENSE](LICENSE))
- **Model weights:** the bundled NISQA weights in `nisqa_jax/weights/` are derived from the
  original TU Berlin checkpoints and are licensed **CC BY-NC-SA 4.0 (non-commercial)**
  — see [nisqa_jax/weights/LICENSE_model_weights](nisqa_jax/weights/LICENSE_model_weights) for the full
  text. **Commercial deployment requires resolving the model-weight license
  separately**; the MIT license above covers this port's source code only.
- **Academic use:** please cite the original NISQA paper (see [CITATION.cff](CITATION.cff))

## Acknowledgements

Original NISQA model by Gabriel Mittag, Babak Naderi, Assmaa Chehadi, and Sebastian Möller (TU Berlin). See [gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) for the original PyTorch implementation, training code, and datasets.
