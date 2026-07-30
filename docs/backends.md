# Backends and deployment

## Support status

| Backend | Status | Qualification |
|---|---|---|
| CPU | Supported and tested | Full self-contained suite on the current JAX/jaxlib 0.6.2 stack |
| NVIDIA CUDA | Supported and manually tested | JAX/jaxlib 0.6.2 on an RTX 3070; release qualification remains manual |
| TPU | Expected portable, not supported as tested | Pure-JAX graph only; no project TPU execution evidence |

The default backend is whichever backend JAX selects. Pass `device="cpu"`,
`device="gpu"`/`"cuda"`, or `device="tpu"` to request a platform explicitly.
An unavailable platform raises instead of silently changing the request.

## Installation

Keep CPU and CUDA environments separate.

CPU:

```bash
python -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jax.txt
python -m pip install -e . --no-deps
python -c 'import jax; print(jax.devices()); assert all(d.platform == "cpu" for d in jax.devices())'
```

NVIDIA CUDA 12:

```bash
python -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e . --no-deps
python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

The CUDA assertion must pass before recording CUDA correctness or performance
evidence. Consult the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
driver and platform requirements.

## Precision

`precision="float32"` is the conformance default. The model uses
`jax.default_matmul_precision("float32")` and explicit float32 reductions for
LayerNorm, attention, and pooling.

`precision="bf16"` is an opt-in compute mode. It is tested on CPU and NVIDIA
Ampere CUDA with separate drift bounds. TPU precision and performance remain
unverified by this project.

## Batching and memory

Peak device memory grows with batch size and padded sequence length. Historical
worst-case measurements on an otherwise idle 8 GB RTX 3070 found these upper
bounds at each checkpoint's full sequence length:

| Checkpoint | Full sequence | Largest measured batch | First measured failure |
|---|---:|---:|---:|
| `nisqa_mos_only` | 1300 | 32 | 48 |
| `nisqa` | 1300 | 16 | 24 |
| `nisqa_tts` | 6000 | 8 | 12 |

Use conservative starting points:

| GPU memory | Self-attention `--bs` | TTS `--bs` |
|---:|---:|---:|
| 8 GB or less | 4–8 | 2–4 |
| 12 GB | 8–16 | 4–8 |
| 24 GB | 16–32 | 8–16 |

These are deployment hints, not guarantees. Other GPU users, allocator state,
audio-length distribution, and framework versions change the boundary. Pass
`--auto_batch` to halve a recognized device-OOM batch down to one sample.

## Compilation cache

Pass `cache_dir` to `load_model` to enable JAX's persistent compilation cache:

```python
from nisqa_jax import load_model
from nisqa_jax.weights import WEIGHTS_DIR

model = load_model(
    WEIGHTS_DIR / "nisqa_mos_only.npz",
    device="gpu",
    cache_dir="xla_cache",
)
```

The cache directory is process-global and contains trusted executable state.
Use one consistent directory per process and never share a world-writable cache
between users, tenants, or services.

Use `prewarm` or the CLI's `--prewarm` option to compile the documented bucket
grid before serving real traffic. See [architecture.md](architecture.md) for
the shape policy and [validation.md](validation.md) for release qualification.
