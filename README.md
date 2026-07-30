# NISQA-JAX

![CI](https://github.com/hcsolakoglu/nisqa-jax/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB)
![Source license](https://img.shields.io/badge/source-MIT-blue)
![Model weights](https://img.shields.io/badge/model%20weights-CC%20BY--NC--SA%204.0-orange)

NISQA-JAX is a PyTorch-free JAX inference port for the three shipped
[NISQA](https://github.com/gabrielmittag/NISQA) speech-quality checkpoints. It
provides a Python API and PyTorch-compatible file, directory, and CSV command
line interfaces using bundled, integrity-checked model artifacts.

> [!IMPORTANT]
> Source code is MIT-licensed. The bundled model weights retain the upstream
> **CC BY-NC-SA 4.0 non-commercial license**. Commercial use requires separate
> permission for the weights.

This is a pre-1.0, inference-only project. Training, fine-tuning, dataset
evaluation, double-ended `NISQA_DE`, and arbitrary NISQA architectures are
outside the supported scope.

## Models

| Artifact | Architecture | Python result |
|---|---|---|
| `nisqa_mos_only.npz` | adaptive CNN + self-attention | `{"mos": ...}` |
| `nisqa.npz` | adaptive CNN + self-attention | `{"mos", "noi", "dis", "col", "loud"}` |
| `nisqa_tts.npz` | standard CNN + bidirectional LSTM | `{"naturalness": ...}` |

Converted artifacts ship in `nisqa_jax/weights/`; normal inference does not
need PyTorch or the original source checkpoints.

## Quick start

Create an isolated CPU environment from a repository checkout:

```bash
python -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jax.txt
python -m pip install -e . --no-deps
```

Predict one audio file:

```python
from nisqa_jax import load_model, predict_file
from nisqa_jax.weights import WEIGHTS_DIR

model = load_model(
    WEIGHTS_DIR / "nisqa_mos_only.npz",
    device="cpu",
    precision="float32",
)
scores = predict_file(model, "/path/to/audio.wav")
print(scores)
```

Equivalent CLI:

```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
nisqa-jax \
  --mode predict_file \
  --pretrained_model "$W" \
  --deg /path/to/audio.wav \
  --device cpu
```

Directory batch:

```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa.npz")')"
nisqa-jax \
  --mode predict_dir \
  --pretrained_model "$W" \
  --data_dir /path/to/wavs \
  --bs 8 \
  --preprocess_workers 4 \
  --on_error collect
```

Use `--auto_batch` to halve a recognized device-OOM batch down to one sample.
Length-aware scheduling and bucket padding limit wasted compute and JIT-shape
proliferation.

## Backends

| Backend | Status |
|---|---|
| CPU | Supported and tested |
| NVIDIA CUDA 12 | Supported and manually tested |
| TPU | Expected portable, but not tested or claimed as supported evidence |

Install the current qualified CUDA stack in a separate environment:

```bash
python -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e . --no-deps
python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

`float32` is the conformance default. `bf16` is opt-in and keeps numerically
sensitive reductions in float32. Backend qualification, GPU memory guidance,
batch sizing, and persistent-cache safety are documented in the
[backend guide](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/backends.md).

## Compatibility

| Axis | Supported range | Configured CI matrix |
|---|---|---|
| Python | 3.10–3.12 | 3.10, 3.11, 3.12 |
| JAX / jaxlib | `>=0.4.30,<0.7` | 0.4.30, 0.5.3, 0.6.2 |
| NumPy | `>=1.26,<2.3` | 1.26.4 and 2.2.6 |
| Precision | float32, optional bf16 | CPU gates; CUDA release qualification is manual |

JAX and jaxlib must remain a matched pair. Exact current CPU and CUDA pins are
in `requirements-jax.txt` and `requirements-gpu.txt`.

## Validation

Install the self-contained test extra and run the local gates:

```bash
python -m pip install -e ".[test]"
python -m pip install "ruff==0.6.9" "mypy==1.11.2"
ruff check .
mypy nisqa_jax/
python scripts/verify_artifacts.py --strict
JAX_PLATFORMS=cpu python -m pytest -q
```

The golden suite compares current JAX intermediates and outputs with committed,
independently hashed PyTorch-reference vectors at a `5e-5` bound. Frozen audio
fixtures protect WAV decoding, mel preprocessing, segmentation, and final
predictions across the supported NumPy range.

Every artifact load verifies its sidecar hash, metadata checksum, tensor names,
shapes, dtypes, finiteness, and supported architecture. The strict release
verifier additionally checks the bundled checksum catalog and rejects unknown
artifacts.

Live PyTorch parity is optional:

```bash
python -m pip install -e ".[parity]"
# Requires trusted source checkpoints and upstream source at nisqa_pytorch/.
JAX_PLATFORMS=cpu python -m pytest -q -m parity
```

See the
[validation guide](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/validation.md)
for compatibility, packaging, vulnerability-audit, clean-room, and CUDA gates.

## Performance evidence

The current JAX 0.6.2 implementation has correctness and CUDA smoke evidence,
but no committed full post-optimization comparison grid. The project therefore
makes no current universal speedup claim.

Historical JAX 0.4.30 results are retained for provenance and explicitly
separated by code revision and PyTorch baseline. See the
[benchmark guide](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/benchmarks/README.md).

## Documentation

| Topic | Guide |
|---|---|
| Architecture and artifact contract | [Architecture](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/architecture.md) |
| Backends, precision, memory, and caching | [Backends](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/backends.md) |
| Validation and compatibility | [Validation](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/validation.md) |
| Benchmark policy and retained evidence | [Benchmarks](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/benchmarks/README.md) |
| Contribution workflow | [Contributing](https://github.com/hcsolakoglu/nisqa-jax/blob/main/CONTRIBUTING.md) |
| Security boundaries | [Security](https://github.com/hcsolakoglu/nisqa-jax/blob/main/SECURITY.md) |
| Release process | [Releasing](https://github.com/hcsolakoglu/nisqa-jax/blob/main/docs/releasing.md) |
| Version history | [Changelog](https://github.com/hcsolakoglu/nisqa-jax/blob/main/CHANGELOG.md) |

## Contributing, citation, and license

Read the
[contribution guide](https://github.com/hcsolakoglu/nisqa-jax/blob/main/CONTRIBUTING.md)
before changing public behavior, dependencies, or model artifacts. Report
suspected vulnerabilities through the private process in
[SECURITY.md](https://github.com/hcsolakoglu/nisqa-jax/blob/main/SECURITY.md),
not a public issue.

Research users should cite this software and the original NISQA paper using
[CITATION.cff](https://github.com/hcsolakoglu/nisqa-jax/blob/main/CITATION.cff).

- Source code: [MIT](https://github.com/hcsolakoglu/nisqa-jax/blob/main/LICENSE)
- Bundled model weights:
  [CC BY-NC-SA 4.0](https://github.com/hcsolakoglu/nisqa-jax/blob/main/nisqa_jax/weights/LICENSE_model_weights)

Original NISQA model by Gabriel Mittag, Babak Naderi, Assmaa Chehadi, and
Sebastian Möller (TU Berlin).
