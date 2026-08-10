# Validation

NISQA-JAX uses layered validation so dependency resolution, audio preprocessing,
model numerics, artifacts, packaging, and GPU execution fail independently.

## Reproducible CPU setup

```bash
python -m venv .venv-verify
. .venv-verify/bin/activate
python -m pip install --upgrade "pip==26.1.2"
python -m pip install -r requirements-jax.txt
python -m pip install -e ".[test]"
python -m pip install \
  "ruff==0.6.9" "mypy==1.11.2" "build==1.2.2.post1" "pip-audit==2.9.0"
```

## Required pull-request gates

```bash
ruff check .
mypy nisqa_jax/
python scripts/verify_artifacts.py --strict
JAX_PLATFORMS=cpu python -m pytest -q tests/test_golden_parity.py
JAX_PLATFORMS=cpu python -m pytest -q tests/test_audio_frontend_regression.py
JAX_PLATFORMS=cpu python -m pytest -q
python -m build
python -m pytest -q tests/test_build_contents.py
python -m pip check
python -m pip_audit -r requirements-jax.txt --progress-spinner off
```

What each gate proves:

| Gate | Contract |
|---|---|
| Ruff/mypy | Configured static and package type checks |
| Strict artifact verifier | Checksums, metadata, manifests, provenance, and known artifact set |
| Golden parity | Current JAX intermediates and outputs remain within `5e-5` of committed, hash-bound PyTorch references |
| Frozen audio frontend | WAV decode, mel arrays, segments, window counts, and final outputs remain within explicit cross-NumPy tolerances |
| Full suite | APIs, validation, batching, cache, error recovery, benchmarks, CLI, and optional-path skip behavior |
| Build-content gate | Wheel and source distribution include exactly the required runtime and verification assets |
| `pip check` | Final installed dependency graph is internally consistent |
| `pip-audit` | Resolved environment has no known published Python-package vulnerability |

The golden gate hard-fails when its fixture set or trust chain is missing.
Optional live-reference tests skip in the normal test environment because
source checkpoints and PyTorch are deliberately absent.

The audit is a release gate for the exact resolved CPU and CUDA environments,
not a substitute for reviewing dependency provenance. A temporary exception
must name the advisory, affected package/path, risk analysis, compensating
control, owner, and expiry; keep it in version control instead of adding an
unexplained ignore flag.

## Compatibility matrix

The CI workflow pairs matching JAX/jaxlib versions and defines:

| Stack | JAX/jaxlib | NumPy | SciPy | pandas |
|---|---:|---:|---:|---:|
| Floor | 0.4.30 | 1.26.4 | 1.11.4 | 2.1.4 |
| Intermediate | 0.5.3 | 2.2.6 | 1.15.3 | 2.3.3 |
| Current | 0.6.2 | 2.2.6 | 1.15.3 | 2.3.3 |

All stacks run across Python 3.10, 3.11, and 3.12. A dependency-range change is
incomplete until the matrix, exact CPU/GPU requirements, clean-room build,
frontend baseline, and README agree.

The current release-candidate audio baseline is ``librosa==0.11.0`` and
``soundfile==0.14.0`` across all CPU matrix cells and the direct CUDA pins. This
baseline has completed the full frontend, CPU, CUDA, artifact, and clean-room
qualification described below. The previous ``0.10.2.post1`` / ``0.12.1`` stack
remains the historical baseline for comparison.

## CUDA qualification

Use a separate environment:

```bash
python -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade "pip==26.1.2"
python -m pip install -r requirements-gpu.txt
python -m pip install -e ".[test]"
python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
JAX_PLATFORMS=cuda python -m pytest -q tests/test_golden_parity.py
JAX_PLATFORMS=cuda python -m pytest -q
python -m pip check
python -m pip install "pip-audit==2.9.0"
python -m pip_audit -r requirements-gpu.txt --progress-spinner off
```

Before a release claims CUDA support, also verify:

- a real WAV through the installed CLI;
- CPU-versus-CUDA outputs for all three checkpoints;
- persistent-cache population and reuse in a fresh process;
- the final resolved environment and GPU/driver details;
- a representative warmed benchmark smoke.

A benchmark smoke validates execution, not published performance. Re-run the
full comparison grid before changing performance claims.

## Live PyTorch-reference parity

The self-contained golden gate is the normal correctness proof. To run the live
reference path:

1. Install `.[parity]`.
2. Place the upstream NISQA source checkout at `nisqa_pytorch/`.
3. Put the trusted source checkpoints in `nisqa_pytorch/weights/`, or set
   `NISQA_SOURCE_WEIGHTS_DIR` to their directory.
4. Run:

```bash
JAX_PLATFORMS=cpu python -m pytest -q -m parity
```

The original checkpoint bytes must match the source hashes recorded in the
converted artifacts. Parity is against PyTorch CPU; cuDNN LSTM accumulation is
not the numerical reference.

## Benchmark evidence

Every performance report must include:

- repository commit;
- JAX, jaxlib, PyTorch, CUDA runtime, driver, and GPU;
- model, batch sizes, sequence lengths, precision, and input distribution;
- warmup and timed iteration counts;
- whether compilation and transfer are included;
- synchronization method and aggregation statistic;
- raw machine-readable output.

Use `nisqa_jax.bench_compare` for hash-bound model-forward comparison,
`nisqa_jax.bench` for JAX-only or end-to-end preprocessing measurements, and
`scripts/benchmark_hf_real.py` for the current bounded real-data comparison.
The committed result is
[`docs/benchmarks/results/hf-minds14-2k.json`](benchmarks/results/hf-minds14-2k.json).
It uses two thousand valid variable-duration Minds14 examples, records the
resolved Hub revision and shard sizes, compares all shipped models against the
upstream PyTorch implementation, and stores CPU correctness plus CUDA
diagnostic metrics separately. On the committed run, all three CPU comparisons
passed with maximum absolute differences below `2.9e-6` at a `5e-5` threshold.
CUDA outputs were finite and shape-equal but exceeded that strict diagnostic
threshold because of expected backend-specific LSTM accumulation drift. The
corpus is not MOS-labelled, so this gate proves runtime and numerical parity
only, not perceptual-quality accuracy. The before/after frontend optimization
analysis is in
[`benchmarks/results/hf-minds14-2k-optimization.md`](benchmarks/results/hf-minds14-2k-optimization.md).
