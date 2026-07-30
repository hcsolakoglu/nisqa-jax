# Contributing to NISQA-JAX

Thank you for improving NISQA-JAX. Contributions should preserve numerical
parity, artifact integrity, the three-checkpoint scope, and honest performance
claims.

## Before you start

- Read the [architecture guide](docs/architecture.md) and
  [validation guide](docs/validation.md).
- For a large feature, new public API, architecture expansion, or dependency
  policy change, open an issue first so scope and compatibility expectations
  are explicit.
- Do not commit source `.tar` checkpoints, user audio, generated result CSVs,
  local compilation caches, or virtual environments.

## Development setup

Use an isolated environment and the exact qualified CPU stack:

```bash
python -m venv .venv-dev
. .venv-dev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jax.txt
python -m pip install -e ".[test]"
python -m pip install \
  "ruff==0.6.9" "mypy==1.11.2" "build==1.2.2.post1" "pip-audit==2.9.0"
```

Run the fast local gates:

```bash
ruff check .
mypy nisqa_jax/
python scripts/verify_artifacts.py --strict
JAX_PLATFORMS=cpu python -m pytest -q tests/test_golden_parity.py
JAX_PLATFORMS=cpu python -m pytest -q tests/test_audio_frontend_regression.py
```

Run the full CPU and packaging gates before requesting merge:

```bash
JAX_PLATFORMS=cpu python -m pytest -q
python -m build
python -m pytest -q tests/test_build_contents.py
python -m pip check
python -m pip_audit --local --progress-spinner off
```

See [docs/validation.md](docs/validation.md) for CUDA, clean-room, and optional
live PyTorch-reference validation.

## Engineering expectations

### Correctness and compatibility

- Keep JAX and jaxlib at matching versions.
- Treat JAX, NumPy, SciPy, pandas, Librosa, and SoundFile changes as one
  compatibility surface. Update CPU/GPU requirements, project bounds, CI,
  frozen frontend evidence, and documentation together when needed.
- Preserve strict float32 parity. Reduced precision must remain opt-in and
  must not weaken the conformance tolerances.
- Reject unsupported architecture or artifact states explicitly rather than
  silently approximating them.

### Artifacts and security

- Bundled artifacts must retain checksum, metadata checksum, tensor-shape
  manifest, source provenance, and license information.
- Never hand-edit a bundled `.npz` or its JSON sidecar. Re-run the deterministic
  conversion flow from a trusted source checkpoint.
- Any artifact change must update the golden/reference fixtures and pass the
  strict verifier.
- Treat source PyTorch checkpoints and JAX compilation caches as trusted input.
  See [SECURITY.md](SECURITY.md).

### Performance

- Benchmark with warmup excluded and asynchronous JAX work synchronized.
- Separate compile/cache lookup, host-to-device transfer, warmed model-forward,
  and end-to-end preprocessing time.
- State the exact hardware, driver, framework versions, precision, shapes,
  iteration count, and aggregation method.
- Do not replace published performance claims with a smoke test or a
  non-equivalent PyTorch configuration.

### Documentation

- Update README and supporting guides when changing public APIs, CLI behavior,
  compatibility ranges, environment setup, artifacts, or benchmark claims.
- Keep historical measurements labeled with their original environment.
- Add new required source-distribution files to `MANIFEST.in` and
  `tests/test_build_contents.py`.

## Pull request checklist

- The change is focused and has no unrelated generated or formatting churn.
- Observable behavior is covered by a test that can fail for a broken
  implementation.
- Relevant lint, type, test, artifact, and package gates were executed.
- CPU/CUDA and compatibility implications are documented.
- New dependencies are necessary, bounded, and included in the appropriate
  Dependabot group.
- `CHANGELOG.md` is updated for user-visible behavior.
- No model weights, user audio, credentials, local paths, or cache files were
  added accidentally.
