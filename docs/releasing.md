# Releasing

The repository currently has no automated package-publication workflow and no
tagged release. The `0.1.0` package metadata is a development version, not
evidence of publication. A release is complete only when its version,
changelog, citation, artifacts, tag, and publication destination agree.

## 1. Define the release

- Choose the semantic version and intended consumers.
- Review `CHANGELOG.md`; move relevant entries from `Unreleased` into the new
  version and date.
- Update `pyproject.toml` and `CITATION.cff` together.
- Confirm README compatibility, backend, benchmark, and licensing statements.
- Decide whether any model artifacts changed. If so, record the trusted source
  hashes and regenerate every dependent fixture.

## 2. Qualify the source

Run the full process in [validation.md](validation.md), including:

- all CPU compatibility/Python matrix jobs;
- strict artifacts and self-contained golden parity;
- frozen audio-frontend regression;
- full test suite;
- wheel and sdist content gates;
- clean-room wheel and sdist installs;
- `pip check` plus the pinned `pip-audit` gate on the resolved CPU and CUDA
  environments;
- CUDA qualification when CUDA support is claimed.
- The current two-thousand-sample real-data benchmark and CPU PyTorch
  correctness result under `docs/benchmarks/results/` when performance claims are
  changed.

Do not waive a failed gate because a smoke test passed. If hosted CI cannot
execute, restore it and obtain a green run before publishing.

## 3. Build from a clean commit

Build into a fresh output directory so a stale ignored package can never be
selected, checked, hashed, or uploaded:

```bash
git status --short
RELEASE_DIST="$(mktemp -d)"
python -m build --outdir "$RELEASE_DIST"
NISQA_JAX_DIST_DIR="$RELEASE_DIST" python -m pytest -q tests/test_build_contents.py
python -m pip install "twine==6.1.0"
python -m twine check "$RELEASE_DIST"/*
sha256sum "$RELEASE_DIST"/*
```

`git status --short` must be empty before building. The package-content gate
requires exactly one wheel and one source distribution in `RELEASE_DIST`.
Preserve their hashes with the release notes.

Install the wheel and source distribution independently into fresh
environments. From outside the source checkout, verify:

- `import nisqa_jax`;
- bundled `WEIGHTS_DIR`;
- strict installed-artifact verification;
- a real-WAV CLI prediction;
- finite forward passes for all three checkpoints.

## 4. Review legal and provenance material

- Source files remain MIT-licensed.
- Bundled model weights remain CC BY-NC-SA 4.0 and non-commercial.
- `LICENSE`, `nisqa_jax/weights/LICENSE_model_weights`, and `CITATION.cff` ship
  in the appropriate artifacts.
- No source checkpoint, user audio, absolute local path, credential, cache, or
  unrelated generated file is present.

## 5. Tag and publish

After review and green CI:

```bash
git tag -a vX.Y.Z -m "NISQA-JAX vX.Y.Z"
git push origin vX.Y.Z
```

Use a signed tag when maintainer signing is configured. Create GitHub release
notes from the changelog and attach distribution hashes. Publish only to an
explicitly chosen package index. Prefer PyPI Trusted Publishing over long-lived
upload tokens when publication automation is added.

## 6. Verify the published release

- Install by version from the publication destination into a clean environment.
- Run import, bundled-weight, artifact-verifier, and real-WAV smoke checks.
- Confirm the Git tag, release notes, metadata version, citation version, and
  changelog all match.
- Open the next `Unreleased` section immediately after release.
