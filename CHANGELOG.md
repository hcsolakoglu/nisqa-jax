# Changelog

All notable changes to NISQA-JAX are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning for published releases.

## Unreleased

### Added

- Qualified CUDA 12 environment and runtime requirements.
- Frozen NumPy 1.26 audio-frontend and all-checkpoint output regression
  fixtures.
- Multi-stack CI matrix for JAX/jaxlib 0.4.30, 0.5.3, and 0.6.2 across Python
  3.10–3.12.
- Architecture, validation, release, contribution, and security documentation.
- A pinned dependency-vulnerability audit gate for the resolved current stack.

### Changed

- Current direct pins now use JAX/jaxlib 0.6.2, NumPy 2.2.6, SciPy 1.15.3,
  and pandas 2.3.3 while retaining the documented compatibility floor.
- GitHub Actions use immutable checkout 7.0.1 and setup-python 7.0.0 SHAs.
- Dependabot groups coupled runtime and GitHub Actions updates.
- Packaging and clean-room tests verify all required artifacts, fixtures,
  metadata, and documentation.
- Distribution metadata now records the MIT source and CC BY-NC-SA 4.0
  model-weight licenses separately.
- User documentation is split into focused architecture, backend, validation,
  benchmark, and release guides; the README is limited to core user workflows.

### Removed

- One-off adversarial probe scripts, reviewer metadata, resolved issue
  inventories, and obsolete architecture experiments from the active tree.
- GitHub repository-management files from source distributions.

### Fixed

- Strict source-state, converted-artifact, metadata, cache, prediction, batch,
  CSV, error-isolation, and benchmark-reporting behavior found during
  repository-wide adversarial review.
- NumPy 2 frontend drift is bounded against independent frozen reference data.

## Initial development baseline - 2026-07-22

This was a repository snapshot, not a tagged or published release.

### Added

- Pure-JAX inference for the three shipped NISQA checkpoints.
- Bundled standalone model artifacts with deterministic conversion metadata.
- File, directory, CSV, and batch prediction APIs.
- CPU/CUDA execution, persistent compilation cache support, benchmarking, and
  PyTorch-reference parity tests.
