# NISQA-JAX documentation

The top-level [README](../README.md) is the user-facing overview and quick
start. These guides define the maintainer-facing contracts:

- [Architecture](architecture.md): components, data flow, model scope, device
  placement, batching, and artifact boundaries.
- [Backends](backends.md): CPU/CUDA/TPU status, precision, device memory,
  batching, and persistent-cache operation.
- [Validation](validation.md): local gates, compatibility matrices, CUDA
  qualification, live PyTorch parity, and benchmark evidence.
- [Benchmarks](benchmarks/README.md): measurement policy, bounded Hugging Face
  real-data comparison, current results, and retained historical results.
- [Releasing](releasing.md): versioning, build, artifact, documentation, and
  publication checklist.

Repository-wide policies:

- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)
- [Changelog](../CHANGELOG.md)
- [Original inference-port design record](history/initial-port-design.md)
