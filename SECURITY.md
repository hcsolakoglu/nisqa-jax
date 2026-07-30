# Security Policy

## Supported code

Security fixes target the latest `main` branch and the most recent published
release, when one exists. Older commits, local snapshots, and modified model
artifacts are not maintained as separate security-support lines.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability involving artifact
loading, checkpoint conversion, command execution, dependency compromise, or
sensitive data exposure.

Use GitHub private vulnerability reporting for this repository when available.
If that channel is unavailable, contact the maintainer using the email listed
in [CITATION.cff](CITATION.cff). Include:

- the affected version or commit;
- the entry point and required attacker capability;
- a minimal reproduction;
- expected impact;
- any known mitigation.

Avoid attaching proprietary audio, unredacted paths, credentials, or untrusted
checkpoint files unless a secure transfer method has been agreed upon.

## Security boundaries

### Bundled JAX artifacts

The bundled `.npz` files and JSON sidecars are treated as trusted release
artifacts. Loading verifies checksums, metadata checksums, tensor names, shapes,
dtypes, and supported architecture contracts. Do not bypass these checks or
replace bundled artifacts with files from an untrusted source.

### Source PyTorch checkpoints

Checkpoint conversion is optional and requires PyTorch deserialization. Even
with `weights_only=True`, source checkpoints must come from a trusted,
hash-verified source. Never convert an unsolicited `.tar` file in a privileged
or sensitive environment.

### Persistent JAX compilation cache

JAX compilation-cache entries are executable artifacts. A process must only use
a cache directory writable by the same trusted principal. Never share a
world-writable cache between users, tenants, containers, or services.

### Audio and CSV input

Audio decoders and dataframe parsers process caller-controlled files. Services
embedding this library should add request-size, duration, file-count, timeout,
concurrency, and storage limits at the application boundary. This package is an
inference library, not a network-service security boundary.

### Dependencies and GPU runtimes

CPU and CUDA dependencies are resolver-managed from Python package indexes.
Use isolated environments, preserve the tested direct pins, inspect lock or
resolver output for deployments, and scan the final installed environment.
Do not mix CPU and CUDA requirements in one long-lived environment.

## Scope limitations

Model-quality concerns, score calibration, bias in upstream training data, and
the non-commercial model-weight license are important deployment risks but are
not software vulnerabilities. They should still be assessed before using
predictions in consequential decisions.
