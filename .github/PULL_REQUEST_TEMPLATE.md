## Summary

<!-- What changed and why? -->

## User-visible behavior

<!-- Describe changed APIs, CLI output, compatibility, artifacts, or performance. Write "None" when appropriate. -->

## Validation

<!-- List exact commands and results. Do not write "CI passed" unless it actually completed successfully. -->

- [ ] Ruff
- [ ] Mypy
- [ ] Strict artifact verification
- [ ] Golden parity
- [ ] Frozen audio-frontend regression
- [ ] Relevant focused and full tests
- [ ] Build and package-content gate
- [ ] Clean-room install when packaging changed
- [ ] CPU/CUDA checks appropriate to the change

## Risk and compatibility

<!-- Cover dependency pairs, Python/JAX/NumPy support, rollout, artifacts, licenses, and rollback. -->

## Checklist

- [ ] The change is focused and contains no unrelated generated or formatting churn.
- [ ] Tests can detect a plausible broken implementation.
- [ ] README and supporting docs match runtime behavior.
- [ ] `CHANGELOG.md` covers user-visible changes.
- [ ] New required sdist files are protected by the package-content gate.
- [ ] No credentials, user audio, source checkpoints, local paths, or caches were committed.
