# Adversarial Review Findings: Current Status

This file is the status index for issues found by the original adversarial
review. The original measurements and probes remain in this directory for
provenance, but their initial "detected issue" wording is historical and must
not be read as the current runtime state. Current behavior is documented in the
top-level `README.md` and enforced by the test suite.

## Resolved or mitigated findings

| ID | Original concern | Current status | Repository evidence |
|---|---|---|---|
| ISSUE-01 | CSV columns differed from PyTorch | **Resolved.** CLI `predict_csv` retains source columns, while batch/CSV prediction output emits `*_pred` plus `model`; the dictionary API keeps semantic names. | `nisqa_jax/predict.py`; `tests/test_jax_port.py`; `tests/test_review_predict_frontend.py` |
| ISSUE-02 | Persistent compilation cache was not configured | **Resolved.** `cache_dir` configures JAX's persistent cache before compilation and lowers the default minimum compile-time threshold. | `nisqa_jax/checkpoint.py`; `tests/test_prewarm.py` |
| ISSUE-03 | Multi-head attention was silently accepted | **Resolved for the stated three-checkpoint scope.** Unsupported `td_sa_nhead > 1` is rejected at configuration load. Multi-head implementation remains intentionally out of scope. | `nisqa_jax/config.py`; `tests/test_validation.py` |
| ISSUE-04 | TTS CSV output used `naturalness` instead of `mos_pred` | **Resolved.** CSV compatibility maps the semantic `naturalness` result to `mos_pred`. | `nisqa_jax/predict.py`; `tests/test_jax_port.py` |
| ISSUE-05 | Empty library batches returned silently | **Resolved.** `predict_batch([])` raises `ValueError("No wav files provided")`. | `nisqa_jax/predict.py`; `tests/test_validation.py` |
| ISSUE-06 | Invalid stereo channel errors were misleading | **Resolved.** Channel bounds are validated explicitly and the error identifies the file and available channel count. | `nisqa_jax/features.py`; `tests/test_validation.py` |
| ISSUE-07 | No TF32 fast path | **Closed as a non-requirement.** Strict float32 remains the conformance default; bf16 is the explicit reduced-precision path. A speculative TF32 mode is not required for production correctness. | `nisqa_jax/model.py`; `README.md` |
| ISSUE-08 | TTS GPU parity appeared to exceed `1e-3` | **Resolved as a reference-method issue.** Current parity uses the PyTorch CPU/f64-truth path; the former cuDNN accumulation drift is documented rather than hidden by widening the tolerance. | `README.md`; golden/live parity tests |
| ISSUE-09 | Every exact variable length could trigger a new JIT shape | **Mitigated.** Length-aware sorting, bucket-rounded padding, persistent caching, and `prewarm` constrain and amortize the shape set. Exact-shape mode remains available explicitly. | `nisqa_jax/predict.py`; `tests/test_batching_fixes.py`; `tests/test_prewarm.py` |
| ISSUE-10 | Full-length batches could OOM an 8 GB GPU | **Mitigated.** Recommended memory-tier batch sizes are documented and `auto_batch` can halve a failed batch down to one sample. Hardware capacity is still a deployment constraint, not a defect the library can eliminate. | `nisqa_jax/predict.py`; `tests/test_jax_port.py`; `README.md` |

## Remaining optional work

The following are possible extensions, not known merge blockers:

- GPU-native audio preprocessing, if profiling shows Librosa is the dominant
  end-to-end cost and exact frontend parity can be retained.
- Additional checkpoint architectures (multi-head attention, NISQA_DE, other
  pool/CNN modes) only if project scope expands beyond the three shipped
  checkpoints.
- GPU performance CI when a stable CUDA runner is available.
- Deployment products such as containers or a service wrapper in separate,
  requirement-driven changes.

Claims such as "zero cold-start", "safe on any GPU", or projected 5–10x
speedups are intentionally omitted: persistent caching still has a first
producer, finite devices can still OOM, and performance must be reported from a
representative measured workload.
