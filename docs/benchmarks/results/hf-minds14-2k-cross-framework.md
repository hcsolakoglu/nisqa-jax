# Minds14 cross-framework benchmark

This report compares the optimized JAX implementation with the separate modern
PyTorch working tree on the same real-audio manifest. It is a runtime and
numerical-parity experiment, not a perceptual-quality leaderboard.

## Canonical run

- Dataset: `PolyAI/minds14`, `train` split.
- Requested configurations: `en-US`, `en-GB`, `de-DE`, `fr-FR`, `es-ES`.
- Contributing configurations at the resolved revision: `en-US`, `en-GB`,
  `de-DE`, `fr-FR`; `es-ES` contributed no rows.
- Resolved Hub revision: `40ce77cb32a384e4d50a568e1ec39ac804019d33`.
- Materialized examples: `2000` real WAV files at `8 kHz`.
- Rejected examples over the common self-attention limit: `3`.
- Sample manifest SHA-256:
  `d9f74a260b487e6785b46df939557904697df52385008863c3be19282650f94c`.
- Audio duration: minimum `1.621375 s`, median `7.338625 s`, mean
  `9.0693585625 s`, p95 `19.91 s`, maximum `45.41825 s`.
- Frontend lane: `native` for both products.
- Scheduler: stable length order, `fixed` mode, serial decode
  (`--decode-threads 0`), four preprocessing workers.
- Self-attention batch size: `4`.
- TTS batch size: `1`. This is a common batch size for both engines and is a
  documented RTX 3070 memory constraint. Batch-four TTS made the co-resident
  JAX/PyTorch process run out of CUDA memory during convolution.
- JAX memory policy: `XLA_PYTHON_CLIENT_PREALLOCATE=false`; allocator unset.
- Precision: float32.

The self-attention and TTS rows therefore compare each model pair under the
same contract, but batch size must not be compared across model rows.

Raw JSON and cProfile artifacts:

- [`nisqa_mos_only`](hf-minds14-2k-cross-nisqa_mos_only.json)
- [`nisqa`](hf-minds14-2k-cross-nisqa.json)
- [`nisqa_tts`](hf-minds14-2k-cross-nisqa_tts.json)
- [`nisqa_mos_only` profile](hf-minds14-2k-cross-nisqa_mos_only.profile.txt)
- [`nisqa` profile](hf-minds14-2k-cross-nisqa.profile.txt)
- [`nisqa_tts` profile](hf-minds14-2k-cross-nisqa_tts.profile.txt)

## Lineage and environment

- JAX repository base commit recorded by the benchmark: `9d4190b`
  (`Fix bounded benchmark decode cleanup`), with `status_clean=false` because
  the scheduler and benchmark changes were still in the working tree when the
  run started.
- PyTorch source commit: `32c56382b854c61ba5e46be7745fcd5a444a426e`
  (`modernize pretrained inference path`), clean working tree.
- Python `3.12.12`.
- JAX/jaxlib `0.6.2`.
- PyTorch `2.13.0+cu130`, CUDA runtime `13.0`.
- GPU: NVIDIA GeForce RTX 3070, `8192 MiB`, driver `595.84`.
- `datasets 5.0.0`, `librosa 0.11.0`, NumPy `2.2.6`, SciPy `1.15.3`.

## End-to-end wall time

These are single-run measurements. They are evidence for this machine and
contract, not universal framework rankings.

| Model | JAX total | PyTorch total | Lower wall time | Difference relative to slower engine |
|---|---:|---:|---|---:|
| `nisqa_mos_only` | `128.160 s` | `110.008 s` | PyTorch | PyTorch `14.2%` lower |
| `nisqa` | `150.837 s` | `119.225 s` | PyTorch | PyTorch `21.0%` lower |
| `nisqa_tts` | `109.304 s` | `186.923 s` | JAX | JAX `41.5%` lower |

There is no single framework winner. PyTorch wins the two self-attention
product paths in this native end-to-end run. JAX wins the long-sequence TTS
path under the required batch-one memory constraint.

## Why PyTorch can still lead

The end-to-end number includes each product's native audio frontend, input
transfer, first-shape compilation, warmed forward, and output transfer. The
stage metrics explain why a faster model core does not automatically produce a
faster product path:

| Model | Engine | First forward | Warmed forward | Preprocess wait | Model-forward files/s |
|---|---|---:|---:|---:|---:|
| `nisqa_mos_only` | JAX | `13.116 s` | `2.389 s` | `108.014 s` | `128.99` |
| `nisqa_mos_only` | PyTorch | `0.677 s` | `18.659 s` | `87.735 s` | `103.44` |
| `nisqa` | JAX | `12.260 s` | `2.621 s` | `131.210 s` | `134.40` |
| `nisqa` | PyTorch | `0.638 s` | `19.498 s` | `95.315 s` | `99.32` |
| `nisqa_tts` | JAX | `42.274 s` | `11.852 s` | `48.525 s` | `36.95` |
| `nisqa_tts` | PyTorch | `0.712 s` | `33.538 s` | `147.782 s` | `58.39` |

JAX has the faster warmed model core for both self-attention models, but its
native preprocessing wait and first-shape compilation erase that advantage in
this single end-to-end run. TTS shows the opposite tradeoff: PyTorch's warmed
core is faster per model-forward file, while its native preprocessing wait is
much larger, so JAX wins total wall time. The `preprocess_worker_seconds` field
is the sum across workers and must not be interpreted as wall time; the table
uses `preprocess_wait_seconds`.

An ablation with `XLA_PYTHON_CLIENT_ALLOCATOR=platform` was intentionally not
used for the canonical result. It is a memory-reclamation fallback and added
substantial JAX allocation overhead, changing the self-attention winner. It is
not a fair replacement for the normal allocator policy in a performance claim.

## Correctness

The CPU real-audio parity subset passed for every shipped checkpoint with a
strict `5e-5` tolerance:

| Model | Samples | Max absolute difference | Result |
|---|---:|---:|---|
| `nisqa_mos_only` | `64` | `1.43e-6` | passed |
| `nisqa` | `64` | `3.34e-6` | passed |
| `nisqa_tts` | `64` | `1.43e-6` | passed |

All GPU output arrays were finite and shape-equal. Native frontend comparisons
are diagnostic only because the two products intentionally use independent
frontends. Their strict output threshold was not treated as a CUDA correctness
gate. Minds14 contains no NISQA MOS ground truth in this experiment, so these
results make no perceptual-quality claim.

## Decision

Keep both optimizations, but do not claim a universal JAX or PyTorch speedup.
The evidence supports a model- and pipeline-specific conclusion:

- PyTorch's optimized eager path remains the faster native end-to-end choice for
  the two self-attention checkpoints in this run.
- JAX's shape bucketing and long-sequence path make it faster for `nisqa_tts`
  under batch-one memory-safe execution.
- For further work, JAX should target native frontend wait time and first-shape
  compilation. PyTorch's self-attention path should target its warmed forward
  cost. Memory allocator settings must remain fixed when making future claims.
