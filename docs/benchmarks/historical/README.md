# Historical JAX 0.4.30 benchmark evidence

These results were recorded on 2026-07-22 using an NVIDIA RTX 3070, driver
595.84, JAX/jaxlib 0.4.30, NumPy 1.26.4, and CUDA PyTorch. They do not describe
the current JAX 0.6.2 implementation.

Two historical grids used different revisions and PyTorch baselines:

| Evidence | Self-attention | TTS/BiLSTM |
|---|---|---|
| Original eager-PyTorch review | 3.04× (`mos`), 3.43× (`dim`) median | 1.96× median |
| Later optimized-PyTorch grid | 3.11× vs eager, 2.27× vs CUDA graphs (`mos`) | 0.96× vs eager, 0.97× vs `torch.compile` |

Speedup is PyTorch latency divided by JAX latency. Measurements are warmed
model-forward latency with compilation excluded and inputs pre-staged on the
GPU. The contradictory TTS values are why they must not be collapsed into a
single headline.

Retained raw evidence:

- [`jax-0.4.30-eager-results.json`](jax-0.4.30-eager-results.json): original
  per-case eager-PyTorch comparison.
- [`jax-0.4.30-optimized-pytorch.txt`](jax-0.4.30-optimized-pytorch.txt):
  later nine-shape optimized-PyTorch grid and environment notes.

The one-off probe programs and resolved issue inventory were removed from the
active tree. Git history preserves them if historical reconstruction is ever
needed.
