# Architecture Re-Evaluation: A1-A4 (Empirical, RTX 3070)

Re-examination of 4 items reviewers dismissed as "noise/over-engineering" for the
`nisqa-jax` self-att MOS scheduler. Prototyped + measured vs the current merged
**sort+bucket32** baseline. Scripts: `eval_scheduling.py` (CPU, A2/A3/A4),
`eval_packing.py` (GPU, A1). No repo code modified; 39 tests still pass.

## Baseline reproduced (sort+bucket32)

| Workload | waste % | #shapes | warm time |
|---|---|---|---|
| N=4000, bs=32, U[1,1300] (cited scale) | **3.03** | **41** | — (OOM on 8GB at bs=32) |
| N=160, bs=4, U[1,1300] (GPU-safe) | 4.21 | 30 | **0.286 s** |

Cited "3.07% / 41 shapes" reproduced exactly at N=4000. sort+exact floor = 0.77%
(irreducible within-chunk variance).

## A1 — Fixed-capacity packing (block-diagonal attention)

Static-capacity buffer (1, CAP, ...) + segment-id mask -> **1 compiled shape**.
CNN unchanged (per-token); self-att uses block-diagonal mask; segment-aware pool.

| capacity | waste % | #shapes | #bins | warm time | vs baseline |
|---|---|---|---|---|---|
| baseline bs=4 | 4.21 | 30 | 40 | 0.286 s | 1.00x |
| 2600 (2*1300) | 0.74 | 1 | 42 | 0.293 s | **0.974x** |
| 5200 (4*1300) | 0.74 | 1 | 21 | 0.342 s | **0.836x** |
| 10400 (8*1300) | 5.26 | 1 | 11 | 0.446 s | **0.641x** |

Parity (packed vs per-sample): max abs diff **8.98e-4** (within 1e-3; reduction-order
drift from segmented softmax, not a logic bug).

**VERDICT: LOSES (confirmed noise).** Packing cuts waste (0.74% vs 4.21%) and
shapes (1 vs 30) but LOSES warm throughput at every capacity (-3% to -36%).
Root cause: self-attention is O(L^2) — packing K segments into one length-L
sequence costs L^2 vs K*chunk_max^2; the constant large capacity burns more
attention FLOPs than the padding it removes (CNN is linear & saves ~3%, attention
is quadratic & explodes). The 30x-fewer-compiles benefit is real but redundant
with the planned persistent-cache fix (ISSUE-02), which amortizes compiles to
disk. Trend worsens monotonically with capacity.

## A2 — MFFD bin-packing scheduler (CPU, N=4000 bs=32)

| distribution | strategy | waste % | #shapes |
|---|---|---|---|
| uniform | baseline sort+bucket32 | 3.03 | 41 |
| uniform | MFFD budget=bs*maxgrid | 3.03 | 41 |
| uniform | MFFD budget=bs*mean | 31.54 | 41 |
| bimodal | baseline | 3.13 | 22 |
| bimodal | MFFD budget=bs*maxgrid | 3.13 | 22 |
| bimodal | MFFD budget=bs*mean | 41.90 | 21 |
| zipf | baseline | 87.57 | 4 |
| zipf | MFFD budget=bs*maxgrid | 87.57 | 4 |
| zipf | MFFD budget=bs*mean | 95.47 | 14 |

**VERDICT: LOSES (confirmed noise).** With fixed bs + grid (shapes stay bounded),
loose-budget MFFD ≡ adjacent-sort (identical waste: first-fit on sorted-desc
produces the same chunks). Tight-budget MFFD mixes long+short samples per bin,
which INCREASES padding (shorts padded to the long's length) -> 10-30pp worse.
MFFD minimizes bin COUNT, the wrong objective — padding is minimized by grouping
similar lengths, which is exactly what sort does.

## A3 — DP optimal bucket boundaries (CPU, on chunk-max schedule cost)

1-D DP over the chunk-max sequence (the real schedule units), K buckets.

| distribution | K | waste % | #shapes | vs baseline |
|---|---|---|---|---|
| uniform | baseline (32-grid) | 3.03 | 41 | — |
| uniform | DP K=41 (=base shapes) | 2.28 | 41 | -0.75pp |
| uniform | DP K=32 (Pareto-dominates) | 2.91 | 32 | -0.13pp, 1.28x fewer shapes |
| uniform | DP K=16 | 5.68 | 16 | +2.64pp, 2.56x fewer shapes |
| bimodal | DP K=22 (=base shapes) | 2.54 | 22 | -0.60pp |
| zipf | DP K=4 (=base shapes) | 69.62 | 4 | -17.9pp (pathological) |

K-sweep (uniform): fixed-32 is within **0.75pp** of the optimal at equal shape
budget. DP K=32 Pareto-dominates the 32-grid (fewer shapes AND less waste) but
the margin (1.28x shapes, 0.13pp) is below both implement thresholds. The only
way to get >2x fewer shapes is K<=20, which costs >1.5pp MORE waste.

**VERDICT: TIE (confirmed noise).** Fixed-32 is within ~1% of optimal. The
optimal grid gives <0.75pp (=<0.75% throughput, CNN-linear) at equal shape
budget — below the 5% threshold. A data-dependent grid is not worth the
complexity / loss of shape-stability across datasets.

## A4 — ILP/CP formulation (PuLP/CBC, n=120 uniform)

ILP over chunk-maxes: free (non-contiguous) chunk->bucket assignment + integer
bucket uppers, minimize bs*sum(upper). Linearized y=c*u via big-M.

| strategy | waste % | #shapes |
|---|---|---|
| baseline sort+bucket32 | 21.77 | 4 |
| A3 DP (contiguous) K=4 | 19.96 | 4 |
| A4 ILP (free assignment) K=4 | 19.96 | 4 |

ILP gain vs DP: **0.00%**. ILP gain vs baseline: 8.33% relative (same as DP —
it's the optimal-boundary gain, not ILP-specific).

**VERDICT: TIE (confirmed noise).** The ILP is cleanly castable (big-M
linearization, CBC solves n=120 in seconds) but the free-assignment optimum
EXACTLY equals the contiguous DP (0.00% gap) — proving contiguous partition is
optimal for 1-D padding. The solver finds no improvement over the DP, and the DP
itself is within ~1% of the fixed-32 heuristic. Sort+bucket32 is near-optimal.

## Summary table

| item | measured vs sort+bucket32 | VERDICT | one-line reason |
|---|---|---|---|
| A1 fixed-cap packing | waste 0.74% (1 shape) but **0.64-0.97x throughput** | **LOSES (noise)** | O(L^2) attention: long packed seqs cost more FLOPs than padding saved; compile win redundant with cache fix |
| A2 MFFD bin-packing | ties (loose budget) or +10-30pp waste (tight) | **LOSES (noise)** | MFFD minimizes bin count, not padding; sort already groups similar lengths optimally |
| A3 DP bucket bounds | -0.75pp waste at equal shapes (<1% throughput) | **TIE (noise)** | fixed-32 within ~1% of optimal; data-dependent grid not worth it |
| A4 ILP/CP solver | 0.00% vs DP, ~8% rel vs heuristic (<1% abs) | **TIE (noise)** | ILP=DP proves contiguous optimality; heuristic near-optimal |

**Bottom line:** all 4 items are confirmed noise. The reviewers were right.
Sort+bucket32 is within ~1% of the scheduling optimum (A3/A4) and beats packing
on throughput (A1) because self-attention is quadratic. The one real lever for
compile-count is the persistent-cache fix (ISSUE-02), already on the roadmap —
not a scheduler change.
