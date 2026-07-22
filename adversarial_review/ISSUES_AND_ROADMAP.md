# Detected Issues & Improvement Roadmap

## Detected Issues

### ISSUE-01 — CSV column names incompatible with PyTorch [MEDIUM]
**Status:** Confirmed on CPU + GPU
**Impact:** Breaks any downstream pipeline expecting PyTorch-compatible `NISQA_results.csv`

The port writes columns `[deg, mos, noi, dis, col, loud]`. PyTorch writes `[deg, mos_pred, noi_pred, dis_pred, col_pred, loud_pred]` plus a `model` column. Root cause: `predict.py:_format_prediction` uses `output_names` (`mos`, `noi`, …) for both the dict API *and* the CSV columns.

| Port | PyTorch |
|---|---|
| `deg, mos, noi, dis, col, loud` | `deg, mos_pred, noi_pred, dis_pred, col_pred, loud_pred, model` |

**Fix:** Append `_pred` suffix to column names in CSV output. Add `model` column with checkpoint stem name. Keep the dict API keys as-is (`mos`, `noi`, …) for programmatic use.

---

### ISSUE-02 — Persistent compilation cache non-functional [MEDIUM]
**Status:** Confirmed on GPU
**Impact:** Every process restart pays full ~0.88s cold compile per shape; no cross-process caching

`load_model` calls `compilation_cache.initialize_cache(cache_dir)` when `cache_dir` is passed, but the cache directory stays **empty** after multiple compiles. Measured:

```
no_cache (true cold):    0.879s
cache_cold (1st load):   0.265s   ← in-process XLA cache, NOT persistent
cache_warm (2nd load):   0.250s
cache_warm (3rd load):   0.252s   ← 1.06x (no benefit)
cache dir contents: []            ← never written to disk
in-process same-instance: 0.25s → 0.0019s  (130x — works fine)
```

**Fix:** Set `JAX_COMPILATION_CACHE_DIR` env var *before* first JAX import, or use `jax.config.update('jax_compilation_cache_dir', path)`. Ensure the JIT'd function closure is stable across reloads (same module-level function, not a per-instance lambda).

---

### ISSUE-03 — `td_sa_nhead > 1` silently accepted but ignored [latent HIGH]
**Status:** Confirmed on CPU
**Impact:** Wrong results for custom checkpoints with multi-head attention (shipped checkpoints use nhead=1, so currently safe)

`model.py` computes attention as a single head (`dim = d_model`, scale `1/sqrt(d_model)`, no head reshape). PyTorch `MultiheadAttention` splits into `nhead` heads of `d_model/nhead` and scales by `1/sqrt(head_dim)`. `config_from_checkpoint_args` accepts any `nhead` without validating `==1`.

**Fix:** Either (a) validate `td_sa_nhead == 1` in config and reject `nhead > 1` with a clear error, or (b) implement proper multi-head attention with head reshape and `1/sqrt(head_dim)` scaling. Option (b) is preferred for broader checkpoint support.

---

### ISSUE-04 — TTS output named `naturalness` vs PyTorch `mos_pred` [LOW-MEDIUM]
**Status:** Confirmed
**Impact:** TTS CSV column incompatible with PyTorch

`config.py` sets `output_names = ("naturalness",)` for `nisqa_tts.tar`, but PyTorch's `predict_mos` always writes `mos_pred`.

**Fix:** Use `mos_pred` for CSV output. Expose `naturalness` as an optional alias in the dict API only.

---

### ISSUE-05 — `predict_batch([])` silently returns empty [LOW]
**Status:** Confirmed
**Impact:** Inconsistent failure mode (CLI raises "No wav files found", library API silently returns empty)

**Fix:** Raise `ValueError("No wav files provided")` at start of `predict_batch` if `len(wav_paths) == 0`.

---

### ISSUE-06 — Stereo out-of-range channel gives misleading error [LOW]
**Status:** Confirmed
**Impact:** Bad UX — user sees "Could not load file" when the real problem is an invalid channel index

`features.py` catches `IndexError` from `y[channel, :]` and rewraps it as a load error.

**Fix:** Check `channel < y.shape[0]` before indexing; raise `ValueError(f"Channel {channel} out of range for file with {y.shape[0]} channels")`.

---

### ISSUE-07 — TF32 matmul precision never enabled on GPU [LOW, perf]
**Status:** Confirmed
**Impact:** Leaves tensor-core throughput on the table on Ampere+ GPUs

Model `__post_init__` wraps forward in `jax.default_matmul_precision("float32")` unconditionally. Measured TF32 drift = 0.0000 (these matmuls are tiny — d_model=64 — so TF32 rounds identically), meaning TF32 is *safe* but never enabled.

**Fix:** Add `precision="tf32"` option that uses `jax.default_matmul_precision("tensorfloat32")` on GPU only. Default to `"float32"` for strict parity, document `"tf32"` as the fast path.

---

### ISSUE-08 — TTS-LSTM float32 parity borderline at ~1.2e-3 [LOW]
**Status:** Confirmed on GPU
**Impact:** 2 cases technically exceed the 1e-3 acceptance threshold (not a logic bug)

2 float32 TTS cases at ~1.2e-3 (bs=32/sl=64 normal, bs=8/sl=128 large). Caused by cuDNN LSTM (fused kernel, different accumulation order) vs `jax.lax.scan` (explicit per-timestep). 0.024% relative error on MOS scale 1–5.

**Fix:** Widen TTS parity tolerance to 2e-3 in tests, or investigate cuDNN-compatible accumulation order (likely not worth the complexity).

---

### ISSUE-09 — Per-shape JIT recompilation with no static-shape cache [LOW, perf]
**Status:** Confirmed
**Impact:** Variable-length batched inference pays ~0.3–0.7s recompile per unique shape

Each new `(batch_size, seq_len)` retraces. Mitigated by ISSUE-02 fix (persistent cache would amortize across processes).

**Fix:** Use padded static shapes with mask, or document that users should batch-pad to fixed lengths. Alternatively, implement shape-padding to a small set of canonical shapes (e.g., powers of 2 for seq_len).

---

### ISSUE-10 — GPU memory pressure at bs=8/full-1300 on 8GB [LOW]
**Status:** Confirmed
**Impact:** OOM warnings on low-memory GPUs at max batch/length

**Fix:** Add automatic batch-size reduction or memory-aware batching. Document recommended batch sizes per GPU memory tier (e.g., bs=4 for 8GB, bs=8 for 12GB, bs=16 for 24GB at full-1300).

---

## Improvement Roadmap — "Much Better Than PyTorch"

The port already beats PyTorch 1.5×–7.2× on GPU with ~1e-7 parity. To make it decisively superior across all dimensions, here's the prioritized roadmap:

### Tier 1: Fix What's Broken (do first)

| # | Issue | Effort | Impact |
|---|---|---|---|
| 1 | **CSV compatibility** (ISSUE-01, 04) — `*_pred` columns + `model` column | 30 min | Unblocks PyTorch-compatible drop-in replacement |
| 2 | **Persistent compilation cache** (ISSUE-02) — actually persist to disk | 1 hr | Eliminates 0.88s cold-start per shape per process |
| 3 | **nhead validation** (ISSUE-03) — reject nhead>1 or implement MHA | 2 hr (validate) / 4 hr (implement) | Prevents silent wrong results on custom checkpoints |
| 4 | **Error messages** (ISSUE-05, 06) — empty batch + stereo OOB | 30 min | Consistent fail-fast behavior |

### Tier 2: Performance Gains (push the speedup further)

| # | Improvement | Est. speedup | Effort |
|---|---|---|---|
| 5 | **TF32 opt-in** (ISSUE-07) — `precision="tf32"` on Ampere+ | 1.3–1.5× on self-att matmuls | 1 hr |
| 6 | **Static-shape padding** (ISSUE-09) — pad to canonical shapes, mask invalid | Eliminates recompiles in batched inference | 3 hr |
| 7 | **CUDA Graphs** — `jax.experimental.pjit` or `jax.jit` with `donated_buffers` | Reduces per-call overhead, frees intermediates | 2 hr |
| 8 | **fp16 inference path** — add alongside bf16 (some GPUs prefer fp16) | 1.5–2× over float32 on tensor cores | 2 hr |
| 9 | **Batched preprocessing on GPU** — move librosa mel-spec to JAX (`jax.scipy.fft`) | Eliminates CPU bottleneck, end-to-end GPU pipeline | 1–2 days |
| 10 | **Memory-aware batching** (ISSUE-10) — auto-reduce bs on OOM | Enables safe deployment on 4–8GB GPUs | 2 hr |

### Tier 3: Feature Parity + Beyond (make it a superset of PyTorch NISQA)

| # | Feature | Value | Effort |
|---|---|---|---|
| 11 | **Multi-head attention** (ISSUE-03 full fix) — proper MHA with head reshape | Supports custom checkpoints, future-proof | 4 hr |
| 12 | **NISQA_DE support** — double-ended model | Covers the full NISQA model family | 1–2 days |
| 13 | **Remaining pool modes** — PoolAvg, PoolMax, PoolLastStep | Full pool compatibility | 2 hr |
| 14 | **Remaining CNN modes** — SkipCNN, DFF | Full CNN compatibility | 4 hr |
| 15 | **ONNX export** — export JAX model to ONNX via `jax2onnx` | Cross-framework deployment | 1 day |
| 16 | **Batched streaming inference** — process audio chunks with stateful LSTM carry | Real-time quality monitoring | 1–2 days |
| 17 | **AOT compilation** — `jax.export` / `jax.stablehlo` for hermetic deployment | No JAX dependency at inference time, sub-ms startup | 1 day |

### Tier 4: Quality of Life

| # | Feature | Value | Effort |
|---|---|---|---|
| 18 | **Python API for batch prediction with progress bar** — `predict_batch` with tqdm | UX for large datasets | 30 min |
| 19 | **Automatic checkpoint download** — fetch from HuggingFace if not local | Zero-config setup | 2 hr |
| 20 | **Docker image** — `nisqa-jax:gpu` with pre-compiled cache baked in | One-command deployment | 2 hr |
| 21 | **REST/gRPC microservice** — FastAPI wrapper with batched inference | Production serving | 4 hr |
| 22 | **Benchmark CI gate** — run `nisqa_bench.py` on PRs, fail if speedup < 2× | Prevents perf regressions | 2 hr |

### Suggested Priority Order

```
Week 1: Tier 1 (issues 1-4) + Tier 2 #5 (TF32) + #7 (CUDA Graphs)
         → Fixes all bugs, pushes speedup to ~4-10×, zero cold-start cost
Week 2: Tier 2 #6 (static shapes) + #9 (GPU preprocessing) + #10 (memory-aware)
         → End-to-end GPU pipeline, no CPU bottleneck, safe on any GPU
Week 3: Tier 3 #11 (MHA) + #12 (NISQA_DE) + #13-14 (remaining modules)
         → Full PyTorch feature parity, superset capability
Week 4: Tier 3 #17 (AOT) + Tier 4 #19-21 (download, Docker, service)
         → Production deployment story
```

### Expected End State

After completing Tiers 1–2, the port would be:
- **5–10× faster** than PyTorch on GPU (TF32 + CUDA Graphs + no recompiles)
- **Zero cold-start** penalty (working persistent cache)
- **End-to-end GPU** (no CPU preprocessing bottleneck)
- **PyTorch-compatible** CSV output (drop-in replacement)
- **Safe on any GPU** (memory-aware batching)

After completing Tier 3, it would be:
- **Full feature superset** of PyTorch NISQA (all architectures, all pool modes, DE support)
- **AOT-deployable** without JAX at inference time
- **Exportable to ONNX** for cross-framework use

This would make the JAX port not just faster, but a strict upgrade over PyTorch NISQA in every dimension: speed, correctness, deployment, and feature coverage.
