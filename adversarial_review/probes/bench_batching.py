#!/usr/bin/env python
"""Benchmark: length-aware batching strategies for nisqa-jax.

Measures, on the RTX 3070 with the self-attention MOS checkpoint, for a synthetic
workload of N files with n_wins ~ Uniform(1, 1300):
  (a) naive in-order batching
  (b) stable length sort + exact chunk-max crop
  (c) stable length sort + bucket-32 grid (round chunk-max up to mult of 32)
Reports padding-waste %, unique compiled (batch, steps) shapes, warm wall-time,
and a per-sample parity proof (sorted-batched vs per-sample, max abs diff).
Also estimates host-RAM before/after for a 1000-file TTS-style batch.

Run with the GPU interpreter: /tmp/nisqa_gpu/bin/python bench_batching.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from nisqa_jax.checkpoint import load_model
from nisqa_jax.config import FeatureConfig
from nisqa_jax.predict import _round_up, default_length_bucket

WEIGHTS = Path(__file__).resolve().parents[2] / "nisqa_jax" / "weights" / "nisqa_mos_only.npz"


def _round_up_local(value: int, bucket: int) -> int:
    return _round_up(value, bucket)


def synthesize_workload(n: int, lo: int, hi: int, feat: FeatureConfig, seed: int = 0) -> list[tuple[np.ndarray, int]]:
    """Build N synthetic (segments, n_wins) with n_wins ~ Uniform[lo, hi].

    Segments are random float32 [n_wins, 1, n_mels, seg_length] -- avoids librosa.
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, int]] = []
    # uniform inclusive on [lo, hi]
    lengths = rng.integers(lo, hi + 1, size=n)
    for nw in lengths:
        seg = rng.standard_normal((int(nw), 1, feat.n_mels, feat.seg_length)).astype(np.float32) * 0.1
        out.append((seg, int(nw)))
    return out


def schedule_chunks(n_wins: list[int], batch_size: int, sort: bool, bucket: int) -> list[list[int]]:
    """Return list of chunks (each a list of original indices) per strategy.

    sort=False -> naive in-order; sort=True -> stable descending length sort.
    bucket>1 -> chunk-max rounded up to grid (recorded via padded length later).
    """
    if sort:
        order = sorted(range(len(n_wins)), key=lambda i: n_wins[i], reverse=True)
    else:
        order = list(range(len(n_wins)))
    return [order[s : s + batch_size] for s in range(0, len(order), batch_size)]


def _padded_steps(chunk_n_wins: list[int], bucket: int) -> int:
    return _round_up_local(max(chunk_n_wins), bucket)


def measure_waste_and_shapes(n_wins: list[int], batch_size: int, sort: bool, bucket: int) -> tuple[float, int]:
    """Padding-waste % and count of unique compiled (batch, steps) shapes."""
    chunks = schedule_chunks(n_wins, batch_size, sort, bucket)
    real_slots = 0
    padded_slots = 0
    shapes: set[tuple[int, int]] = set()
    for chunk in chunks:
        cnw = [n_wins[i] for i in chunk]
        steps = _padded_steps(cnw, bucket)
        bsz = batch_size  # remainder padded to batch_size (fixed remainder padding)
        shapes.add((bsz, steps))
        padded_slots += bsz * steps
        real_slots += sum(cnw)
    waste = 100.0 * (padded_slots - real_slots) / padded_slots if padded_slots else 0.0
    return waste, len(shapes)


def run_strategy(model, workload: list[tuple[np.ndarray, int]], batch_size: int,
                 sort: bool, bucket: int) -> np.ndarray:
    """Execute a strategy and return outputs in ORIGINAL order [N, n_out]."""
    n = len(workload)
    n_wins = [w[1] for w in workload]
    chunks = schedule_chunks(n_wins, batch_size, sort, bucket)
    feat = model.config.feature
    n_out = len(model.config.output_names)
    results = np.zeros((n, n_out), dtype=np.float32)
    for chunk in chunks:
        prepared = [workload[i] for i in chunk]
        actual_n = np.asarray([w[1] for w in prepared], dtype=np.int32)
        steps = _padded_steps([int(x) for x in actual_n], bucket)
        # fixed remainder padding: repeat last sample to fill batch_size
        bsz = len(prepared)
        if bsz < batch_size:
            prepared = list(prepared) + [prepared[-1]] * (batch_size - bsz)
            actual_n = np.concatenate([actual_n, np.asarray([1] * (batch_size - bsz), dtype=np.int32)])
        x = np.zeros((len(prepared), steps, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
        for j, (seg, nw) in enumerate(prepared):
            x[j, :nw] = seg
        out = model.predict_segments(x, actual_n)
        for j, orig in enumerate(chunk):
            results[orig] = np.asarray(out[j], dtype=np.float32)
    return results


def per_sample_reference(model, workload: list[tuple[np.ndarray, int]], indices: list[int]) -> np.ndarray:
    """Run each selected sample alone (batch=1, exact n_wins) -- the parity reference.

    Subsampled because each unique n_wins triggers a separate JIT compile.
    """
    feat = model.config.feature
    n_out = len(model.config.output_names)
    out = np.zeros((len(indices), n_out), dtype=np.float32)
    for k, i in enumerate(indices):
        seg, nw = workload[i]
        x = np.zeros((1, nw, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
        x[0, :nw] = seg
        out[k] = np.asarray(model.predict_segments(x, np.asarray([nw], dtype=np.int32))[0])
    return out


def time_strategy(model, workload, batch_size, sort, bucket, repeats: int = 3) -> float:
    """Warm wall-time (best of repeats) for the full workload."""
    # warmup
    run_strategy(model, workload, batch_size, sort, bucket)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_strategy(model, workload, batch_size, sort, bucket)
        best = min(best, time.perf_counter() - t0)
    return best


def estimate_ram(n_files: int, feat: FeatureConfig, max_segments: int) -> None:
    """Host-RAM for holding preprocessed segments: padded (old) vs unpadded (new).

    Assumes average n_wins = max_segments/2 (uniform length distribution).
    """
    bytes_per_step = 1 * feat.n_mels * feat.seg_length * 4  # float32
    old_padded = n_files * max_segments * bytes_per_step
    avg_nw = max_segments // 2
    new_unpadded = n_files * avg_nw * bytes_per_step
    print(f"  host-RAM (old, padded to max_segments): {old_padded / 1e9:.2f} GB")
    print(f"  host-RAM (new, unpadded real segments): {new_unpadded / 1e9:.2f} GB")
    print(f"  reduction: {(old_padded - new_unpadded) / old_padded * 100:.1f}% "
          f"({(old_padded - new_unpadded) / 1e9:.2f} GB saved)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=1300)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = load_model(WEIGHTS, device=args.device, precision="float32")
    feat = model.config.feature
    bucket_default = default_length_bucket(model.config)
    print(f"model: {model.config.model_name} td={model.config.td} max_segments={feat.max_segments} "
          f"seg_length={feat.seg_length} seg_hop={feat.seg_hop_length} n_mels={feat.n_mels}")
    print(f"default_length_bucket={bucket_default}  N={args.n} bs={args.bs} lengths~U[{args.lo},{args.hi}]")

    workload = synthesize_workload(args.n, args.lo, args.hi, feat)
    n_wins = [w[1] for w in workload]

    strategies = [
        ("naive in-order", False, 1),
        ("sort + exact", True, 1),
        (f"sort + bucket{bucket_default}", True, bucket_default),
    ]

    print("\n=== Padding waste & unique compiled shapes ===")
    print(f"{'strategy':<22} {'waste %':>8} {'#shapes':>8}")
    for name, sort, bucket in strategies:
        waste, nshapes = measure_waste_and_shapes(n_wins, args.bs, sort, bucket)
        print(f"{name:<22} {waste:8.2f} {nshapes:8d}")

    print("\n=== Warm wall-time (best of 3, same workload) ===")
    print(f"{'strategy':<22} {'time (s)':>10}")
    times = {}
    for name, sort, bucket in strategies:
        t = time_strategy(model, workload, args.bs, sort, bucket)
        times[name] = t
        print(f"{name:<22} {t:10.3f}")

    print("\n=== Parity: sorted-bucket vs per-sample reference ===")
    # Subsample for parity (each unique n_wins compiles a fresh shape).
    rng = np.random.default_rng(123)
    parity_idx = sorted(set(int(x) for x in rng.integers(0, args.n, size=min(24, args.n))))
    ref = per_sample_reference(model, workload, parity_idx)
    sorted_full = run_strategy(model, workload, args.bs, sort=True, bucket=bucket_default)
    exact_full = run_strategy(model, workload, args.bs, sort=True, bucket=1)
    sorted_sub = sorted_full[parity_idx]
    exact_sub = exact_full[parity_idx]
    max_abs = float(np.max(np.abs(sorted_sub - ref)))
    max_abs_exact = float(np.max(np.abs(exact_sub - ref)))
    print(f"  parity samples: {len(parity_idx)} (n_wins: {[workload[i][1] for i in parity_idx]})")
    print(f"  max abs diff (sorted+bucket{bucket_default} vs per-sample): {max_abs:.3e}")
    print(f"  max abs diff (sort+exact vs per-sample):           {max_abs_exact:.3e}")
    print(f"  parity within 1e-6: {max_abs <= 1e-6 and max_abs_exact <= 1e-6}")

    print("\n=== Host-RAM: 1000-file TTS-style batch (max_segments=6000) ===")
    tts_feat = feat  # n_mels/seg_length same; max_segments differs
    estimate_ram(1000, tts_feat, max_segments=6000)
    print("  (self-att max_segments=1300 for reference:)")
    estimate_ram(1000, tts_feat, max_segments=1300)

    print("\n=== Summary ===")
    base = times["naive in-order"]
    for name, _, _ in strategies:
        print(f"  {name:<22} {times[name]:.3f}s  ({base / times[name]:.2f}x vs naive)")


if __name__ == "__main__":
    main()
