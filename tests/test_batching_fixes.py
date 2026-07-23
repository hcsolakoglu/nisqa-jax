"""Lane B: batching / prewarm / benchmark / public-API reliability fixes.

Covers the finding ledger F1-F9: length-bucket JIT-shape survival, prewarm grid,
huge-batch DoS clamping, cost-aware batching, bench.py ragged padding, collect-
mode stable schema + path-bearing errors, CSV model-identity fallback, channel
type tightening + clear CSV-column errors, and non-finite input rejection.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))
MOS_ONLY = WEIGHTS_ROOT / "nisqa_mos_only.npz"
TTS = WEIGHTS_ROOT / "nisqa_tts.npz"

sys.path.insert(0, str(ROOT))

from nisqa_jax.checkpoint import load_model  # noqa: E402
from nisqa_jax.features import load_melspec, segment_melspec  # noqa: E402
from nisqa_jax.model import NisqaJaxModel  # noqa: E402
from nisqa_jax.predict import (  # noqa: E402
    _cost_aware_batch_size,
    _cost_aware_chunks,
    _cost_exponent,
    _fixed_chunks,
    _model_identity,
    _validate_batch_size,
    _validate_positive_int,
    default_prewarm_grid,
    predict_batch,
)
from _testutil import default_test_device  # noqa: E402


def _skip_if_weights_missing() -> None:
    if not MOS_ONLY.exists():
        pytest.skip(f"weights artifact unavailable: {MOS_ONLY}")


def _model() -> NisqaJaxModel:
    return load_model(MOS_ONLY, device=default_test_device())


# ---------------------------------------------------------------------------
# F1: length_bucket no-op fix — bucket-rounded padded shape survives JIT
# ---------------------------------------------------------------------------

def _patch_uniform_preprocess(monkeypatch, cfg, n_wins_val: int):
    """estimate + preprocess both return a fixed n_wins (uniform-length files)."""
    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return n_wins_val

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        x = np.full((n_wins_val, 1, cfg.n_mels, cfg.seg_length), 1.0, dtype=np.float32)
        return x, np.asarray(n_wins_val, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)


def test_bucket_padded_shape_survives_into_predict_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct exact lengths in the same bucket compile at the same rounded step.

    Four files with exact lengths 34,34,33,33 (bucket 32 -> all round to 64),
    batch_size=2 -> two chunks whose chunk-max is 34 and 33. Both must reach
    predict_segments with padded_steps=64 (one JIT cache entry), not 34/33.
    """
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    bucket = 32
    # Two files at 34, two at 33 — sorted desc -> [34,34,33,33] -> chunks [34,34],[33,33].
    lengths = [34, 34, 33, 33]
    paths = [Path(f"f{i}.wav") for i in range(4)]

    calls: list[tuple[int, int]] = []  # (x_shape1, padded_steps)

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return lengths[paths.index(Path(path))]

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        n = lengths[paths.index(Path(path))]
        x = np.full((n, 1, cfg.n_mels, cfg.seg_length), 1.0, dtype=np.float32)
        return x, np.asarray(n, dtype=np.int32)

    real_predict = model.predict_segments

    def recording_predict(x, n_wins, *, padded_steps=None, **kw):
        calls.append((int(x.shape[1]), int(padded_steps) if padded_steps is not None else -1))
        return real_predict(x, n_wins, padded_steps=padded_steps, **kw)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    monkeypatch.setattr(model, "predict_segments", recording_predict)

    predict_batch(model, paths, batch_size=2, length_bucket=bucket)
    assert len(calls) == 2, f"expected 2 chunk compiles, got {calls}"
    # Both chunks compile at the bucket-rounded step 64 (not the exact 34/33).
    assert {c[1] for c in calls} == {64}, f"padded_steps not bucket-rounded: {calls}"
    assert all(c[0] == 64 for c in calls), f"x.shape[1] not bucket-rounded: {calls}"


def test_bucket_padded_outputs_match_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bucket-rounded padding is masked by n_wins -> scores identical to exact."""
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    paths = [Path("a.wav"), Path("b.wav")]
    _patch_uniform_preprocess(monkeypatch, cfg, n_wins_val=33)

    bucketed = predict_batch(model, paths, batch_size=2, length_bucket=32)
    exact = predict_batch(model, paths, batch_size=2, length_bucket=1)
    numeric_cols = [c for c in bucketed.columns if c not in {"deg", "model"}]
    # Masking makes the padded region semantically inert, but different padded
    # widths compile to different XLA kernels whose float32 reduction order
    # differs by ~1e-7 (kernel tiling noise, not a correctness divergence).
    np.testing.assert_allclose(
        bucketed[numeric_cols].to_numpy(),
        exact[numeric_cols].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_predict_segments_padded_steps_crops_to_padded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct predict_segments with padded_steps keeps that exact time axis."""
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    rng = np.random.default_rng(0)
    n = 20
    x = rng.normal(size=(2, 64, 1, feat.n_mels, feat.seg_length)).astype(np.float32)
    n_wins = np.array([n, n], dtype=np.int32)
    # padded_steps=32 -> compile shape is 32, not 64 (excessive tail cropped to 32, not n_wins=20).
    out_padded = model.predict_segments(x, n_wins, padded_steps=32)
    # Reference: same 32-step crop run with the same padded_steps (identical
    # compile shape -> bit-identical, confirming padded_steps fixes the axis).
    out_ref = model.predict_segments(x[:, :32], n_wins, padded_steps=32)
    np.testing.assert_allclose(out_padded, out_ref, rtol=0, atol=0)
    # And padded_steps=None would crop to max(n_wins)=20 -> a different shape
    # whose result is masked-equal (within float32 kernel noise).
    out_crop20 = model.predict_segments(x[:, :32], n_wins)
    np.testing.assert_allclose(out_padded, out_crop20, rtol=1e-5, atol=1e-5)


def test_predict_segments_padded_steps_validates() -> None:
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    x = np.zeros((2, 32, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
    n_wins = np.array([10, 20], dtype=np.int32)
    with pytest.raises(ValueError, match="padded_steps=10 must be >= max.n_wins.=20"):
        model.predict_segments(x, n_wins, padded_steps=10)
    with pytest.raises(ValueError, match="padded_steps=64 must be <= x.shape.1.=32"):
        model.predict_segments(x, n_wins, padded_steps=64)
    with pytest.raises(ValueError, match="padded_steps must be an int"):
        model.predict_segments(x, n_wins, padded_steps=16.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# F2: CLI --prewarm compiles a documented bucket grid up to max_segments
# ---------------------------------------------------------------------------

def test_default_prewarm_grid_self_att() -> None:
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    grid = default_prewarm_grid(cfg)
    assert grid == [32, 64, 128, 256, 512, 1024, 1300]
    # Grid reaches max_segments (the longest real batch is a cache hit).
    assert grid[-1] == cfg.feature.max_segments
    # Not a single length (the prior bug prewarmed only [bucket]).
    assert len(grid) > 1


def test_default_prewarm_grid_tts() -> None:
    if not TTS.exists():
        pytest.skip("nisqa_tts artifact unavailable")
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(TTS)
    grid = default_prewarm_grid(cfg)
    assert grid == [64, 128, 256, 512, 1024, 2048, 4096, 6000]
    assert grid[-1] == 6000


def test_default_prewarm_grid_bucket_one_doubles() -> None:
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    grid = default_prewarm_grid(cfg, bucket=1)
    assert grid == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1300]


def test_cli_prewarm_compiles_grid_not_single_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--prewarm must call prewarm with the full grid, not one bucket length."""
    _skip_if_weights_missing()
    from nisqa_jax import predict as predict_mod
    captured: dict = {}

    def fake_prewarm(model, batch_sizes, bucket_lengths, cache_dir=None):
        captured["batch_sizes"] = list(batch_sizes)
        captured["bucket_lengths"] = list(bucket_lengths)

    monkeypatch.setattr(predict_mod, "prewarm", fake_prewarm)
    # Avoid actually loading the model's JIT by stubbing load_model to return a
    # lightweight object carrying the config the grid helper needs.
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)

    class _StubModel:
        def __init__(self, c):
            self.config = c

    monkeypatch.setattr(predict_mod, "load_model", lambda *a, **k: _StubModel(cfg))
    # predict_batch must not run (no real model); stub it to a no-op frame.
    monkeypatch.setattr(predict_mod, "predict_batch", lambda *a, **k: pd.DataFrame())

    # main prewarms (captured), then _collect_paths finds no wavs in the empty
    # dir and raises — the prewarm call has already happened, so assert captured.
    with pytest.raises(ValueError, match="No wav files found"):
        predict_mod.main([
            "--mode", "predict_dir",
            "--pretrained_model", str(MOS_ONLY),
            "--data_dir", str(tmp_path),
            "--bs", "4",
            "--prewarm",
        ])
    assert captured["batch_sizes"] == [4]
    # The grid (multiple lengths up to max_segments), not a single bucket length.
    assert captured["bucket_lengths"] == [32, 64, 128, 256, 512, 1024, 1300]
    assert len(captured["bucket_lengths"]) > 1


# ---------------------------------------------------------------------------
# F3: huge batch_size OOM/DoS — type validation + executable-shape clamp
# ---------------------------------------------------------------------------

def test_validate_batch_size_rejects_bool() -> None:
    with pytest.raises(ValueError, match="batch_size must be an int"):
        _validate_batch_size(True)
    with pytest.raises(ValueError, match="batch_size must be an int"):
        _validate_batch_size(False)


def test_validate_batch_size_rejects_float() -> None:
    with pytest.raises(ValueError, match="batch_size must be an int"):
        _validate_batch_size(2.0)


def test_validate_batch_size_rejects_negative() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        _validate_batch_size(-3)


def test_predict_batch_rejects_bool_batch_size() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="batch_size must be an int"):
        predict_batch(model, [Path("a.wav")], batch_size=True)  # type: ignore[arg-type]


def test_predict_batch_rejects_float_batch_size() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="batch_size must be an int"):
        predict_batch(model, [Path("a.wav")], batch_size=2.0)  # type: ignore[arg-type]


def test_validate_positive_int_rejects_bool() -> None:
    with pytest.raises(ValueError, match="must be an int"):
        _validate_positive_int(True, "length_bucket")
    with pytest.raises(ValueError, match="must be an int"):
        _validate_positive_int(False, "preprocess_workers")


def test_validate_positive_int_rejects_float() -> None:
    with pytest.raises(ValueError, match="must be an int"):
        _validate_positive_int(2.0, "length_bucket")


def test_validate_positive_int_rejects_zero() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        _validate_positive_int(0, "preprocess_workers")


def test_validate_positive_int_accepts_one() -> None:
    assert _validate_positive_int(1, "length_bucket") == 1
    assert _validate_positive_int(64, "preprocess_workers") == 64


def test_predict_batch_rejects_bool_length_bucket() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="length_bucket must be an int"):
        predict_batch(model, [Path("a.wav")], length_bucket=True)  # type: ignore[arg-type]


def test_predict_batch_rejects_float_length_bucket() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="length_bucket must be an int"):
        predict_batch(model, [Path("a.wav")], length_bucket=32.0)  # type: ignore[arg-type]


def test_predict_batch_rejects_zero_length_bucket() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="length_bucket must be >= 1"):
        predict_batch(model, [Path("a.wav")], length_bucket=0)


def test_predict_batch_rejects_bool_preprocess_workers() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="preprocess_workers must be an int"):
        predict_batch(model, [Path("a.wav")], preprocess_workers=True)  # type: ignore[arg-type]


def test_predict_batch_rejects_float_preprocess_workers() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="preprocess_workers must be an int"):
        predict_batch(model, [Path("a.wav")], preprocess_workers=2.0)  # type: ignore[arg-type]


def test_predict_batch_rejects_zero_preprocess_workers() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="preprocess_workers must be >= 1"):
        predict_batch(model, [Path("a.wav")], preprocess_workers=0)


def test_predict_batch_clamps_huge_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """batch_size >> len(paths) must not allocate a huge dummy batch.

    4 files + batch_size=10_000_000 -> the executable batch is clamped to 4, so
    predict_segments never sees a batch dim larger than 4 (no 10M-row array).
    """
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(4)]
    _patch_uniform_preprocess(monkeypatch, cfg, n_wins_val=4)

    seen_batch_dims: list[int] = []
    real_predict = model.predict_segments

    def recording(x, n_wins, *, padded_steps=None, **kw):
        seen_batch_dims.append(int(x.shape[0]))
        return real_predict(x, n_wins, padded_steps=padded_steps, **kw)

    monkeypatch.setattr(model, "predict_segments", recording)
    df = predict_batch(model, paths, batch_size=10_000_000)
    assert len(df) == 4
    # No chunk ever exceeded the real sample count (clamped to 4).
    assert max(seen_batch_dims) <= 4, f"unclamped batch dim allocated: {seen_batch_dims}"


def test_auto_batch_clamps_pathological_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto_batch with a huge batch_size cannot repeatedly allocate huge arrays."""
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(4)]
    _patch_uniform_preprocess(monkeypatch, cfg, n_wins_val=4)

    seen: list[int] = []
    real_predict = model.predict_segments

    def predict_ooms_above_two(x, n_wins, *, padded_steps=None, **kw):
        seen.append(int(x.shape[0]))
        if x.shape[0] > 2:
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory trying to allocate")
        return real_predict(x, n_wins, padded_steps=padded_steps, **kw)

    monkeypatch.setattr(model, "predict_segments", predict_ooms_above_two)
    df = predict_batch(model, paths, batch_size=10_000_000, auto_batch=True)
    assert len(df) == 4
    # Every retry batch dim is bounded by the clamped executable size (<=4).
    assert max(seen) <= 4, f"auto_batch allocated pathological batch: {seen}"


# ---------------------------------------------------------------------------
# F4: cost-aware batching mode
# ---------------------------------------------------------------------------

def test_cost_aware_batch_size_isolates_outlier() -> None:
    # ref_len=32, max_bs=32: a 1300-length file -> bs=1; a 32-length -> bs=32.
    assert _cost_aware_batch_size(1300, 32, 32) == 1
    assert _cost_aware_batch_size(32, 32, 32) == 32
    # 64-length (2x median) -> bs=16 (halved one power of two).
    assert _cost_aware_batch_size(64, 32, 32) == 16


def test_cost_exponent_model_aware() -> None:
    """self_att -> 2 (B*L^2), lstm -> 1 (B*L)."""
    assert _cost_exponent("self_att") == 2
    assert _cost_exponent("lstm") == 1


def test_cost_aware_self_att_isolates_more_aggressively_than_lstm() -> None:
    """A 4x-long outlier: self_att (exp=2) shrinks batch 16x, LSTM (exp=1) 4x.

    ref_len=32, max_bs=32, outlier length=128 (4x median):
      LSTM (exp=1):  cap = 32 * (32/128)^1 = 8  -> bs=8  (4x shrink)
      self_att(exp=2): cap = 32 * (32/128)^2 = 2  -> bs=2  (16x shrink)
    Both bounded to power-of-two sizes; self_att isolates quadratically more.
    """
    # LSTM (linear cost): 4x longer -> 4x smaller batch -> bs=8.
    assert _cost_aware_batch_size(128, 32, 32, exponent=1) == 8
    # self_att (quadratic cost): 4x longer -> 16x smaller batch -> bs=2.
    assert _cost_aware_batch_size(128, 32, 32, exponent=2) == 2
    # self_att batch size is strictly smaller than LSTM for the same outlier.
    assert _cost_aware_batch_size(128, 32, 32, exponent=2) < _cost_aware_batch_size(128, 32, 32, exponent=1)


def test_cost_aware_chunks_self_att_smaller_outlier_chunk_than_lstm() -> None:
    """self_att chunking isolates a long outlier into a smaller batch than LSTM."""
    order = list(range(64))
    n_wins = [128] + [32] * 63  # one 4x outlier + 63 median-length files
    lstm_chunks = _cost_aware_chunks(order, n_wins, 32, exponent=1)
    sa_chunks = _cost_aware_chunks(order, n_wins, 32, exponent=2)
    # The outlier is always chunk[0] (sorted descending); self_att batch is smaller.
    _, lstm_bs0 = lstm_chunks[0]
    _, sa_bs0 = sa_chunks[0]
    assert sa_bs0 < lstm_bs0, f"self_att bs={sa_bs0} not < LSTM bs={lstm_bs0}"
    assert sa_bs0 == 2   # 32 * (32/128)^2 = 2
    assert lstm_bs0 == 8  # 32 * (32/128)^1 = 8
    # Both keep bounded power-of-two sizes for the short majority.
    assert all(bs <= 32 for _, bs in sa_chunks)
    assert all(bs <= 32 for _, bs in lstm_chunks)
    # Both are deterministic (same input -> same output).
    assert _cost_aware_chunks(order, n_wins, 32, exponent=2) == sa_chunks


def test_cost_aware_chunks_isolate_long_outlier() -> None:
    """1x1300 + 999x32: the outlier forms its own bs=1 chunk; rest are bs=32."""
    order = list(range(1000))
    n_wins = [1300] + [32] * 999
    chunks = _cost_aware_chunks(order, n_wins, 32)
    idx0, bs0 = chunks[0]
    assert idx0 == [0]  # the 1300-length outlier alone
    assert bs0 == 1
    # All remaining chunks are full bs=32 except the final remainder.
    assert all(len(idx) == 32 for idx, _ in chunks[1:-1])
    assert len(chunks[-1][0]) <= 32


def test_cost_aware_cost_proxy_major_reduction() -> None:
    """Cost proxy (sum chunk_bs * chunk_max_length) must drop substantially."""
    order = list(range(1000))
    n_wins = [1300] + [32] * 999

    def cost_pairs(pairs):
        return sum(len(idx) * max(n_wins[i] for i in idx) for idx, _ in pairs)

    def cost_plain(chunks):
        return sum(len(c) * max(n_wins[i] for i in c) for c in chunks)

    fixed = _fixed_chunks(order, 32)
    aware = _cost_aware_chunks(order, n_wins, 32)
    c_fixed = cost_plain(fixed)
    c_aware = cost_pairs(aware)
    assert c_aware < 0.6 * c_fixed, f"cost-aware {c_aware} not << fixed {c_fixed}"
    # Sanity: the reduction is at least 2x for this heavy-tailed distribution.
    assert c_fixed / c_aware >= 2.0


def test_cost_aware_outputs_identical_to_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cost-aware regroups chunks but per-file scores are bit-identical to fixed."""
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    # Heavy-tailed: one long, several short. Distinct values per file so a
    # misassignment would change scores.
    lengths = [130, 4, 4, 4, 8, 8, 16, 16]
    paths = [Path(f"f{i}.wav") for i in range(len(lengths))]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return lengths[paths.index(Path(path))]

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        n = lengths[paths.index(Path(path))]
        # Distinct fill value per file so outputs differ and misorder is caught.
        v = float(paths.index(Path(path)) + 1)
        x = np.full((n, 1, cfg.n_mels, cfg.seg_length), v, dtype=np.float32)
        return x, np.asarray(n, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)

    fixed = predict_batch(model, paths, batch_size=4, batch_mode="fixed")
    aware = predict_batch(model, paths, batch_size=4, batch_mode="cost_aware")
    # Original input order preserved.
    assert fixed["deg"].tolist() == [str(p) for p in paths]
    assert aware["deg"].tolist() == [str(p) for p in paths]
    # Identical scores (only grouping changed); different batch shapes compile
    # to different XLA kernels so allow float32 tiling noise (~1e-7).
    numeric_cols = [c for c in fixed.columns if c not in {"deg", "model"}]
    np.testing.assert_allclose(
        aware[numeric_cols].to_numpy(),
        fixed[numeric_cols].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_cost_aware_rejects_no_sort() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="requires sort_by_length"):
        predict_batch(model, [Path("a.wav"), Path("b.wav")], batch_mode="cost_aware", sort_by_length=False)


def test_predict_batch_rejects_invalid_batch_mode() -> None:
    _skip_if_weights_missing()
    model = _model()
    with pytest.raises(ValueError, match="batch_mode must be"):
        predict_batch(model, [Path("a.wav")], batch_mode="weird")  # type: ignore[arg-type]


def test_cost_aware_walltime_not_slower_than_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wall-time sanity: cost-aware must not be slower than fixed on a heavy tail.

    Preprocessing is stubbed to synthetic segments so the wall-time reflects the
    scheduler's padded-compute effect (fixed pads 31 short files to the outlier's
    length; cost-aware isolates the outlier). On CPU the cost-aware path should
    be at least as fast; allow a generous 1.5x margin for scheduling noise.
    """
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    # 1 long outlier + 63 short files, bs=64.
    n_long, n_short = 256, 4
    lengths = [n_long] + [n_short] * 63
    paths = [Path(f"f{i}.wav") for i in range(len(lengths))]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        return lengths[paths.index(Path(path))]

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        n = lengths[paths.index(Path(path))]
        x = np.full((n, 1, cfg.n_mels, cfg.seg_length), 0.1, dtype=np.float32)
        return x, np.asarray(n, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)

    t0 = time.perf_counter()
    predict_batch(model, paths, batch_size=64, batch_mode="fixed")
    t_fixed = time.perf_counter() - t0

    t0 = time.perf_counter()
    predict_batch(model, paths, batch_size=64, batch_mode="cost_aware")
    t_aware = time.perf_counter() - t0

    assert t_aware <= t_fixed * 1.5, (
        f"cost-aware {t_aware:.3f}s slower than fixed {t_fixed:.3f}s"
    )


# ---------------------------------------------------------------------------
# F5: bench.py mixed/ragged lengths + scheduler label
# ---------------------------------------------------------------------------

def _write_wav(path: Path, sr: int, seconds: float) -> None:
    n = int(sr * seconds)
    samples = np.arange(n, dtype=np.float32) / sr
    sf.write(path, 0.05 * np.sin(2 * np.pi * 440 * samples), sr)


def test_bench_end_to_end_ragged_lengths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed lengths must not crash bench's naive scheduler (ragged stack bug)."""
    _skip_if_weights_missing()
    from nisqa_jax import bench as bench_mod
    model = _model()
    cfg = model.config
    sr = int(cfg.feature.sr or 48000)
    data = tmp_path / "wavs"
    data.mkdir()
    # Distinct durations -> distinct n_wins -> previously ragged np.stack crash.
    for i, secs in enumerate([0.5, 1.0, 1.5, 2.0]):
        _write_wav(data / f"f{i}.wav", sr, secs)
    import io
    import contextlib

    buf = io.StringIO()
    argv = [
        "--pretrained_model", str(MOS_ONLY),
        "--batch_size", "2",
        "--data_dir", str(data),
        "--preprocess_workers", "1",
    ]
    dev = default_test_device()
    if dev:
        argv += ["--device", dev]
    monkeypatch.setattr(sys, "argv", ["bench"] + argv)
    with contextlib.redirect_stdout(buf):
        bench_mod.main()
    result = json.loads(buf.getvalue())
    assert result["scheduler"] == "naive_in_order"
    assert result["file_count"] == 4


def test_bench_use_predict_batch_scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--use_predict_batch benchmarks the real predict_batch scheduler."""
    _skip_if_weights_missing()
    from nisqa_jax import bench as bench_mod
    model = _model()
    sr = int(model.config.feature.sr or 48000)
    data = tmp_path / "wavs"
    data.mkdir()
    for i, secs in enumerate([0.5, 1.0]):
        _write_wav(data / f"f{i}.wav", sr, secs)
    import io
    import contextlib

    buf = io.StringIO()
    argv = [
        "--pretrained_model", str(MOS_ONLY),
        "--batch_size", "2",
        "--data_dir", str(data),
        "--use_predict_batch",
        "--batch_mode", "cost_aware",
    ]
    dev = default_test_device()
    if dev:
        argv += ["--device", dev]
    monkeypatch.setattr(sys, "argv", ["bench"] + argv)
    with contextlib.redirect_stdout(buf):
        bench_mod.main()
    result = json.loads(buf.getvalue())
    assert result["scheduler"] == "predict_batch"
    assert result["batch_mode"] == "cost_aware"


# ---------------------------------------------------------------------------
# F6: on_error='collect' stable schema + path-bearing error strings
# ---------------------------------------------------------------------------

def test_collect_schema_always_includes_error_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """All files succeed -> `error` column still present (NaN), stable schema."""
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    paths = [Path("a.wav"), Path("b.wav")]
    _patch_uniform_preprocess(monkeypatch, cfg, n_wins_val=4)
    df = predict_batch(model, paths, batch_size=2, on_error="collect")
    assert "error" in df.columns
    # Stable order: deg, *_pred, model, error.
    assert list(df.columns)[-1] == "error"
    assert df["error"].isna().all()
    assert len(df) == 2


def test_collect_error_string_includes_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each error cell must explicitly include the failing file path."""
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    paths = [Path("good.wav"), Path("bad.wav")]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        if Path(path).name == "bad.wav":
            raise ValueError("corrupt audio header")
        x = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    df = predict_batch(model, paths, batch_size=2, on_error="collect")
    err = str(df.loc[df["deg"] == str(paths[1]), "error"].iloc[0])
    assert "bad.wav" in err
    assert "corrupt audio header" in err


def test_collect_model_failure_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A global model-forward failure is chunk-level and NOT per-file collectible."""
    _skip_if_weights_missing()
    model = _model()
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(4)]
    _patch_uniform_preprocess(monkeypatch, cfg, n_wins_val=4)

    def predict_always_ooms(x, n_wins, *, padded_steps=None, **kw):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")

    monkeypatch.setattr(model, "predict_segments", predict_always_ooms)
    # collect mode cannot isolate a model OOM; it must abort (not silently drop).
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        predict_batch(model, paths, batch_size=2, on_error="collect")


# ---------------------------------------------------------------------------
# F7: CSV model identity — source_name > display_name/model_label > stem
# ---------------------------------------------------------------------------

def test_model_identity_falls_back_to_source_stem() -> None:
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    stub = SimpleNamespace(source_name=None, source_path=cfg.source_path)
    assert _model_identity(stub) == cfg.source_path.stem


def test_model_identity_prefers_source_name() -> None:
    """source_name (checkpoint lane's run label) is the canonical identity."""
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    # Simulate the checkpoint lane's ModelConfig carrying a source_name field.
    stub = SimpleNamespace(
        source_name="NISQA_mos_only_run42",
        display_name="should_not_win",
        model_label="also_not",
        source_path=cfg.source_path,
    )
    assert _model_identity(stub) == "NISQA_mos_only_run42"


def test_model_identity_source_name_beats_display_name() -> None:
    """source_name is preferred over display_name/model_label when both present."""
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    stub = SimpleNamespace(
        source_name="canonical_run_label",
        display_name="alternate_label",
        source_path=cfg.source_path,
    )
    assert _model_identity(stub) == "canonical_run_label"


def test_model_identity_empty_source_name_falls_to_display_name() -> None:
    """An empty source_name falls through to display_name, then stem."""
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    stub = SimpleNamespace(
        source_name="",
        display_name="NISQA_MOS_v1.4",
        source_path=cfg.source_path,
    )
    assert _model_identity(stub) == "NISQA_MOS_v1.4"


def test_model_identity_prefers_display_name() -> None:
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    # Simulate a future lane-A ModelConfig that carries a display_name field.
    stub = SimpleNamespace(display_name="NISQA_MOS_v1.4", source_path=cfg.source_path)
    assert _model_identity(stub) == "NISQA_MOS_v1.4"
    # Empty display_name falls back.
    stub2 = SimpleNamespace(display_name="", source_path=cfg.source_path)
    assert _model_identity(stub2) == cfg.source_path.stem


# ---------------------------------------------------------------------------
# F8: channel type tightening + clear CSV-column / checkpoint input errors
# ---------------------------------------------------------------------------

def test_load_melspec_rejects_bool_channel(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    sr = int(feat.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    wav = tmp_path / "mono.wav"
    sf.write(wav, 0.05 * np.sin(2 * np.pi * 440 * samples), sr)
    with pytest.raises(ValueError, match="channel must be None or an int"):
        load_melspec(wav, feat, channel=True)  # type: ignore[arg-type]


def test_load_melspec_rejects_float_channel(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    sr = int(feat.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    wav = tmp_path / "mono.wav"
    sf.write(wav, 0.05 * np.sin(2 * np.pi * 440 * samples), sr)
    with pytest.raises(ValueError, match="channel must be None or an int"):
        load_melspec(wav, feat, channel=0.0)  # type: ignore[arg-type]


def test_segment_melspec_rejects_nonfinite_with_path() -> None:
    _skip_if_weights_missing()
    from nisqa_jax.checkpoint import load_converted_checkpoint
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    spec = np.zeros((cfg.feature.n_mels, cfg.feature.seg_length + 5), dtype=np.float32)
    spec[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite.*NaN/Inf.*corrupt.wav"):
        segment_melspec("corrupt.wav", spec, cfg.feature)


def test_cli_predict_csv_missing_column_clear_error(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    from nisqa_jax import predict as predict_mod
    csv = tmp_path / "files.csv"
    csv.write_text("filename,other\na.wav,1\n")
    with pytest.raises(ValueError, match="csv_deg column 'deg' not found"):
        predict_mod._collect_paths(SimpleNamespace(
            mode="predict_csv", data_dir=str(tmp_path), csv_file="files.csv", csv_deg="deg"))


def test_cli_predict_csv_missing_file_clear_error(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    from nisqa_jax import predict as predict_mod
    with pytest.raises(ValueError, match="CSV file not found"):
        predict_mod._collect_paths(SimpleNamespace(
            mode="predict_csv", data_dir=str(tmp_path), csv_file="nope.csv", csv_deg="deg"))


# ---------------------------------------------------------------------------
# F9: predict_segments rejects non-finite segment tensors
# ---------------------------------------------------------------------------

def test_predict_segments_rejects_nan_input() -> None:
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    x = np.zeros((2, 16, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
    x[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite.*NaN/Inf"):
        model.predict_segments(x, np.array([8, 16], dtype=np.int32))


def test_predict_segments_rejects_inf_input() -> None:
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    x = np.zeros((2, 16, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
    x[1, 5, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite.*NaN/Inf"):
        model.predict_segments(x, np.array([8, 16], dtype=np.int32))


def test_predict_segments_rejects_bool_n_wins() -> None:
    _skip_if_weights_missing()
    model = _model()
    feat = model.config.feature
    x = np.zeros((2, 16, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        model.predict_segments(x, np.array([True, False]))
