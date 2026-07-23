from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))
MOS_ONLY = WEIGHTS_ROOT / "nisqa_mos_only.npz"

sys.path.insert(0, str(ROOT))

from nisqa_jax.config import config_from_checkpoint_args  # noqa: E402
from nisqa_jax.checkpoint import load_converted_checkpoint, load_model  # noqa: E402
from nisqa_jax.features import load_melspec  # noqa: E402
from nisqa_jax.predict import predict_batch  # noqa: E402
from _testutil import default_test_device  # noqa: E402


def _skip_if_weights_missing() -> None:
    if not MOS_ONLY.exists():
        pytest.skip(f"weights artifact unavailable: {MOS_ONLY}")


def _base_args(nhead: int = 1) -> dict:
    """Args matching the shipped nisqa_mos_only checkpoint (self_att, single-head)."""
    cfg, _ = load_converted_checkpoint(MOS_ONLY)
    f = cfg.feature
    return {
        "model": cfg.model_name,
        "cnn_model": cfg.cnn_model,
        "td": cfg.td,
        "td_2": cfg.td_2,
        "pool": cfg.pool,
        "cnn_pool_1": cfg.cnn_pool_1,
        "cnn_pool_2": cfg.cnn_pool_2,
        "cnn_pool_3": cfg.cnn_pool_3,
        "td_sa_d_model": cfg.td_sa_d_model,
        "td_sa_nhead": nhead,
        "td_sa_num_layers": cfg.td_sa_num_layers,
        "td_sa_h": cfg.td_sa_h,
        "td_lstm_h": cfg.td_lstm_h,
        "td_lstm_bidirectional": cfg.td_lstm_bidirectional,
        "ms_sr": f.sr,
        "ms_n_fft": f.n_fft,
        "ms_hop_length": f.hop_length_seconds,
        "ms_win_length": f.win_length_seconds,
        "ms_n_mels": f.n_mels,
        "ms_fmax": f.fmax,
        "ms_seg_length": f.seg_length,
        "ms_seg_hop_length": f.seg_hop_length,
        "ms_max_segments": f.max_segments,
    }


def _valid_x_n_wins(model, *, batch: int = 2, steps: int = 16) -> tuple[np.ndarray, np.ndarray]:
    feat = model.config.feature
    rng = np.random.default_rng(0)
    x = rng.normal(size=(batch, steps, 1, feat.n_mels, feat.seg_length)).astype(np.float32)
    n_wins = np.array([max(1, steps - 4), steps], dtype=np.int32)
    return x, n_wins


# ---------------------------------------------------------------------------
# C1: multi-head self-attention rejected at config boundary
# ---------------------------------------------------------------------------


def test_config_rejects_multi_head_attention() -> None:
    _skip_if_weights_missing()
    args = _base_args(nhead=4)
    with pytest.raises(NotImplementedError, match="multi-head self-attention"):
        config_from_checkpoint_args(args, Path("/fake/nisqa.tar"), "deadbeef")


def test_config_accepts_single_head_attention() -> None:
    _skip_if_weights_missing()
    args = _base_args(nhead=1)
    cfg = config_from_checkpoint_args(args, Path("/fake/nisqa.tar"), "deadbeef")
    assert cfg.td_sa_nhead == 1


def test_config_accepts_missing_nhead() -> None:
    """LSTM checkpoints have td_sa_nhead=None and must still load."""
    _skip_if_weights_missing()
    args = _base_args(nhead=1)
    args["td_sa_nhead"] = None
    cfg = config_from_checkpoint_args(args, Path("/fake/nisqa.tar"), "deadbeef")
    assert cfg.td_sa_nhead is None


# ---------------------------------------------------------------------------
# C1b: model identity (output_names) is derived from the architecture combo,
# NOT the checkpoint filename (F4 regression).
# ---------------------------------------------------------------------------


def _tts_args() -> dict:
    """Args matching the shipped nisqa_tts checkpoint (standard CNN, BiLSTM)."""
    cfg, _ = load_converted_checkpoint(WEIGHTS_ROOT / "nisqa_tts.npz")
    f = cfg.feature
    return {
        "model": cfg.model_name,
        "cnn_model": cfg.cnn_model,
        "td": cfg.td,
        "td_2": cfg.td_2,
        "pool": cfg.pool,
        "cnn_pool_1": cfg.cnn_pool_1,
        "cnn_pool_2": cfg.cnn_pool_2,
        "cnn_pool_3": cfg.cnn_pool_3,
        "td_sa_d_model": cfg.td_sa_d_model,
        "td_sa_nhead": cfg.td_sa_nhead,
        "td_sa_num_layers": cfg.td_sa_num_layers,
        "td_sa_h": cfg.td_sa_h,
        "td_sa_pos_enc": None,
        "td_lstm_h": cfg.td_lstm_h,
        "td_lstm_bidirectional": cfg.td_lstm_bidirectional,
        # td_lstm_num_layers is not stored on ModelConfig (always 1 for the
        # implemented BiLSTM path) but is required by the source-args audit.
        "td_lstm_num_layers": 1,
        # standard CNN uses an fc_out head (cnn_fc_out_h); not on ModelConfig.
        "cnn_fc_out_h": 20,
        "ms_sr": f.sr,
        "ms_n_fft": f.n_fft,
        "ms_hop_length": f.hop_length_seconds,
        "ms_win_length": f.win_length_seconds,
        "ms_n_mels": f.n_mels,
        "ms_fmax": f.fmax,
        "ms_seg_length": f.seg_length,
        "ms_seg_hop_length": f.seg_hop_length,
        "ms_max_segments": f.max_segments,
    }


def test_config_tts_identity_independent_of_filename() -> None:
    """A renamed nisqa_tts.tar must still load as naturalness."""
    _skip_if_weights_missing()
    if not (WEIGHTS_ROOT / "nisqa_tts.npz").exists():
        pytest.skip("nisqa_tts artifact unavailable")
    args = _tts_args()
    # Pass a deliberately misleading filename — the architecture combo is the discriminator.
    cfg = config_from_checkpoint_args(args, Path("/some/renamed_checkpoint.tar"), "deadbeef")
    assert cfg.output_names == ("naturalness",)


def test_config_mos_only_identity_not_promoted_to_naturalness() -> None:
    """A renamed nisqa_mos_only.tar must NOT become naturalness (it is self_att)."""
    _skip_if_weights_missing()
    args = _base_args(nhead=1)
    # Pass the tts filename to a mos_only (self_att) args dict — must stay `mos`.
    cfg = config_from_checkpoint_args(args, Path("/fake/nisqa_tts.tar"), "deadbeef")
    assert cfg.output_names == ("mos",)


def test_config_dim_identity_independent_of_filename() -> None:
    """NISQA_DIM (5-head) is identified by model_name, not filename."""
    _skip_if_weights_missing()
    if not (WEIGHTS_ROOT / "nisqa.npz").exists():
        pytest.skip("nisqa (DIM) artifact unavailable")
    cfg, _ = load_converted_checkpoint(WEIGHTS_ROOT / "nisqa.npz")
    args = _base_args(nhead=1)
    args["model"] = cfg.model_name  # NISQA_DIM
    args["cnn_model"] = cfg.cnn_model
    args["td"] = cfg.td
    args["pool"] = cfg.pool
    out = config_from_checkpoint_args(args, Path("/fake/whatever.tar"), "deadbeef")
    assert out.output_names == ("mos", "noi", "dis", "col", "loud")


# ---------------------------------------------------------------------------
# C2 / C3: device_segments / predict_segments input validation
# ---------------------------------------------------------------------------


def test_predict_segments_rejects_zero_n_wins() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, _ = _valid_x_n_wins(model)
    with pytest.raises(ValueError, match="n_wins must be >= 1"):
        model.predict_segments(x, np.array([0, 5], dtype=np.int32))


def test_predict_segments_rejects_negative_n_wins() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, _ = _valid_x_n_wins(model)
    with pytest.raises(ValueError, match="n_wins must be >= 1"):
        model.predict_segments(x, np.array([-3, 5], dtype=np.int32))


def test_predict_segments_rejects_n_wins_exceeding_steps() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, _ = _valid_x_n_wins(model)
    with pytest.raises(ValueError, match="n_wins must be <= x.shape"):
        model.predict_segments(x, np.array([x.shape[1] + 1, 5], dtype=np.int32))


def test_predict_segments_rejects_wrong_rank_x() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    bad = np.zeros((2, 16, feat.n_mels, feat.seg_length), dtype=np.float32)  # ndim==4
    with pytest.raises(ValueError, match="5-D ndarray"):
        model.predict_segments(bad, np.array([5, 6], dtype=np.int32))


def test_predict_segments_rejects_2d_n_wins() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, _ = _valid_x_n_wins(model)
    with pytest.raises(ValueError, match="n_wins must be a 1-D ndarray"):
        model.predict_segments(x, np.array([[5], [6]], dtype=np.int32))


def test_predict_segments_rejects_n_wins_length_mismatch() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, _ = _valid_x_n_wins(model)
    with pytest.raises(ValueError, match="len.n_wins"):
        model.predict_segments(x, np.array([5, 6, 7], dtype=np.int32))


def test_predict_segments_rejects_float_n_wins() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, _ = _valid_x_n_wins(model)
    with pytest.raises(ValueError, match="integer dtype"):
        model.predict_segments(x, np.array([5.0, 6.0], dtype=np.float32))


def test_predict_segments_rejects_wrong_trailing_shape() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    bad = np.zeros((2, 16, 2, feat.n_mels, feat.seg_length), dtype=np.float32)  # channel dim 2
    with pytest.raises(ValueError, match=r"x\.shape\[2:\]"):
        model.predict_segments(bad, np.array([5, 6], dtype=np.int32))


def test_predict_segments_rejects_empty_batch() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    x = np.zeros((0, 16, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
    with pytest.raises(ValueError, match="batch size must be greater than 0"):
        model.predict_segments(x, np.array([], dtype=np.int32))


def test_predict_segments_accepts_valid_input() -> None:
    """Sanity: well-formed input still runs and returns finite output."""
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    x, n_wins = _valid_x_n_wins(model)
    out = model.predict_segments(x, n_wins)
    assert out.shape == (2, 1)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# C4: predict_batch([]) rejected
# ---------------------------------------------------------------------------


def test_predict_batch_rejects_empty_input() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    with pytest.raises(ValueError, match="No wav files provided"):
        predict_batch(model, [], batch_size=1)


# ---------------------------------------------------------------------------
# C4b: predict_batch batch_size validation (early ValueError, not downstream
# RuntimeError from range(0, n, 0) ZeroDivisionError)
# ---------------------------------------------------------------------------


def test_predict_batch_rejects_batch_size_zero() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    with pytest.raises(ValueError, match="batch_size must be >= 1, got 0"):
        predict_batch(model, [Path("a.wav")], batch_size=0)


def test_predict_batch_rejects_negative_batch_size() -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    with pytest.raises(ValueError, match="batch_size must be >= 1, got -3"):
        predict_batch(model, [Path("a.wav")], batch_size=-3)


# ---------------------------------------------------------------------------
# C5: out-of-range channel produces precise error
# ---------------------------------------------------------------------------


def test_load_melspec_out_of_range_channel_message(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    sr = int(feat.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    stereo = np.stack(
        [0.03 * np.sin(2 * np.pi * 220 * samples), 0.05 * np.sin(2 * np.pi * 440 * samples)],
        axis=1,
    )
    wav = tmp_path / "stereo.wav"
    sf.write(wav, stereo, sr)
    with pytest.raises(ValueError, match="Channel 5 out of range for file with 2 channels"):
        load_melspec(wav, feat, channel=5)


def test_load_melspec_valid_channel_still_works(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    sr = int(feat.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    stereo = np.stack(
        [0.03 * np.sin(2 * np.pi * 220 * samples), 0.05 * np.sin(2 * np.pi * 440 * samples)],
        axis=1,
    )
    wav = tmp_path / "stereo.wav"
    sf.write(wav, stereo, sr)
    spec = load_melspec(wav, feat, channel=1)
    assert spec.ndim == 2
    assert spec.shape[0] == feat.n_mels


# ---------------------------------------------------------------------------
# C5b: mono channel validation (regression — guard previously inside ndim>1)
# ---------------------------------------------------------------------------


def _write_mono_wav(tmp_path: Path, feat) -> Path:
    sr = int(feat.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    wav = tmp_path / "mono.wav"
    sf.write(wav, 0.05 * np.sin(2 * np.pi * 440 * samples), sr)
    return wav


def test_load_melspec_mono_channel_zero_ok(tmp_path: Path) -> None:
    """mono + channel=0 must work (select the only channel / no-op)."""
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    wav = _write_mono_wav(tmp_path, feat)
    spec = load_melspec(wav, feat, channel=0)
    assert spec.ndim == 2
    assert spec.shape[0] == feat.n_mels


def test_load_melspec_mono_channel_one_rejected(tmp_path: Path) -> None:
    """mono + channel=1 must raise (only channel 0 exists)."""
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    wav = _write_mono_wav(tmp_path, feat)
    with pytest.raises(ValueError, match="Channel 1 out of range for file with 1 channels"):
        load_melspec(wav, feat, channel=1)


def test_load_melspec_mono_negative_channel_rejected(tmp_path: Path) -> None:
    """mono + negative channel must raise."""
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY, device=default_test_device())
    feat = model.config.feature
    wav = _write_mono_wav(tmp_path, feat)
    with pytest.raises(ValueError, match="Channel -1 out of range for file with 1 channels"):
        load_melspec(wav, feat, channel=-1)
