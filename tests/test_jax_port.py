from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "weights"))
PYTORCH_ROOT = ROOT / "nisqa pytorch"
SOURCE_WEIGHTS_ROOT = Path(os.environ.get("NISQA_SOURCE_WEIGHTS_DIR", PYTORCH_ROOT / "weights"))

sys.path.insert(0, str(ROOT))

from nisqa_jax.checkpoint import convert_checkpoint, load_converted_checkpoint, load_model  # noqa: E402
from nisqa_jax.features import preprocess_file, segment_melspec  # noqa: E402
from nisqa_jax.predict import predict_batch, predict_file  # noqa: E402
from _testutil import default_test_device  # noqa: E402


JAX_ARTIFACTS = [
    WEIGHTS_ROOT / "nisqa.npz",
    WEIGHTS_ROOT / "nisqa_mos_only.npz",
    WEIGHTS_ROOT / "nisqa_tts.npz",
]
SOURCE_CHECKPOINTS = [
    SOURCE_WEIGHTS_ROOT / "nisqa.tar",
    SOURCE_WEIGHTS_ROOT / "nisqa_mos_only.tar",
    SOURCE_WEIGHTS_ROOT / "nisqa_tts.tar",
]
SOURCE_TO_ARTIFACT = dict(zip(SOURCE_CHECKPOINTS, JAX_ARTIFACTS))


def _parity_tolerance(checkpoint: Path) -> tuple[float, float]:
    """Per-checkpoint (rtol, atol) for PyTorch parity assertions.

    All parity assertions compare the JAX port against the PyTorch **CPU**
    reference (``load_model(..., device="cpu")`` + ``torch_model`` on CPU).
    This is the mathematically correct check: a float64 ground-truth LSTM
    (built from the same weights) shows the JAX scan and PyTorch's non-fused
    CPU LSTM reference path both agree with f64 truth to ~1e-6, while
    PyTorch's **cuDNN** GPU LSTM is the outlier — its fused kernel accumulates
    the four gates in a different order, drifting to ~3.5e-5 at bs=32/sl=64
    and ~7e-3 at bs=8/sl=6000 (10-100x the CPU paths). The ~1.2e-3 drift
    previously observed was therefore cuDNN's accumulation, NOT a port bug
    (see ``adversarial_review/ISSUES_AND_ROADMAP.md`` ISSUE-08).

    Measured full-model CPU parity (JAX vs PT-CPU, 6 seeds x 8 shapes incl.
    bs=32/sl=64, bs=8/sl=128, bs=8/sl=6000) peaks at 8.9e-6. 5e-5 gives a
    ~5.6x safety margin and matches the self-attention bound. The earlier
    2e-3 widening (which accommodated the cuDNN-GPU outlier that this CPU
    test never exercised) is reverted.
    """
    return 5e-5, 5e-5


def _require_source_checkpoints() -> None:
    missing = [path for path in SOURCE_CHECKPOINTS if not path.exists()]
    if missing:
        pytest.skip(f"PyTorch source checkpoints are unavailable: {missing}")


def _require_torch_reference() -> tuple[Any, Any, Any]:
    _require_source_checkpoints()
    if not (PYTORCH_ROOT / "nisqa" / "NISQA_lib.py").exists():
        pytest.skip("PyTorch reference source repo is unavailable")
    sys.path.insert(0, str(PYTORCH_ROOT))
    try:
        torch = importlib.import_module("torch")
        nl = importlib.import_module("nisqa.NISQA_lib")
        segment_specs = nl.segment_specs
    except Exception as exc:  # pragma: no cover - depends on optional reference install.
        pytest.skip(f"PyTorch reference dependencies are unavailable: {exc}")
    return torch, nl, segment_specs


def _model_args(args: dict) -> dict:
    return {
        "ms_seg_length": args["ms_seg_length"],
        "ms_n_mels": args["ms_n_mels"],
        "cnn_model": args["cnn_model"],
        "cnn_c_out_1": args["cnn_c_out_1"],
        "cnn_c_out_2": args["cnn_c_out_2"],
        "cnn_c_out_3": args["cnn_c_out_3"],
        "cnn_kernel_size": args["cnn_kernel_size"],
        "cnn_dropout": args["cnn_dropout"],
        "cnn_pool_1": args["cnn_pool_1"],
        "cnn_pool_2": args["cnn_pool_2"],
        "cnn_pool_3": args["cnn_pool_3"],
        "cnn_fc_out_h": args["cnn_fc_out_h"],
        "td": args["td"],
        "td_sa_d_model": args["td_sa_d_model"],
        "td_sa_nhead": args["td_sa_nhead"],
        "td_sa_pos_enc": args["td_sa_pos_enc"],
        "td_sa_num_layers": args["td_sa_num_layers"],
        "td_sa_h": args["td_sa_h"],
        "td_sa_dropout": args["td_sa_dropout"],
        "td_lstm_h": args["td_lstm_h"],
        "td_lstm_num_layers": args["td_lstm_num_layers"],
        "td_lstm_dropout": args["td_lstm_dropout"],
        "td_lstm_bidirectional": args["td_lstm_bidirectional"],
        "td_2": args["td_2"],
        "td_2_sa_d_model": args["td_2_sa_d_model"],
        "td_2_sa_nhead": args["td_2_sa_nhead"],
        "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
        "td_2_sa_num_layers": args["td_2_sa_num_layers"],
        "td_2_sa_h": args["td_2_sa_h"],
        "td_2_sa_dropout": args["td_2_sa_dropout"],
        "td_2_lstm_h": args["td_2_lstm_h"],
        "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
        "td_2_lstm_dropout": args["td_2_lstm_dropout"],
        "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
        "pool": args["pool"],
        "pool_att_h": args["pool_att_h"],
        "pool_att_dropout": args["pool_att_dropout"],
    }


def _torch_model(checkpoint: Path):
    torch, nl, _ = _require_torch_reference()
    ck = torch.load(checkpoint, map_location="cpu")
    args = ck["args"]
    cls = {"NISQA": nl.NISQA, "NISQA_DIM": nl.NISQA_DIM}[args["model"]]
    model = cls(**_model_args(args))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return torch, model.eval(), args


def _synthetic_segments_from_model(model, *, batch_size: int = 2, steps: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(
        size=(batch_size, steps, 1, model.config.feature.n_mels, model.config.feature.seg_length),
    ).astype(np.float32)
    n_wins = np.asarray([max(1, steps - 5), steps], dtype=np.int32)
    x[0, n_wins[0] :] = 0.0
    return x, n_wins


@pytest.mark.parametrize("artifact", JAX_ARTIFACTS)
def test_standalone_jax_artifacts_load(artifact: Path) -> None:
    assert artifact.exists()
    cfg, params = load_converted_checkpoint(artifact)
    assert cfg.source_sha256
    assert params["cnn"]["conv1"]["w"].shape == (3, 3, 1, 16)
    assert all(f"bn{i}" not in params["cnn"] for i in range(1, 7))
    model = load_model(artifact, device=default_test_device())
    assert model.config.output_names == cfg.output_names


@pytest.mark.parametrize("artifact", JAX_ARTIFACTS)
def test_standalone_artifact_metadata_manifest(artifact: Path) -> None:
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    assert metadata["conversion_version"] == 4
    assert "model_config" in metadata
    assert all("/bn" not in name for name in metadata["shape_manifest"])
    assert all("/q/" not in name and "/k/" not in name and "/v/" not in name for name in metadata["shape_manifest"])


@pytest.mark.parametrize("checkpoint", SOURCE_CHECKPOINTS)
def test_checkpoint_conversion_cache_is_deterministic(checkpoint: Path, tmp_path: Path) -> None:
    _require_source_checkpoints()
    cache_a = tmp_path / "a"
    cache_b = tmp_path / "b"
    convert_checkpoint(checkpoint, cache_dir=cache_a)
    convert_checkpoint(checkpoint, cache_dir=cache_b)

    meta_a = json.loads(next(cache_a.glob("*.json")).read_text())
    meta_b = json.loads(next(cache_b.glob("*.json")).read_text())
    assert meta_a["shape_manifest"] == meta_b["shape_manifest"]
    assert meta_a["source_sha256"] == meta_b["source_sha256"]
    assert meta_a["conversion_version"] == 4

    npz_a = np.load(next(cache_a.glob("*.npz")))
    npz_b = np.load(next(cache_b.glob("*.npz")))
    assert sorted(npz_a.files) == sorted(npz_b.files)
    for name in npz_a.files:
        np.testing.assert_array_equal(npz_a[name], npz_b[name])


@pytest.mark.parametrize("checkpoint", SOURCE_CHECKPOINTS)
def test_standalone_artifact_matches_fresh_conversion(checkpoint: Path, tmp_path: Path) -> None:
    _require_source_checkpoints()
    fresh_cache = tmp_path / "fresh"
    cfg, params = convert_checkpoint(checkpoint, cache_dir=fresh_cache)
    artifact_cfg, artifact_params = load_converted_checkpoint(SOURCE_TO_ARTIFACT[checkpoint])
    assert artifact_cfg.source_sha256 == cfg.source_sha256
    assert artifact_cfg.output_names == cfg.output_names

    def flatten(tree, prefix=""):
        if isinstance(tree, dict):
            out = {}
            for key, value in tree.items():
                out.update(flatten(value, f"{prefix}{key}/"))
            return out
        if isinstance(tree, tuple):
            out = {}
            for idx, value in enumerate(tree):
                out.update(flatten(value, f"{prefix}{idx}/"))
            return out
        return {prefix[:-1]: np.asarray(tree)}

    fresh = flatten(params)
    artifact = flatten(artifact_params)
    assert sorted(artifact) == sorted(fresh)
    for name in fresh:
        np.testing.assert_array_equal(artifact[name], fresh[name])


def test_segment_melspec_matches_pytorch() -> None:
    _, _, torch_segment_specs = _require_torch_reference()
    cfg, _ = load_converted_checkpoint(WEIGHTS_ROOT / "nisqa_mos_only.npz")
    spec = np.arange(cfg.feature.n_mels * 32, dtype=np.float32).reshape(cfg.feature.n_mels, 32)
    jax_x, jax_n = segment_melspec("synthetic.wav", spec, cfg.feature)
    torch_x, torch_n = torch_segment_specs(
        "synthetic.wav",
        spec,
        cfg.feature.seg_length,
        cfg.feature.seg_hop_length,
        cfg.feature.max_segments,
    )
    # segment_melspec now returns the real [n_wins, ...] segments (no max_segments
    # padding); compare against the unpadded prefix of the PyTorch reference.
    n = int(jax_n)
    np.testing.assert_array_equal(jax_x, torch_x.numpy()[:n])
    assert int(jax_n) == int(torch_n)


def test_segment_melspec_hop_matches_pytorch() -> None:
    _, _, torch_segment_specs = _require_torch_reference()
    cfg, _ = load_converted_checkpoint(WEIGHTS_ROOT / "nisqa_mos_only.npz")
    feature = replace(cfg.feature, seg_length=5, seg_hop_length=2, max_segments=10)
    spec = np.arange(feature.n_mels * 10, dtype=np.float32).reshape(feature.n_mels, 10)
    jax_x, jax_n = segment_melspec("synthetic.wav", spec, feature)
    torch_x, torch_n = torch_segment_specs(
        "synthetic.wav",
        spec,
        feature.seg_length,
        feature.seg_hop_length,
        feature.max_segments,
    )
    n = int(jax_n)
    np.testing.assert_array_equal(jax_x, torch_x.numpy()[:n])
    assert int(jax_n) == int(torch_n) == 3


def test_segment_melspec_rejects_even_segment_length() -> None:
    cfg, _ = load_converted_checkpoint(WEIGHTS_ROOT / "nisqa_mos_only.npz")
    feature = replace(cfg.feature, seg_length=4)
    spec = np.zeros((feature.n_mels, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="seg_length must be odd"):
        segment_melspec("even.wav", spec, feature)


def test_too_short_sample_error_mentions_path() -> None:
    cfg, _ = load_converted_checkpoint(WEIGHTS_ROOT / "nisqa_mos_only.npz")
    spec = np.zeros((cfg.feature.n_mels, cfg.feature.seg_length - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="tiny.wav"):
        segment_melspec("tiny.wav", spec, cfg.feature)


@pytest.mark.parametrize("checkpoint", SOURCE_CHECKPOINTS)
def test_jax_forward_matches_pytorch_checkpoint(checkpoint: Path) -> None:
    torch, torch_model, _ = _torch_model(checkpoint)
    jax_model = load_model(SOURCE_TO_ARTIFACT[checkpoint], device=default_test_device())
    x, n_wins = _synthetic_segments_from_model(jax_model)

    with torch.no_grad():
        expected = torch_model(torch.from_numpy(x), torch.from_numpy(n_wins)).numpy()
    actual = jax_model.predict_segments(x, n_wins)
    rtol, atol = _parity_tolerance(checkpoint)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("checkpoint", SOURCE_CHECKPOINTS)
def test_jax_staged_outputs_match_pytorch_checkpoint(checkpoint: Path) -> None:
    torch, torch_model, _ = _torch_model(checkpoint)
    jax_model = load_model(SOURCE_TO_ARTIFACT[checkpoint], device=default_test_device())
    x, n_wins = _synthetic_segments_from_model(jax_model, steps=12)

    with torch.no_grad():
        x_torch = torch.from_numpy(x)
        n_torch = torch.from_numpy(n_wins)
        expected_cnn = torch_model.cnn(x_torch, n_torch)
        expected_td, expected_n_wins = torch_model.time_dependency(expected_cnn, n_torch)
        expected_td, _ = torch_model.time_dependency_2(expected_td, expected_n_wins)

    stages = jax_model.predict_stages(x, n_wins)
    rtol, atol = _parity_tolerance(checkpoint)
    np.testing.assert_allclose(stages["cnn"], expected_cnn.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(stages["time_dependency"], expected_td.numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("artifact", JAX_ARTIFACTS)
def test_bf16_outputs_are_finite_and_close_to_float32(artifact: Path) -> None:
    fp32_model = load_model(artifact, device=default_test_device(), precision="float32")
    bf16_model = load_model(artifact, device=default_test_device(), precision="bf16")
    x, n_wins = _synthetic_segments_from_model(fp32_model, steps=16)
    expected = fp32_model.predict_segments(x, n_wins)
    actual = bf16_model.predict_segments(x, n_wins)

    assert bf16_model.precision == "bf16"
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize("artifact", JAX_ARTIFACTS)
def test_masked_padding_and_cropping_are_invariant(artifact: Path) -> None:
    model = load_model(artifact, device=default_test_device())
    x, n_wins = _synthetic_segments_from_model(model, steps=18)

    base = model.predict_segments(x, n_wins)
    changed_tail = x.copy()
    changed_tail[0, n_wins[0] : n_wins.max()] = 999.0
    np.testing.assert_allclose(model.predict_segments(changed_tail, n_wins), base, rtol=0, atol=0)

    extended = np.concatenate([x, np.full_like(x[:, :3], -999.0)], axis=1)
    np.testing.assert_allclose(model.predict_segments(extended, n_wins), base, rtol=0, atol=0)


def test_predict_segments_rejects_zero_window_inputs() -> None:
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    x, _ = _synthetic_segments_from_model(model, steps=2)
    with pytest.raises(ValueError, match="n_wins must be >= 1"):
        model.predict_segments(x, np.zeros((x.shape[0],), dtype=np.int32))


def test_predict_batch_parallel_preprocessing_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path("first.wav"), Path("second.wav"), Path("third.wav")]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        value = float(paths.index(Path(path)) + 1)
        # Unpadded real segments: [n_wins, 1, n_mels, seg_length] (no max_segments pad).
        x = np.full((4, 1, cfg.n_mels, cfg.seg_length), value, dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    serial = predict_batch(model, paths, batch_size=3, preprocess_workers=1)
    parallel = predict_batch(model, paths, batch_size=3, preprocess_workers=2)
    assert parallel["deg"].tolist() == [str(path) for path in paths]
    # `model` is a string column (checkpoint stem); drop it alongside `deg`
    # before the numeric allclose comparison.
    numeric_cols = [c for c in parallel.columns if c not in {"deg", "model"}]
    np.testing.assert_allclose(
        parallel[numeric_cols].to_numpy(),
        serial[numeric_cols].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_predict_batch_rejects_invalid_preprocess_workers() -> None:
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    with pytest.raises(ValueError, match="preprocess_workers"):
        predict_batch(model, [Path("unused.wav")], preprocess_workers=0)


def test_predict_batch_partial_final_chunk_restores_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # 5 files with bs=2 -> final chunk has 1 real sample; a dummy row is added to
    # keep batch dim == batch_size, then discarded. Regression: the dummy row's
    # segment must be cropped to its (n_wins=1) entry or the broadcast fails and
    # the final real sample is lost.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(5)]
    n_wins_vals = [10, 3, 7, 1, 5]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return n_wins_vals[paths.index(Path(path))]

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        n = n_wins_vals[paths.index(Path(path))]
        x = np.full((n, 1, cfg.n_mels, cfg.seg_length), float(n), dtype=np.float32)
        return x, np.asarray(n, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    df = predict_batch(model, paths, batch_size=2)
    # All 5 real samples returned (dummy discarded), original order restored.
    assert len(df) == 5
    assert df["deg"].tolist() == [str(p) for p in paths]


@pytest.mark.parametrize(
    "artifact,expected_columns",
    [
        (
            WEIGHTS_ROOT / "nisqa.npz",
            ["deg", "mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred", "model"],
        ),
        (WEIGHTS_ROOT / "nisqa_mos_only.npz", ["deg", "mos_pred", "model"]),
        (WEIGHTS_ROOT / "nisqa_tts.npz", ["deg", "mos_pred", "model"]),
    ],
)
def test_predict_batch_csv_columns_match_pytorch(
    artifact: Path, expected_columns: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # PyTorch NISQA writes `deg, *_pred, model` (NISQA_model.py:76-79,
    # NISQA_lib.py:1461-1465). The TTS `naturalness` head is reported as
    # `mos_pred`. Dict API keys (`predict_file`) are covered separately.
    model = load_model(artifact, device=default_test_device())
    cfg = model.config.feature
    paths = [Path("a.wav"), Path("b.wav")]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        # Unpadded real segments: [n_wins, 1, n_mels, seg_length] (no max_segments pad).
        x = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    df = predict_batch(model, paths, batch_size=2)
    assert list(df.columns) == expected_columns
    assert df["deg"].tolist() == [str(path) for path in paths]
    assert (df["model"] == model.config.source_path.stem).all()


def test_predict_file_dict_api_keeps_output_names(tmp_path: Path) -> None:
    # Programmatic dict API must keep clean output_names keys (incl. the TTS
    # `naturalness` alias), independent of the CSV `*_pred` mapping.
    model = load_model(WEIGHTS_ROOT / "nisqa_tts.npz", device=default_test_device())
    cfg = model.config.feature
    wav = tmp_path / "synthetic.wav"
    sr = int(cfg.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    sf.write(wav, 0.05 * np.sin(2 * np.pi * 440 * samples), sr)
    out = predict_file(model, wav)
    assert list(out.keys()) == list(model.config.output_names)


@pytest.mark.parametrize("checkpoint", SOURCE_CHECKPOINTS)
def test_generated_wav_prediction_matches_pytorch(checkpoint: Path, tmp_path: Path) -> None:
    torch, torch_model, _ = _torch_model(checkpoint)
    jax_model = load_model(SOURCE_TO_ARTIFACT[checkpoint], device=default_test_device())
    sr = int(jax_model.config.feature.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    wav = tmp_path / f"{checkpoint.stem}.wav"
    sf.write(wav, 0.05 * np.sin(2 * np.pi * 440 * samples), sr)

    actual = predict_file(jax_model, wav)
    x, n_wins = preprocess_file(wav, jax_model.config.feature)
    with torch.no_grad():
        expected = torch_model(torch.from_numpy(x[None, :]), torch.from_numpy(n_wins.reshape(1))).numpy()[0]
    expected_dict = {name: float(expected[idx]) for idx, name in enumerate(jax_model.config.output_names)}
    assert actual.keys() == expected_dict.keys()
    rtol, atol = _parity_tolerance(checkpoint)
    np.testing.assert_allclose(list(actual.values()), list(expected_dict.values()), rtol=rtol, atol=atol)


def test_stereo_channel_prediction_path(tmp_path: Path) -> None:
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    sr = int(model.config.feature.sr or 48000)
    samples = np.arange(sr * 2, dtype=np.float32) / sr
    wav = tmp_path / "stereo.wav"
    stereo = np.stack(
        [
            0.03 * np.sin(2 * np.pi * 220 * samples),
            0.05 * np.sin(2 * np.pi * 440 * samples),
        ],
        axis=1,
    )
    sf.write(wav, stereo, sr)
    out0 = predict_file(model, wav, channel=0)
    out1 = predict_file(model, wav, channel=1)
    assert out0.keys() == out1.keys() == {"mos"}
    assert all(np.isfinite(value) for value in out0.values())
    assert all(np.isfinite(value) for value in out1.values())


# ---------------------------------------------------------------------------
# O1: per-file error isolation in predict_batch (on_error="collect"/"raise")
# ---------------------------------------------------------------------------

def _patch_preprocess_with_one_bad(monkeypatch: pytest.MonkeyPatch, cfg, paths, bad_idx):
    """estimate_n_wins succeeds for all; preprocess_file raises for `bad_idx`."""
    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        p = Path(path)
        if paths.index(p) == bad_idx:
            raise ValueError("corrupt audio header")
        value = float(paths.index(p) + 1)
        x = np.full((4, 1, cfg.n_mels, cfg.seg_length), value, dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)


def test_predict_batch_collect_isolates_bad_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1 corrupt file among 4 good ones: collect mode must return the 4 good rows
    # and report the bad one with an `error` message, never crashing the batch.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(5)]
    _patch_preprocess_with_one_bad(monkeypatch, cfg, paths, bad_idx=2)

    df = predict_batch(model, paths, batch_size=2, on_error="collect")
    assert len(df) == 5  # one row per input file, original order
    assert df["deg"].tolist() == [str(p) for p in paths]
    assert "error" in df.columns
    # Good rows: finite predictions, NaN error.
    good = df.drop(index=2)
    assert good["mos_pred"].notna().all()
    assert good["error"].isna().all()
    # Bad row: NaN prediction, non-empty error message naming the failure.
    assert np.isnan(df.loc[2, "mos_pred"])
    assert "corrupt audio header" in str(df.loc[2, "error"])


def test_predict_batch_collect_parallel_isolates_bad_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same isolation contract under the prefetch (multi-worker) pipeline.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(5)]
    _patch_preprocess_with_one_bad(monkeypatch, cfg, paths, bad_idx=0)

    df = predict_batch(model, paths, batch_size=2, preprocess_workers=2, on_error="collect")
    assert len(df) == 5
    assert df["deg"].tolist() == [str(p) for p in paths]
    assert "corrupt audio header" in str(df.loc[0, "error"])
    assert df.drop(index=0)["mos_pred"].notna().all()


def test_predict_batch_raise_names_the_bad_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # raise mode (default) aborts but wraps the exception with the failing path.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path("good_a.wav"), Path("bad.wav"), Path("good_b.wav")]
    _patch_preprocess_with_one_bad(monkeypatch, cfg, paths, bad_idx=1)

    with pytest.raises(RuntimeError, match="bad.wav"):
        predict_batch(model, paths, batch_size=2, on_error="raise")


def test_predict_batch_collect_isolates_estimate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # A too-short file fails at the header-estimate stage (before any GPU work);
    # collect mode records it and still returns the rest.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path("ok1.wav"), Path("tiny.wav"), Path("ok2.wav")]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        if Path(path).name == "tiny.wav":
            raise ValueError("Sample too short. File: tiny.wav")
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        x = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)

    df = predict_batch(model, paths, batch_size=2, on_error="collect")
    assert len(df) == 3
    assert "too short" in str(df.loc[1, "error"]).lower()
    assert df.drop(index=1)["mos_pred"].notna().all()


def test_predict_batch_rejects_invalid_on_error() -> None:
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    with pytest.raises(ValueError, match="on_error"):
        predict_batch(model, [Path("x.wav")], on_error="skip")


# ---------------------------------------------------------------------------
# O2(b): auto_batch OOM recovery
# ---------------------------------------------------------------------------

def test_predict_batch_auto_batch_recovers_from_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a GPU that OOMs above batch_size=2: predict_segments raises an
    # OOM-shaped error when the assembled batch is larger than 2, succeeds
    # otherwise. auto_batch must halve 8 -> 4 -> 2 and return all rows.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(8)]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        assert feature_cfg == cfg
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        assert feature_cfg == cfg
        x = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    real_predict = model.predict_segments

    def predict_that_ooms_above_two(x, n_wins):
        if x.shape[0] > 2:
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory trying to allocate")
        return real_predict(x, n_wins)

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    monkeypatch.setattr(model, "predict_segments", predict_that_ooms_above_two)

    df = predict_batch(model, paths, batch_size=8, auto_batch=True)
    assert len(df) == 8  # every sample recovered at the reduced batch size
    assert df["deg"].tolist() == [str(p) for p in paths]
    assert df["mos_pred"].notna().all()


def test_predict_batch_auto_batch_off_propagates_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without auto_batch, an OOM-shaped error propagates unchanged.
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path(f"f{i}.wav") for i in range(4)]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        x = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    def predict_that_ooms(x, n_wins):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    monkeypatch.setattr(model, "predict_segments", predict_that_ooms)

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        predict_batch(model, paths, batch_size=4, auto_batch=False)


def test_predict_batch_auto_batch_oom_at_bs1_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # If even a single sample OOMs, auto_batch cannot reduce further and must
    # re-raise (down to bs=1 is the floor).
    model = load_model(WEIGHTS_ROOT / "nisqa_mos_only.npz", device=default_test_device())
    cfg = model.config.feature
    paths = [Path("f0.wav"), Path("f1.wav")]

    def fake_estimate_n_wins(path: Path, feature_cfg) -> int:
        return 4

    def fake_preprocess_file(path: Path, feature_cfg, *, channel=None):
        x = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return x, np.asarray(4, dtype=np.int32)

    def predict_always_ooms(x, n_wins):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fake_estimate_n_wins)
    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess_file)
    monkeypatch.setattr(model, "predict_segments", predict_always_ooms)

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        predict_batch(model, paths, batch_size=2, auto_batch=True)
