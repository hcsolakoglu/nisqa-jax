from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import librosa as lb
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from nisqa_jax.config import FeatureConfig
from nisqa_jax.features import load_melspec, segment_melspec
from nisqa_jax.predict import _format_cli_results, main, predict_batch


def _feature_config(**overrides) -> FeatureConfig:
    values = {
        "sr": 16_000,
        "n_fft": 512,
        "hop_length_seconds": 0.01,
        "win_length_seconds": 0.02,
        "n_mels": 24,
        "fmax": 8_000,
        "seg_length": 15,
        "seg_hop_length": 2,
        "max_segments": 1_000,
    }
    values.update(overrides)
    return FeatureConfig(**values)


def _upstream_get_librosa_melspec(
    file_path: Path,
    cfg: FeatureConfig,
    *,
    channel: int | None = None,
) -> np.ndarray:
    """Independent transcription of upstream NISQA_lib.get_librosa_melspec."""
    if channel is None:
        y, sr = lb.load(file_path, sr=cfg.sr)
    else:
        y, sr = lb.load(file_path, sr=cfg.sr, mono=False)
        if y.ndim > 1:
            y = y[channel, :]
    spec = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=cfg.n_fft,
        hop_length=int(sr * cfg.hop_length_seconds),
        win_length=int(sr * cfg.win_length_seconds),
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=cfg.n_mels,
        fmin=0.0,
        fmax=cfg.fmax,
        htk=False,
        norm="slaney",
    )
    return lb.core.amplitude_to_db(spec, ref=1.0, amin=1e-4, top_db=80.0)


def _upstream_segments_unpadded(spec: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Independent equivalent of upstream segment_specs before zero-padding."""
    n_wins_raw = spec.shape[1] - (cfg.seg_length - 1)
    starts = range(0, n_wins_raw, cfg.seg_hop_length)
    return np.stack([spec[:, start : start + cfg.seg_length] for start in starts])[:, None, :, :]


@pytest.mark.parametrize("kind", ["silence", "sweep", "clipped", "noise"])
def test_frontend_matches_independent_upstream_reference(
    kind: str,
    tmp_path: Path,
) -> None:
    cfg = _feature_config()
    sr = int(cfg.sr)
    t = np.arange(sr * 2, dtype=np.float32) / sr
    if kind == "silence":
        samples = np.zeros_like(t)
    elif kind == "sweep":
        samples = 0.1 * np.sin(2 * np.pi * (100 * t + 1_900 * t**2 / 2))
    elif kind == "clipped":
        samples = np.clip(4 * np.sin(2 * np.pi * 440 * t), -0.2, 0.2)
    else:
        samples = np.random.default_rng(1234).normal(0, 0.03, len(t)).astype(np.float32)

    wav = tmp_path / f"{kind}.wav"
    sf.write(wav, samples, sr, subtype="FLOAT")
    expected_spec = _upstream_get_librosa_melspec(wav, cfg)
    actual_spec = load_melspec(wav, cfg)
    np.testing.assert_allclose(actual_spec, expected_spec, rtol=0, atol=2e-6)

    expected_segments = _upstream_segments_unpadded(expected_spec, cfg)
    actual_segments, actual_n_wins = segment_melspec(wav, actual_spec, cfg)
    assert int(actual_n_wins) == len(expected_segments)
    np.testing.assert_allclose(actual_segments, expected_segments, rtol=0, atol=2e-6)


def test_stereo_frontend_matches_each_upstream_channel(tmp_path: Path) -> None:
    cfg = _feature_config()
    sr = int(cfg.sr)
    t = np.arange(sr * 2, dtype=np.float32) / sr
    stereo = np.stack(
        [
            0.03 * np.sin(2 * np.pi * 220 * t),
            0.08 * np.sin(2 * np.pi * 880 * t),
        ],
        axis=1,
    )
    wav = tmp_path / "stereo.wav"
    sf.write(wav, stereo, sr, subtype="FLOAT")

    actual = []
    for channel in (0, 1):
        expected = _upstream_get_librosa_melspec(wav, cfg, channel=channel)
        current = load_melspec(wav, cfg, channel=channel)
        np.testing.assert_allclose(current, expected, rtol=0, atol=2e-6)
        actual.append(current)
    assert not np.array_equal(actual[0], actual[1])


def test_segment_limit_is_checked_before_window_index_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _feature_config(seg_length=5, seg_hop_length=1, max_segments=2)
    spec = np.zeros((cfg.n_mels, 20), dtype=np.float32)

    def unexpected_arange(*args, **kwargs):
        raise AssertionError("window indices allocated before max_segments validation")

    monkeypatch.setattr("nisqa_jax.features.np.arange", unexpected_arange)
    with pytest.raises(ValueError, match=r"n_wins 16 > max_length 2"):
        segment_melspec("oversized.wav", spec, cfg)


class _FakeModel:
    def __init__(self, cfg: FeatureConfig) -> None:
        self.config = SimpleNamespace(
            feature=cfg,
            td="lstm",
            output_names=("mos",),
            source_name="test-model",
            source_path=Path("test-model.tar"),
        )

    def predict_segments(
        self,
        x: np.ndarray,
        n_wins: np.ndarray,
        *,
        padded_steps: int | None = None,
    ) -> np.ndarray:
        del x, padded_steps
        return np.asarray(n_wins, dtype=np.float32).reshape(-1, 1)


def test_single_chunk_uses_requested_preprocess_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _feature_config()
    model = _FakeModel(cfg)
    paths = [Path("first.wav"), Path("second.wav"), Path("third.wav")]
    main_thread = threading.get_ident()
    worker_threads: set[int] = set()
    first_two = threading.Barrier(2)
    counter_lock = threading.Lock()
    started = 0

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", lambda path, feature_cfg: 4)

    def fake_preprocess(path: Path, feature_cfg: FeatureConfig, *, channel=None):
        nonlocal started
        assert feature_cfg == cfg
        assert channel is None
        with counter_lock:
            ordinal = started
            started += 1
        worker_threads.add(threading.get_ident())
        if ordinal < 2:
            first_two.wait(timeout=5)
        segments = np.zeros((4, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        return segments, np.asarray(4, dtype=np.int32)

    monkeypatch.setattr("nisqa_jax.predict.preprocess_file", fake_preprocess)
    result = predict_batch(model, paths, batch_size=3, preprocess_workers=2, sort_by_length=False)

    assert result["deg"].tolist() == [str(path) for path in paths]
    assert main_thread not in worker_threads
    assert len(worker_threads) == 2


def test_parallel_collect_returns_when_every_header_estimate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _feature_config()
    model = _FakeModel(cfg)
    paths = [Path("first.wav"), Path("second.wav")]

    def fail_estimate(path: Path, feature_cfg: FeatureConfig) -> int:
        assert feature_cfg == cfg
        raise ValueError(f"bad header for {path.name}")

    monkeypatch.setattr("nisqa_jax.predict.estimate_n_wins", fail_estimate)
    result = predict_batch(model, paths, on_error="collect", preprocess_workers=2)

    assert result["deg"].tolist() == [str(path) for path in paths]
    assert result["mos_pred"].isna().all()
    assert result["error"].str.contains("bad header").all()


def test_cli_csv_preserves_input_table_and_appends_legacy_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "clip": ["nested/b.wav", "a.wav"],
            "speaker": ["speaker-b", "speaker-a"],
            "reference_mos": [2.5, 4.0],
        }
    )
    source.to_csv(tmp_path / "input.csv", index=False)
    output_dir = tmp_path / "out"
    captured_paths: list[Path] = []

    monkeypatch.setattr("nisqa_jax.predict.load_model", lambda *args, **kwargs: object())

    def fake_predict_batch(model, paths, **kwargs):
        del model, kwargs
        captured_paths.extend(paths)
        return pd.DataFrame(
            {
                "deg": [str(path) for path in paths],
                "mos_pred": [3.1, 4.2],
                "model": ["NISQAv2_mos_only", "NISQAv2_mos_only"],
            }
        )

    monkeypatch.setattr("nisqa_jax.predict.predict_batch", fake_predict_batch)
    main(
        [
            "--mode",
            "predict_csv",
            "--pretrained_model",
            "model.npz",
            "--data_dir",
            str(tmp_path),
            "--csv_file",
            "input.csv",
            "--csv_deg",
            "clip",
            "--output_dir",
            str(output_dir),
        ]
    )

    assert captured_paths == [tmp_path / "nested/b.wav", tmp_path / "a.wav"]
    saved = pd.read_csv(output_dir / "NISQA_results.csv")
    assert list(saved.columns) == ["clip", "speaker", "reference_mos", "mos_pred", "model"]
    pd.testing.assert_series_equal(saved["clip"], source["clip"])
    pd.testing.assert_series_equal(saved["speaker"], source["speaker"])
    np.testing.assert_allclose(saved["reference_mos"], source["reference_mos"])
    np.testing.assert_allclose(saved["mos_pred"], [3.1, 4.2])


def test_file_and_directory_cli_results_use_basename() -> None:
    paths = [Path("/audio/nested/b.wav"), Path("/audio/a.wav")]
    predictions = pd.DataFrame(
        {
            "deg": [str(path) for path in paths],
            "mos_pred": [2.0, 3.0],
            "model": ["model", "model"],
        }
    )

    for mode in ("predict_file", "predict_dir"):
        formatted = _format_cli_results(mode, paths, predictions)
        assert formatted["deg"].tolist() == ["b.wav", "a.wav"]
