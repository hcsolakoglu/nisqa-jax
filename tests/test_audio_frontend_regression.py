"""Frozen WAV regression coverage for the audio frontend and shipped models.

The fixture contains exact WAV bytes and full frontend arrays recorded with
NumPy 1.26.4.  Keeping the reference arrays as data makes this test independent
of the implementation under test while allowing small, explicitly bounded
cross-NumPy floating-point differences.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _testutil import default_test_device
from nisqa_jax.checkpoint import load_model
from nisqa_jax.features import load_melspec, segment_melspec
from nisqa_jax.predict import predict_file
from nisqa_jax.weights import WEIGHTS_DIR

FIXTURE_DIR = Path(__file__).with_name("fixtures")
MANIFEST_PATH = FIXTURE_DIR / "audio_frontend_regression.json"
ARRAYS_PATH = FIXTURE_DIR / "audio_frontend_regression.npz.b64"
WAV_PATH = FIXTURE_DIR / "audio_frontend_regression.wav.gz.b64"

FRONTEND_CASES = (
    pytest.param("adapt", "nisqa_mos_only.npz", id="adapt"),
    pytest.param("tts", "nisqa_tts.npz", id="tts"),
)
SCORE_CASES = (
    pytest.param("nisqa", id="nisqa"),
    pytest.param("nisqa_mos_only", id="nisqa-mos-only"),
    pytest.param("nisqa_tts", id="nisqa-tts"),
)


@pytest.fixture(scope="module")
def frozen_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], Path, dict[str, np.ndarray]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["fixture_version"] == 1
    assert manifest["baseline"]["numpy"] == "1.26.4"
    assert float(manifest["frontend_atol"]) <= 2e-5
    assert float(manifest["score_atol"]) <= 2e-6

    wav_bytes = gzip.decompress(base64.b64decode(WAV_PATH.read_text(encoding="ascii")))
    assert hashlib.sha256(wav_bytes).hexdigest() == manifest["wav_sha256"]
    wav_path = tmp_path_factory.mktemp("audio-frontend") / "frozen.wav"
    wav_path.write_bytes(wav_bytes)

    archive_bytes = base64.b64decode(ARRAYS_PATH.read_text(encoding="ascii"))
    assert hashlib.sha256(archive_bytes).hexdigest() == manifest["baseline_npz_sha256"]
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    assert set(arrays) == set(manifest["profiles"])
    for name, expected in manifest["profiles"].items():
        assert list(arrays[name].shape) == expected["shape"]
        assert str(arrays[name].dtype) == expected["dtype"]
    return manifest, wav_path, arrays


@pytest.mark.parametrize(("profile", "artifact_name"), FRONTEND_CASES)
def test_frozen_wav_frontend_matches_numpy_126_baseline(
    frozen_baseline: tuple[dict[str, Any], Path, dict[str, np.ndarray]],
    profile: str,
    artifact_name: str,
) -> None:
    manifest, wav_path, expected = frozen_baseline
    model = load_model(WEIGHTS_DIR / artifact_name, device=default_test_device())

    spec = load_melspec(wav_path, model.config.feature)
    segments, n_wins = segment_melspec(wav_path, spec, model.config.feature)

    atol = float(manifest["frontend_atol"])
    np.testing.assert_allclose(spec, expected[f"{profile}_spec"], rtol=0, atol=atol)
    np.testing.assert_allclose(segments, expected[f"{profile}_segments"], rtol=0, atol=atol)
    np.testing.assert_array_equal(n_wins, expected[f"{profile}_n_wins"])


@pytest.mark.parametrize("model_name", SCORE_CASES)
def test_frozen_wav_scores_match_numpy_126_baseline(
    frozen_baseline: tuple[dict[str, Any], Path, dict[str, np.ndarray]],
    model_name: str,
) -> None:
    manifest, wav_path, expected = frozen_baseline
    model = load_model(WEIGHTS_DIR / f"{model_name}.npz", device=default_test_device())

    result = predict_file(model, wav_path)
    assert set(result) == set(model.config.output_names)
    actual_scores = np.asarray([result[name] for name in model.config.output_names])
    np.testing.assert_allclose(
        actual_scores,
        expected[f"{model_name}_scores"],
        rtol=0,
        atol=float(manifest["score_atol"]),
    )
