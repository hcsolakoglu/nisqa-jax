from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nisqa_jax import bench
from nisqa_jax import bench_compare
from nisqa_jax.bench_compare import (
    _comparison_unavailable_reason,
    _resolve_torch_checkpoint,
    _verify_torch_checkpoint,
)


def test_torch_reference_is_explicit_for_converted_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.npz"
    reference = tmp_path / "source.tar"

    assert _resolve_torch_checkpoint(artifact, None) is None
    assert _resolve_torch_checkpoint(reference, None) == reference
    assert _resolve_torch_checkpoint(artifact, str(reference)) == reference.resolve()


def test_torch_reference_must_match_source_hash(tmp_path: Path) -> None:
    reference = tmp_path / "source.tar"
    reference.write_bytes(b"trusted checkpoint bytes")
    expected = hashlib.sha256(reference.read_bytes()).hexdigest()

    assert _verify_torch_checkpoint(reference, expected) == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_torch_checkpoint(reference, "0" * 64)


def test_reduced_precision_reference_comparison_is_rejected() -> None:
    reason = _comparison_unavailable_reason(
        requested=True,
        disabled=False,
        precision="bf16",
        torch_installed=True,
        torch_cuda_available=True,
        jax_platform="gpu",
    )

    assert reason is not None
    assert "only --precision float32" in reason


def test_one_chunk_uses_requested_preprocess_workers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    submitted: list[Path] = []
    worker_counts: list[int] = []

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class RecordingExecutor:
        def __init__(self, *, max_workers: int):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, model, path, *, channel):
            submitted.append(path)
            return ImmediateFuture(fn(model, path, channel=channel))

    class FakeModel:
        config = SimpleNamespace(feature=SimpleNamespace(n_mels=2, seg_length=3))
        device = SimpleNamespace(platform="cpu", __str__=lambda self: "cpu")
        precision = "float32"
        _compute_params = {}

        def device_segments(self, x, n_wins):
            return jnp.asarray(x), jnp.asarray(n_wins)

        @staticmethod
        def _forward(_params, x, _n_wins):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32)

    def fake_preprocess(_model, _path, *, channel):
        assert channel is None
        segments = np.zeros((2, 1, 2, 3), dtype=np.float32)
        return (segments, np.int32(2)), 0.001

    monkeypatch.setattr(bench, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(bench, "load_model", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(bench, "_preprocess_one", fake_preprocess)

    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    args = argparse.Namespace(
        pretrained_model="model.npz",
        device="cpu",
        cache_dir=None,
        precision="float32",
        preprocess_workers=2,
        batch_size=8,
        channel=None,
        use_predict_batch=False,
    )
    bench._run_end_to_end(args, paths)
    result = json.loads(capsys.readouterr().out)

    assert worker_counts == [2]
    assert submitted == paths
    assert result["first_shape_call_count"] == 1
    assert result["warmed_model_call_count"] == 0
    assert "input_transfer_seconds" in result
    assert "output_transfer_seconds" in result


def test_synthetic_benchmark_allocates_only_measured_sequence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen_shapes: list[tuple[int, ...]] = []

    class FakeModel:
        config = SimpleNamespace(feature=SimpleNamespace(max_segments=6000, n_mels=2, seg_length=3))
        device = "cpu"
        precision = "float32"
        _compute_params = {}

        def device_segments(self, x, n_wins):
            seen_shapes.append(x.shape)
            return jnp.asarray(x), jnp.asarray(n_wins)

        @staticmethod
        def _forward(_params, x, _n_wins):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32)

    monkeypatch.setattr(bench, "load_model", lambda *_args, **_kwargs: FakeModel())
    args = argparse.Namespace(
        pretrained_model="tts.npz",
        device="cpu",
        cache_dir=None,
        precision="float32",
        batch_size=2,
        steps=1,
        min_samples_per_second=None,
    )

    bench._run_synthetic(args)
    result = json.loads(capsys.readouterr().out)

    assert seen_shapes == [(2, 64, 1, 2, 3)]
    assert result["seq_len"] == 64


def test_jax_only_comparison_does_not_import_torch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeModel:
        config = SimpleNamespace(
            feature=SimpleNamespace(n_mels=2, seg_length=3),
            source_sha256="0" * 64,
        )
        device = SimpleNamespace(platform="cpu")
        precision = "float32"
        _compute_params = {}

        def device_segments(self, x, n_wins):
            return jnp.asarray(x), jnp.asarray(n_wins)

        @staticmethod
        def _forward(_params, x, _n_wins):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32)

    monkeypatch.setattr(bench_compare, "load_model", lambda *_args, **_kwargs: FakeModel())

    def fail_if_torch_imported():
        raise AssertionError("JAX-only benchmark imported optional PyTorch")

    monkeypatch.setattr(bench_compare, "_torch", fail_if_torch_imported)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_compare",
            "--pretrained_model",
            "model.npz",
            "--device",
            "cpu",
            "--batch_size",
            "1",
            "--seq_len",
            "2",
            "--steps",
            "1",
            "--no_torch",
        ],
    )

    bench_compare.main()
    result = json.loads(capsys.readouterr().out)

    assert result["torch_comparison_requested"] is False
    assert result["torch_comparison_enabled"] is False
    assert result["torch_cuda_available"] is None
