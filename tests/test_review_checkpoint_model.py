from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))
MOS = WEIGHTS / "nisqa_mos_only.npz"

sys.path.insert(0, str(ROOT))

from nisqa_jax.checkpoint import (  # noqa: E402
    _expected_artifact_manifest,
    _expected_source_state_keys,
    _sha256,
    _validate_flat_parameter_names,
    _validate_source_state_dict,
    canonical_metadata_checksum,
    load_converted_checkpoint,
)
from nisqa_jax.config import config_from_checkpoint_args  # noqa: E402


def _require_weights() -> None:
    if not MOS.exists():
        pytest.skip(f"weights artifact unavailable: {MOS}")


def _source_args() -> dict[str, object]:
    cfg, _ = load_converted_checkpoint(MOS)
    feature = cfg.feature
    return {
        "model": "NISQA",
        "name": "NISQAv2_mos_only",
        "double_ended": False,
        "cnn_model": "adapt",
        "cnn_c_out_1": 16,
        "cnn_c_out_2": 32,
        "cnn_c_out_3": 64,
        "cnn_kernel_size": (3, 3),
        "cnn_pool_1": (24, 7),
        "cnn_pool_2": (12, 5),
        "cnn_pool_3": (6, 3),
        "cnn_fc_out_h": None,
        "td": "self_att",
        "td_2": "skip",
        "td_sa_d_model": 64,
        "td_sa_nhead": 1,
        "td_sa_num_layers": 2,
        "td_sa_h": 64,
        "td_sa_pos_enc": False,
        "td_lstm_h": None,
        "td_lstm_num_layers": None,
        "td_lstm_bidirectional": None,
        "pool": "att",
        "pool_att_h": 128,
        "pool_output_size": 1,
        "ms_sr": feature.sr,
        "ms_n_fft": feature.n_fft,
        "ms_hop_length": feature.hop_length_seconds,
        "ms_win_length": feature.win_length_seconds,
        "ms_n_mels": feature.n_mels,
        "ms_fmax": feature.fmax,
        "ms_seg_length": feature.seg_length,
        "ms_seg_hop_length": feature.seg_hop_length,
        "ms_max_segments": feature.max_segments,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cnn_c_out_1", 32),
        ("cnn_kernel_size", (5, 5)),
        ("td_sa_d_model", 128),
        ("td_sa_nhead", None),
        ("td_sa_num_layers", 3),
        ("pool_att_h", 64),
        ("ms_n_fft", 2048),
        ("ms_seg_hop_length", 1),
    ],
)
def test_source_args_reject_non_shipped_dimensions(key: str, value: object) -> None:
    _require_weights()
    args = _source_args()
    args[key] = value
    with pytest.raises(NotImplementedError, match=key):
        config_from_checkpoint_args(args, Path("renamed.tar"), "d" * 64)


def test_source_state_accounting_allows_only_expected_bn_trackers() -> None:
    _require_weights()
    cfg, _ = load_converted_checkpoint(MOS)
    required, ignored = _expected_source_state_keys(cfg)
    assert "time_dependency.model.layers.0.self_attn.in_proj_weight" in required
    assert "time_dependency.model.layers.0.self_attn.in_proj_bias" in required
    assert "time_dependency.model.layers.0.self_attn.in_proj.weight" not in required
    state = dict.fromkeys(required | ignored)
    _validate_source_state_dict(state, cfg)

    state["unrelated.debug_tensor"] = None
    with pytest.raises(ValueError, match="unexpected keys.*unrelated.debug_tensor"):
        _validate_source_state_dict(state, cfg)


def test_source_state_accounting_rejects_missing_required_tensor() -> None:
    _require_weights()
    cfg, _ = load_converted_checkpoint(MOS)
    required, _ = _expected_source_state_keys(cfg)
    missing = sorted(required)[0]
    state = dict.fromkeys(required - {missing})
    with pytest.raises(ValueError, match="missing required keys"):
        _validate_source_state_dict(state, cfg)


def test_shipped_manifests_match_exact_kernel_contract() -> None:
    _require_weights()
    for artifact in sorted(WEIGHTS.glob("*.npz")):
        cfg, _ = load_converted_checkpoint(artifact)
        metadata = json.loads(artifact.with_suffix(".json").read_text())
        assert metadata["shape_manifest"] == _expected_artifact_manifest(cfg)


@pytest.mark.parametrize(
    "names",
    [
        ["time_dependency/layers/1/w", "time_dependency/layers/2/w"],
        ["time_dependency/layers/0/w", "time_dependency/layers/name/w"],
        ["pool", "pool/linear/w"],
        ["pool//linear/w"],
    ],
)
def test_flat_parameter_names_reject_ambiguous_tuple_trees(names: list[str]) -> None:
    with pytest.raises(ValueError, match="tuple|collide|invalid"):
        _validate_flat_parameter_names(names)


def _copy_artifact(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    npz = tmp_path / "model.npz"
    sidecar = tmp_path / "model.json"
    shutil.copy2(MOS, npz)
    shutil.copy2(MOS.with_suffix(".json"), sidecar)
    return npz, sidecar, json.loads(sidecar.read_text())


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    metadata["metadata_sha256"] = canonical_metadata_checksum(metadata)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def test_loader_rejects_self_consistent_but_unsupported_weight_shape(tmp_path: Path) -> None:
    _require_weights()
    npz, sidecar, metadata = _copy_artifact(tmp_path)
    with np.load(npz, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["cnn/conv1/b"] = arrays["cnn/conv1/b"][:-1]
    np.savez(npz, **arrays)
    metadata["shape_manifest"]["cnn/conv1/b"] = [15]  # type: ignore[index]
    metadata["npz_sha256"] = _sha256(npz)
    _write_metadata(sidecar, metadata)

    with pytest.raises(ValueError, match="exact shipped JAX parameter contract"):
        load_converted_checkpoint(npz)


@pytest.mark.parametrize("field", ["source_path", "source_sha256"])
def test_loader_rejects_inconsistent_duplicated_source_metadata(
    tmp_path: Path,
    field: str,
) -> None:
    _require_weights()
    npz, sidecar, metadata = _copy_artifact(tmp_path)
    metadata[field] = "other.tar" if field == "source_path" else "f" * 64
    _write_metadata(sidecar, metadata)

    with pytest.raises(ValueError, match=rf"top-level {field}.*does not match"):
        load_converted_checkpoint(npz)


def test_loader_accepts_legacy_v4_absolute_source_path(tmp_path: Path) -> None:
    _require_weights()
    npz, sidecar, metadata = _copy_artifact(tmp_path)
    legacy_path = "/conversion-host/upstream/weights/nisqa_mos_only.tar"
    metadata["source_path"] = legacy_path
    metadata["model_config"]["source_path"] = legacy_path  # type: ignore[index]
    metadata.pop("source_name", None)
    metadata["model_config"].pop("source_name", None)  # type: ignore[union-attr]
    metadata.pop("npz_sha256", None)
    metadata.pop("metadata_sha256", None)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, _ = load_converted_checkpoint(npz)
    assert cfg.source_path == Path(legacy_path)
    messages = [str(item.message) for item in caught]
    assert any("lacks 'npz_sha256'" in message for message in messages)
    assert any("lacks 'metadata_sha256'" in message for message in messages)


def test_conflicting_process_global_cache_directory_is_rejected(tmp_path: Path) -> None:
    script = """
from nisqa_jax.checkpoint import _configure_persistent_cache
_configure_persistent_cache(r"{first}")
try:
    _configure_persistent_cache(r"{second}")
except ValueError as exc:
    assert "process-global" in str(exc) and "refusing conflicting" in str(exc)
else:
    raise AssertionError("conflicting cache directory was accepted")
""".format(first=tmp_path / "first", second=tmp_path / "second")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
