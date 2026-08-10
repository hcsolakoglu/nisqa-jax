"""Adversarial tests for checkpoint/config/artifact correctness and security.

Covers the checkpoint-lane hardening:
  * Explicit rejection of unsupported PyTorch architectures before conversion
    (multi-layer LSTM, positional encoding, adapt+fc_out, bidirectional=False,
    invalid self-attention dims) — shipped checkpoints still convert.
  * Stable original model identity (``args['name']`` -> ``source_name``) on
    ModelConfig and JSON metadata, with backward-compatible fallback for old
    artifacts that lack the field.
  * Safe torch loading (``weights_only=True``): stock source tars convert; a
    non-safelist payload is refused (no silent fallback to unsafe pickle).
  * Hardened converted-metadata load: semantic validation rejects unknown/
    impossible output names and model configs, NaN/Inf parameter arrays, and
    top-level/model_config inconsistencies; canonical metadata checksum rejects
    any remaining field tampering.
  * No absolute build paths or secrets enter artifacts.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))
REF_WEIGHTS = Path(os.environ.get("NISQA_REF_WEIGHTS", "/tmp/nisqa_ref/weights"))
MOS_ONLY_NPZ = WEIGHTS_ROOT / "nisqa_mos_only.npz"
MOS_ONLY_JSON = WEIGHTS_ROOT / "nisqa_mos_only.json"
TTS_NPZ = WEIGHTS_ROOT / "nisqa_tts.npz"

sys.path.insert(0, str(ROOT))

from nisqa_jax.config import (  # noqa: E402
    SUPPORTED_ARCH_COMBOS,
    config_from_checkpoint_args,
    derive_output_names,
    validate_model_config,
)
from nisqa_jax.checkpoint import (  # noqa: E402
    canonical_metadata_checksum,
    convert_checkpoint,
    load_converted_checkpoint,
)
from nisqa_jax.checkpoint import _torch_version_lt  # noqa: E402
from nisqa_jax.config import ModelConfig  # noqa: E402


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/generate_golden_fixtures.py", "tests/test_jax_port.py"],
)
def test_reference_tools_use_the_safe_checkpoint_loader(relative_path: str) -> None:
    """Maintainer/reference paths must not reintroduce direct unsafe torch.load."""
    path = ROOT / relative_path
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    direct_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    assert not direct_loads, f"{relative_path} contains a direct torch.load call"
    assert "_load_torch_checkpoint" in source


def _skip_if_weights_missing() -> None:
    if not MOS_ONLY_NPZ.exists():
        pytest.skip(f"weights artifact unavailable: {MOS_ONLY_NPZ}")


def _skip_if_ref_missing() -> None:
    if not (REF_WEIGHTS / "nisqa_mos_only.tar").exists():
        pytest.skip(f"source reference weights unavailable: {REF_WEIGHTS}")


def _skip_if_no_torch() -> None:
    """Conversion tests require torch; skip cleanly when it is not installed.

    Even if the /tmp reference .tar files happen to exist on disk, conversion
    calls ``_torch()`` which raises RuntimeError without torch installed. Skip
    (do not error) so the suite is green in a torch-free environment; this is
    test behavior only and does not weaken any actual gate.
    """
    pytest.importorskip("torch")


def _self_att_args(nhead: int = 1) -> dict:
    """Args matching the shipped nisqa_mos_only checkpoint (adapt, self_att, att)."""
    cfg, _ = load_converted_checkpoint(MOS_ONLY_NPZ)
    f = cfg.feature
    return {
        "model": cfg.model_name,
        "name": cfg.source_name,
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
        "td_sa_pos_enc": False,
        "td_lstm_h": cfg.td_lstm_h,
        "td_lstm_bidirectional": cfg.td_lstm_bidirectional,
        "cnn_fc_out_h": None,
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


def _lstm_args() -> dict:
    """Args matching the shipped nisqa_tts checkpoint (standard CNN, BiLSTM)."""
    cfg, _ = load_converted_checkpoint(TTS_NPZ)
    f = cfg.feature
    return {
        "model": cfg.model_name,
        "name": cfg.source_name,
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
        "td_lstm_num_layers": 1,
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


def _copy_artifact(tmp_path: Path, name: str = "mos") -> tuple[Path, Path]:
    npz = tmp_path / f"{name}.npz"
    js = tmp_path / f"{name}.json"
    shutil.copy2(MOS_ONLY_NPZ, npz)
    shutil.copy2(MOS_ONLY_JSON, js)
    return npz, js


def _load_meta(js: Path) -> dict:
    return json.loads(js.read_text())


def _save_meta(js: Path, meta: dict) -> None:
    js.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _rebuild_npz(npz: Path, arrays: dict[str, np.ndarray]) -> None:
    npz.unlink()
    np.savez(npz, **arrays)


# ===========================================================================
# 1. Architecture rejection at the config boundary (source-args audit)
# ===========================================================================


class TestArchRejection:
    def test_multi_layer_lstm_rejected(self) -> None:
        """td_lstm_num_layers=2 must be rejected (JAX implements a single layer)."""
        _skip_if_weights_missing()
        if not TTS_NPZ.exists():
            pytest.skip("nisqa_tts artifact unavailable")
        args = _lstm_args()
        args["td_lstm_num_layers"] = 2
        with pytest.raises(NotImplementedError, match="td_lstm_num_layers must be 1"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_three_layer_lstm_rejected(self) -> None:
        _skip_if_weights_missing()
        if not TTS_NPZ.exists():
            pytest.skip("nisqa_tts artifact unavailable")
        args = _lstm_args()
        args["td_lstm_num_layers"] = 3
        with pytest.raises(NotImplementedError, match="td_lstm_num_layers must be 1"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_lstm_num_layers_none_rejected(self) -> None:
        """LSTM checkpoint missing td_lstm_num_layers must be rejected."""
        _skip_if_weights_missing()
        if not TTS_NPZ.exists():
            pytest.skip("nisqa_tts artifact unavailable")
        args = _lstm_args()
        args["td_lstm_num_layers"] = None
        with pytest.raises(NotImplementedError, match="td_lstm_num_layers must be 1"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_lstm_unidirectional_rejected(self) -> None:
        """td_lstm_bidirectional=False must be rejected (only BiLSTM implemented)."""
        _skip_if_weights_missing()
        if not TTS_NPZ.exists():
            pytest.skip("nisqa_tts artifact unavailable")
        args = _lstm_args()
        args["td_lstm_bidirectional"] = False
        with pytest.raises(NotImplementedError, match="td_lstm_bidirectional must be True"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_self_att_pos_enc_rejected(self) -> None:
        """Enabled td_sa_pos_enc must be rejected (positional encodings unimplemented)."""
        _skip_if_weights_missing()
        args = _self_att_args()
        args["td_sa_pos_enc"] = True
        with pytest.raises(NotImplementedError, match="td_sa_pos_enc is enabled"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_adapt_cnn_with_fc_out_h_rejected(self) -> None:
        """adapt CNN + cnn_fc_out_h set must be rejected (adapt has no fc_out head)."""
        _skip_if_weights_missing()
        args = _self_att_args()
        args["cnn_fc_out_h"] = 64
        with pytest.raises(NotImplementedError, match="cnn_fc_out_h=64"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_self_att_invalid_num_layers_rejected(self) -> None:
        _skip_if_weights_missing()
        args = _self_att_args()
        args["td_sa_num_layers"] = 0
        with pytest.raises(NotImplementedError, match="td_sa_num_layers must be a positive int"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_self_att_invalid_d_model_rejected(self) -> None:
        _skip_if_weights_missing()
        args = _self_att_args()
        args["td_sa_d_model"] = 0
        with pytest.raises(NotImplementedError, match="td_sa_d_model must be a positive int"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_self_att_lstm_layers_mixed_rejected(self) -> None:
        """self_att checkpoint with td_lstm_num_layers set must be rejected."""
        _skip_if_weights_missing()
        args = _self_att_args()
        args["td_lstm_num_layers"] = 1
        with pytest.raises(NotImplementedError, match="td_lstm_num_layers is set"):
            config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")

    def test_shipped_self_att_args_accepted(self) -> None:
        """Sanity: the shipped mos_only args still pass the audit."""
        _skip_if_weights_missing()
        cfg = config_from_checkpoint_args(_self_att_args(), Path("/fake/x.tar"), "deadbeef")
        assert cfg.td == "self_att" and cfg.output_names == ("mos",)

    def test_shipped_lstm_args_accepted(self) -> None:
        """Sanity: the shipped tts args still pass the audit."""
        _skip_if_weights_missing()
        if not TTS_NPZ.exists():
            pytest.skip("nisqa_tts artifact unavailable")
        cfg = config_from_checkpoint_args(_lstm_args(), Path("/fake/x.tar"), "deadbeef")
        assert cfg.td == "lstm" and cfg.output_names == ("naturalness",)


# ===========================================================================
# 2. Stable original model identity (source_name)
# ===========================================================================


class TestSourceName:
    def test_source_name_captured_from_args(self) -> None:
        _skip_if_weights_missing()
        args = _self_att_args()
        args["name"] = "my_run_label_v3"
        cfg = config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")
        assert cfg.source_name == "my_run_label_v3"

    def test_source_name_none_when_missing(self) -> None:
        _skip_if_weights_missing()
        args = _self_att_args()
        del args["name"]
        cfg = config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")
        assert cfg.source_name is None

    def test_source_name_coerces_non_str_to_none(self) -> None:
        _skip_if_weights_missing()
        args = _self_att_args()
        args["name"] = 12345
        cfg = config_from_checkpoint_args(args, Path("/fake/x.tar"), "deadbeef")
        assert cfg.source_name is None

    @pytest.mark.parametrize(
        "artifact",
        [
            WEIGHTS_ROOT / "nisqa.npz",
            WEIGHTS_ROOT / "nisqa_mos_only.npz",
            WEIGHTS_ROOT / "nisqa_tts.npz",
        ],
    )
    def test_shipped_metadata_has_source_name(self, artifact: Path) -> None:
        if not artifact.exists():
            pytest.skip(f"weights artifact unavailable: {artifact}")
        meta = json.loads(artifact.with_suffix(".json").read_text())
        assert "source_name" in meta, f"{artifact.name} JSON missing top-level source_name"
        assert meta["source_name"] is not None, f"{artifact.name} source_name is None"
        assert meta["model_config"].get("source_name") == meta["source_name"], (
            f"{artifact.name} model_config.source_name != top-level source_name"
        )

    def test_shipped_source_name_values(self) -> None:
        """The three shipped checkpoints carry their known training-run labels."""
        _skip_if_weights_missing()
        expected = {
            "nisqa_mos_only.json": "NISQAv2_mos_only",
            "nisqa.json": "NISQAv2",
            "nisqa_tts.json": "NISQA_TTS_v1",
        }
        for name, label in expected.items():
            js = WEIGHTS_ROOT / name
            if not js.exists():
                continue
            assert json.loads(js.read_text())["source_name"] == label


# ===========================================================================
# 3. Backward-compatible fallback for old artifacts
# ===========================================================================


class TestBackwardCompat:
    def test_old_metadata_without_source_name_loads_with_fallback(self, tmp_path: Path) -> None:
        """An old JSON lacking source_name + metadata_sha256 loads (warn, fallback)."""
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        # Strip the new fields to simulate a pre-source-name artifact.
        meta.pop("source_name", None)
        meta.pop("metadata_sha256", None)
        meta["model_config"].pop("source_name", None)
        _save_meta(js, meta)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg, _params = load_converted_checkpoint(npz)
        assert cfg.source_name is None, "old artifact should fall back to source_name=None"
        # metadata_sha256 absence warns (not fails).
        assert any("metadata_sha256" in str(w.message) for w in caught), "missing metadata_sha256 should warn"

    def test_old_metadata_with_only_source_name_loads(self, tmp_path: Path) -> None:
        """Old JSON with source_name but without metadata_sha256 still loads."""
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        meta.pop("metadata_sha256", None)
        _save_meta(js, meta)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg, _ = load_converted_checkpoint(npz)
        assert cfg.source_name == "NISQAv2_mos_only"
        assert any("metadata_sha256" in str(w.message) for w in caught)


# ===========================================================================
# 4. Safe torch loading (weights_only=True)
# ===========================================================================


class TestSafeTorchLoad:
    def test_shipped_source_tar_converts_with_weights_only(self, tmp_path: Path) -> None:
        """Stock NISQA .tar converts under weights_only=True (no unsafe fallback)."""
        _skip_if_no_torch()
        _skip_if_ref_missing()
        cfg, params = convert_checkpoint(REF_WEIGHTS / "nisqa_mos_only.tar", cache_dir=tmp_path)
        assert cfg.td == "self_att"
        assert cfg.source_name == "NISQAv2_mos_only"
        # Params are finite float32 arrays.
        flat = _flatten_params(params)
        assert flat, "conversion produced no params"
        for name, arr in flat.items():
            assert arr.dtype == np.float32, f"{name} dtype {arr.dtype}"
            assert np.isfinite(arr).all(), f"{name} has non-finite values"

    @pytest.mark.parametrize("tar", ["nisqa_mos_only.tar", "nisqa.tar", "nisqa_tts.tar"])
    def test_all_shipped_source_tars_convert(self, tmp_path: Path, tar: str) -> None:
        _skip_if_no_torch()
        _skip_if_ref_missing()
        if not (REF_WEIGHTS / tar).exists():
            pytest.skip(f"{tar} unavailable")
        cfg, _ = convert_checkpoint(REF_WEIGHTS / tar, cache_dir=tmp_path)
        assert cfg.source_name is not None
        assert (cfg.model_name, cfg.cnn_model, cfg.td, cfg.pool) in SUPPORTED_ARCH_COMBOS

    def test_unsafe_payload_refused_no_fallback(self, tmp_path: Path) -> None:
        """A checkpoint carrying a non-safelist object is refused, not unsafe-loaded."""
        pytest.importorskip("torch")
        import torch

        # A custom class is not on the weights_only safelist.
        class _Evil:
            def __reduce__(self):  # type: ignore[no-untyped-def]
                return (print, ("pwned",))

        bad = tmp_path / "evil.tar"
        torch.save(
            {
                "args": {"model": "NISQA", "cnn_model": "adapt", "td": "self_att", "pool": "att", "name": "x"},
                "model_state_dict": {},
                "evil": _Evil(),
            },
            bad,
        )
        with pytest.raises(RuntimeError, match="weights_only=True"):
            convert_checkpoint(bad, cache_dir=tmp_path)


class TestTorchVersionParse:
    """Numeric major/minor parsing for the <1.13 weights_only classification.

    The TypeError handler in ``_load_torch_checkpoint`` must classify a torch
    version as "too old for weights_only" iff it is strictly older than 1.13.
    Parsing is numeric (not lexicographic string compare) so suffixes
    (``+cu128``, ``.dev0``, ``+cpu``) and multi-digit minor numbers do not
    misclassify. A non-parseable version is treated as 0.0 (safely "old") so
    the caller never silently passes an unsupported torch.
    """

    @pytest.mark.parametrize(
        "version,expected_lt",
        [
            ("1.9.0", True),
            ("1.10.2", True),
            ("1.10.2+cpu", True),
            ("1.12.1", True),
            ("1.12.1.dev0", True),
            ("1.13.0", False),
            ("1.13.0+cpu", False),
            ("1.13.1", False),
            ("2.0.0", False),
            ("2.11.0+cu128", False),
            ("2.11.0.dev0", False),
        ],
    )
    def test_version_classification(self, version: str, expected_lt: bool) -> None:
        assert _torch_version_lt(version, 1, 13) is expected_lt

    def test_multidigit_minor_not_lexicographic(self) -> None:
        """Lexicographic compare would wrongly rank '1.9' > '1.13'; numeric is correct."""
        assert _torch_version_lt("1.9.0", 1, 13) is True
        assert _torch_version_lt("1.13.0", 1, 9) is False

    def test_unparseable_version_treated_as_old(self) -> None:
        """A garbage version string must classify as 'too old' (safe), not pass."""
        assert _torch_version_lt("garbage", 1, 13) is True
        assert _torch_version_lt("", 1, 13) is True


def _flatten_params(tree: object, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        out: dict[str, np.ndarray] = {}
        for k, v in tree.items():
            out.update(_flatten_params(v, f"{prefix}{k}/"))
        return out
    if isinstance(tree, tuple | list):
        out = {}
        for i, v in enumerate(tree):
            out.update(_flatten_params(v, f"{prefix}{i}/"))
        return out
    return {prefix[:-1]: np.asarray(tree)}


# ===========================================================================
# 5. Hardened converted-metadata load: semantic + structural + checksum
# ===========================================================================


class TestMetadataTamperRejection:
    def test_nan_parameter_rejected(self, tmp_path: Path) -> None:
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        with np.load(npz) as loaded:
            arrays = {n: loaded[n] for n in loaded.files}
        first = next(iter(sorted(arrays)))
        arrays[first] = arrays[first].copy()
        arrays[first][0] = np.nan
        _rebuild_npz(npz, arrays)
        # npz bytes changed -> npz_sha256 mismatch fires first; assert it is
        # caught as a corruption error (either non-finite or sha mismatch).
        with pytest.raises(ValueError, match="non-finite|SHA256 does not match"):
            load_converted_checkpoint(npz)

    def test_nan_rejected_with_consistent_hash(self, tmp_path: Path) -> None:
        """Inject NaN, then fix npz_sha256 so the finiteness check itself fires."""
        _skip_if_weights_missing()
        from nisqa_jax.checkpoint import _sha256

        npz, js = _copy_artifact(tmp_path)
        with np.load(npz) as loaded:
            arrays = {n: loaded[n] for n in loaded.files}
        first = next(iter(sorted(arrays)))
        arrays[first] = arrays[first].copy()
        arrays[first][0] = np.nan
        _rebuild_npz(npz, arrays)
        meta = _load_meta(js)
        meta["npz_sha256"] = _sha256(npz)
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="non-finite values"):
            load_converted_checkpoint(npz)

    def test_inf_parameter_rejected(self, tmp_path: Path) -> None:
        _skip_if_weights_missing()
        from nisqa_jax.checkpoint import _sha256

        npz, js = _copy_artifact(tmp_path)
        with np.load(npz) as loaded:
            arrays = {n: loaded[n] for n in loaded.files}
        first = next(iter(sorted(arrays)))
        arrays[first] = arrays[first].copy()
        arrays[first][0] = np.inf
        _rebuild_npz(npz, arrays)
        meta = _load_meta(js)
        meta["npz_sha256"] = _sha256(npz)
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="non-finite values"):
            load_converted_checkpoint(npz)

    def test_output_name_relabel_rejected(self, tmp_path: Path) -> None:
        """Relabeling mos->naturalness in model_config is rejected at load."""
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        # Tamper: claim the mos head is 'naturalness'.
        meta["model_config"]["output_names"] = ["naturalness"]
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="inconsistent|unsupported|does not match"):
            load_converted_checkpoint(npz)

    def test_unknown_output_name_rejected(self, tmp_path: Path) -> None:
        """An output name outside the known set is rejected at load."""
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        # Build a DIM-style config (5 heads) but with a bogus name, on a mos npz.
        meta["model_config"]["model_name"] = "NISQA_DIM"
        meta["model_config"]["output_names"] = ["mos", "noi", "dis", "col", "bogus"]
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="unknown output names|inconsistent|does not match"):
            load_converted_checkpoint(npz)

    def test_unsupported_combo_rejected_at_load(self, tmp_path: Path) -> None:
        """A tampered model_config with an unsupported combo is rejected at load."""
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        # standard CNN + self_att is not an implemented combo.
        meta["model_config"]["cnn_model"] = "standard"
        meta["model_config"]["td"] = "self_att"
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="unsupported architecture combo|does not match"):
            load_converted_checkpoint(npz)

    def test_top_level_model_name_mismatch_rejected(self, tmp_path: Path) -> None:
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        meta["model_name"] = "NISQA_DIM"  # disagrees with model_config.model_name
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="top-level model_name.*does not match|does not match"):
            load_converted_checkpoint(npz)

    def test_top_level_output_names_mismatch_rejected(self, tmp_path: Path) -> None:
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        meta["output_names"] = ["naturalness"]  # disagrees with model_config
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="top-level output_names.*not match"):
            load_converted_checkpoint(npz)

    def test_metadata_sha256_tamper_rejected(self, tmp_path: Path) -> None:
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        meta["metadata_sha256"] = "0" * 64
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="canonical metadata checksum does not match"):
            load_converted_checkpoint(npz)

    def test_source_sha256_tamper_rejected(self, tmp_path: Path) -> None:
        """source_sha256 is not otherwise checked on load; the canonical checksum guards it."""
        _skip_if_weights_missing()
        npz, js = _copy_artifact(tmp_path)
        meta = _load_meta(js)
        meta["source_sha256"] = "f" * 64
        meta["model_config"]["source_sha256"] = "f" * 64
        _save_meta(js, meta)
        with pytest.raises(ValueError, match="canonical metadata checksum does not match"):
            load_converted_checkpoint(npz)

    def test_shipped_artifacts_load_clean_no_checksum_warning(self) -> None:
        """Shipped artifacts embed metadata_sha256 -> no missing-checksum warning."""
        _skip_if_weights_missing()
        for npz in [WEIGHTS_ROOT / "nisqa.npz", MOS_ONLY_NPZ, TTS_NPZ]:
            if not npz.exists():
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg, _ = load_converted_checkpoint(npz)
            assert cfg.source_name is not None
            assert not any("metadata_sha256" in str(w.message) for w in caught), (
                f"{npz.name} should not warn about metadata_sha256"
            )


# ===========================================================================
# 6. Canonical metadata checksum capability (external verifiability)
# ===========================================================================


class TestCanonicalChecksum:
    def test_checksum_stable_across_formatting(self, tmp_path: Path) -> None:
        """Pretty-printing vs compact JSON yields the same canonical checksum."""
        _skip_if_weights_missing()
        meta = _load_meta(MOS_ONLY_JSON)
        compact = json.loads(json.dumps(meta))
        # Re-serialize with different whitespace; canonical checksum ignores it.
        a = canonical_metadata_checksum(meta)
        b = canonical_metadata_checksum(compact)
        assert a == b

    def test_checksum_excludes_self_and_npz_hash(self) -> None:
        """metadata_sha256 and npz_sha256 must not affect the canonical checksum."""
        _skip_if_weights_missing()
        meta = _load_meta(MOS_ONLY_JSON)
        base = canonical_metadata_checksum(meta)
        m2 = dict(meta)
        m2["metadata_sha256"] = "0" * 64
        m2["npz_sha256"] = "1" * 64
        assert canonical_metadata_checksum(m2) == base

    def test_checksum_detects_semantic_edit(self) -> None:
        _skip_if_weights_missing()
        meta = _load_meta(MOS_ONLY_JSON)
        base = canonical_metadata_checksum(meta)
        m2 = json.loads(json.dumps(meta))
        m2["source_name"] = "tampered"
        m2["model_config"]["source_name"] = "tampered"
        assert canonical_metadata_checksum(m2) != base

    def test_checksum_round_trips_on_load(self) -> None:
        """The embedded metadata_sha256 equals canonical_metadata_checksum(metadata)."""
        _skip_if_weights_missing()
        for js in [WEIGHTS_ROOT / "nisqa.json", MOS_ONLY_JSON, WEIGHTS_ROOT / "nisqa_tts.json"]:
            if not js.exists():
                continue
            meta = json.loads(js.read_text())
            assert meta["metadata_sha256"] == canonical_metadata_checksum(meta), (
                f"{js.name} embedded metadata_sha256 is not self-consistent"
            )


# ===========================================================================
# 7. No absolute build paths or secrets in artifacts
# ===========================================================================


class TestNoPathOrSecretLeakage:
    @pytest.mark.parametrize(
        "artifact",
        [
            WEIGHTS_ROOT / "nisqa.json",
            WEIGHTS_ROOT / "nisqa_mos_only.json",
            WEIGHTS_ROOT / "nisqa_tts.json",
        ],
    )
    def test_no_absolute_path_in_metadata(self, artifact: Path) -> None:
        if not artifact.exists():
            pytest.skip(f"weights artifact unavailable: {artifact}")
        raw = artifact.read_text()
        meta = json.loads(raw)
        # source_path (top-level + nested) must be a bare filename.
        for src in (meta.get("source_path"), meta.get("model_config", {}).get("source_path")):
            assert src is not None
            assert not str(src).startswith("/"), f"{artifact.name} source_path is absolute: {src!r}"
            assert "/" not in str(src), f"{artifact.name} source_path has a separator: {src!r}"
        # No build-machine path leaked anywhere in the JSON text.
        assert "/home/" not in raw and "/media/" not in raw and "/tmp/" not in raw, (
            f"{artifact.name} contains an absolute path"
        )

    @pytest.mark.parametrize(
        "artifact",
        [
            WEIGHTS_ROOT / "nisqa.json",
            WEIGHTS_ROOT / "nisqa_mos_only.json",
            WEIGHTS_ROOT / "nisqa_tts.json",
        ],
    )
    def test_no_secret_patterns_in_metadata(self, artifact: Path) -> None:
        if not artifact.exists():
            pytest.skip(f"weights artifact unavailable: {artifact}")
        raw = artifact.read_text().lower()
        for secret in ("api_key", "apikey", "secret", "password", "token", "aws", "sk-"):
            assert secret not in raw, f"{artifact.name} contains suspicious token {secret!r}"

    def test_converted_artifact_has_no_absolute_path(self, tmp_path: Path) -> None:
        """A freshly converted artifact stores only the bare source filename."""
        _skip_if_no_torch()
        _skip_if_ref_missing()
        convert_checkpoint(REF_WEIGHTS / "nisqa_mos_only.tar", cache_dir=tmp_path)
        jsons = list(tmp_path.glob("*.json"))
        assert jsons, "conversion produced no JSON sidecar"
        meta = json.loads(jsons[0].read_text())
        src = meta["source_path"]
        assert not str(src).startswith("/") and "/" not in str(src)
        assert str(tmp_path) not in jsons[0].read_text(), "cache dir path leaked into JSON"


# ===========================================================================
# 8. validate_model_config / derive_output_names unit coverage
# ===========================================================================


class TestConfigHelpers:
    def test_derive_output_names_dim(self) -> None:
        assert derive_output_names("NISQA_DIM", ("NISQA_DIM", "adapt", "self_att", "att")) == (
            "mos",
            "noi",
            "dis",
            "col",
            "loud",
        )

    def test_derive_output_names_tts(self) -> None:
        assert derive_output_names("NISQA", ("NISQA", "standard", "lstm", "last_step_bi")) == ("naturalness",)

    def test_derive_output_names_mos(self) -> None:
        assert derive_output_names("NISQA", ("NISQA", "adapt", "self_att", "att")) == ("mos",)

    def test_validate_model_config_rejects_bad_combo(self) -> None:
        _skip_if_weights_missing()
        cfg, _ = load_converted_checkpoint(MOS_ONLY_NPZ)
        bad = ModelConfig(
            source_path=cfg.source_path,
            source_sha256=cfg.source_sha256,
            model_name="NISQA",
            source_name=cfg.source_name,
            cnn_model="standard",
            td="self_att",
            td_2="skip",
            pool="att",
            output_names=("mos",),
            feature=cfg.feature,
            cnn_pool_1=None,
            cnn_pool_2=None,
            cnn_pool_3=None,
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_lstm_h=None,
            td_lstm_bidirectional=None,
        )
        with pytest.raises(ValueError, match="unsupported architecture combo"):
            validate_model_config(bad)

    def test_validate_model_config_rejects_relabel(self) -> None:
        _skip_if_weights_missing()
        cfg, _ = load_converted_checkpoint(MOS_ONLY_NPZ)
        bad = ModelConfig(
            source_path=cfg.source_path,
            source_sha256=cfg.source_sha256,
            model_name=cfg.model_name,
            source_name=cfg.source_name,
            cnn_model=cfg.cnn_model,
            td=cfg.td,
            td_2=cfg.td_2,
            pool=cfg.pool,
            output_names=("naturalness",),  # wrong for a mos/self_att checkpoint
            feature=cfg.feature,
            cnn_pool_1=cfg.cnn_pool_1,
            cnn_pool_2=cfg.cnn_pool_2,
            cnn_pool_3=cfg.cnn_pool_3,
            td_sa_d_model=cfg.td_sa_d_model,
            td_sa_nhead=cfg.td_sa_nhead,
            td_sa_num_layers=cfg.td_sa_num_layers,
            td_sa_h=cfg.td_sa_h,
            td_lstm_h=cfg.td_lstm_h,
            td_lstm_bidirectional=cfg.td_lstm_bidirectional,
        )
        with pytest.raises(ValueError, match="output_names.*inconsistent|unknown output names"):
            validate_model_config(bad)
