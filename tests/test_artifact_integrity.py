"""H1: artifact loader manifest validation tests.

Verifies ``load_converted_checkpoint`` rejects corrupted/tampered artifacts
(wrong shape, missing key, wrong dtype, npz hash mismatch) while still loading
the shipped artifacts. Since F5 the shipped artifacts embed ``npz_sha256``
(and store only the bare source filename, no build-machine paths), so they load
cleanly with no missing-hash warning.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "weights"))
MOS_ONLY_NPZ = WEIGHTS_ROOT / "nisqa_mos_only.npz"
MOS_ONLY_JSON = WEIGHTS_ROOT / "nisqa_mos_only.json"

sys.path.insert(0, str(ROOT))

from nisqa_jax.checkpoint import load_converted_checkpoint  # noqa: E402


def _skip_if_weights_missing() -> None:
    if not MOS_ONLY_NPZ.exists():
        pytest.skip(f"weights artifact unavailable: {MOS_ONLY_NPZ}")


def _copy_artifact(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the shipped mos_only artifact into tmp_path for mutation."""
    npz = tmp_path / "mos.npz"
    js = tmp_path / "mos.json"
    shutil.copy2(MOS_ONLY_NPZ, npz)
    shutil.copy2(MOS_ONLY_JSON, js)
    return npz, js


def _load_metadata(js: Path) -> dict:
    return json.loads(js.read_text())


def _save_metadata(js: Path, meta: dict) -> None:
    js.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _rebuild_npz(npz: Path, arrays: dict[str, np.ndarray]) -> None:
    npz.unlink()
    np.savez(npz, **arrays)


# ---------------------------------------------------------------------------
# Shipped artifacts: load OK with embedded npz_sha256 (no missing-hash warning)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artifact", [
    WEIGHTS_ROOT / "nisqa.npz",
    WEIGHTS_ROOT / "nisqa_mos_only.npz",
    WEIGHTS_ROOT / "nisqa_tts.npz",
])
def test_shipped_artifacts_load_clean_with_npz_sha256(artifact: Path) -> None:
    if not artifact.exists():
        pytest.skip(f"weights artifact unavailable: {artifact}")
    meta = json.loads(artifact.with_suffix(".json").read_text())
    assert "npz_sha256" in meta and meta["npz_sha256"], (
        f"shipped artifact {artifact.name} must embed npz_sha256 after F5"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, params = load_converted_checkpoint(artifact)
    assert cfg.output_names  # loaded successfully
    # No missing-npz_sha256 warning should be emitted (the hash is now embedded).
    assert not any("npz_sha256" in str(w.message) for w in caught), (
        "shipped artifact should not warn about npz_sha256 after F5"
    )


# ---------------------------------------------------------------------------
# F5: no metadata field contains an absolute/build-machine path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artifact", [
    WEIGHTS_ROOT / "nisqa.npz",
    WEIGHTS_ROOT / "nisqa_mos_only.npz",
    WEIGHTS_ROOT / "nisqa_tts.npz",
])
def test_shipped_metadata_has_no_absolute_paths(artifact: Path) -> None:
    if not artifact.exists():
        pytest.skip(f"weights artifact unavailable: {artifact}")
    js = artifact.with_suffix(".json")
    raw = js.read_text()
    meta = json.loads(raw)
    # No build-machine path leaked anywhere in the JSON text.
    assert "/media/" not in raw, f"{js.name} contains a /media/ path"
    # source_path (top-level and nested in model_config) must be a bare filename.
    for src in (meta.get("source_path"), meta.get("model_config", {}).get("source_path")):
        assert src is not None, f"{js.name} missing source_path"
        assert not str(src).startswith("/"), f"{js.name} source_path is absolute: {src!r}"
        assert "/" not in str(src), f"{js.name} source_path contains a path separator: {src!r}"


# ---------------------------------------------------------------------------
# Corrupt manifest shape -> reject
# ---------------------------------------------------------------------------

def test_corrupt_manifest_shape_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    meta = _load_metadata(js)
    # Mutate one manifest shape to a wrong value.
    first_key = next(iter(meta["shape_manifest"]))
    meta["shape_manifest"][first_key] = [999]
    _save_metadata(js, meta)
    with pytest.raises(ValueError, match="shape .* does not match manifest"):
        load_converted_checkpoint(npz)


# ---------------------------------------------------------------------------
# Drop a key from the npz -> reject (missing from npz)
# ---------------------------------------------------------------------------

def test_missing_npz_key_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    with np.load(npz) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    dropped = next(iter(sorted(arrays)))
    del arrays[dropped]
    _rebuild_npz(npz, arrays)
    with pytest.raises(ValueError, match="tensor keys do not match shape_manifest"):
        load_converted_checkpoint(npz)


# ---------------------------------------------------------------------------
# Extra key in npz (not in manifest) -> reject
# ---------------------------------------------------------------------------

def test_extra_npz_key_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    with np.load(npz) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["__bogus/extra"] = np.zeros((1,), dtype=np.float32)
    _rebuild_npz(npz, arrays)
    with pytest.raises(ValueError, match="tensor keys do not match shape_manifest"):
        load_converted_checkpoint(npz)


# ---------------------------------------------------------------------------
# Drop a key from the manifest (npz has it) -> reject (extra in npz)
# ---------------------------------------------------------------------------

def test_missing_manifest_key_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    meta = _load_metadata(js)
    first_key = next(iter(sorted(meta["shape_manifest"])))
    del meta["shape_manifest"][first_key]
    _save_metadata(js, meta)
    with pytest.raises(ValueError, match="tensor keys do not match shape_manifest"):
        load_converted_checkpoint(npz)


# ---------------------------------------------------------------------------
# Wrong dtype (non-float) in npz -> reject
# ---------------------------------------------------------------------------

def test_non_float_dtype_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    with np.load(npz) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    first_key = next(iter(sorted(arrays)))
    arrays[first_key] = arrays[first_key].astype(np.int32)
    _rebuild_npz(npz, arrays)
    with pytest.raises(ValueError, match="non-float32-compatible dtype"):
        load_converted_checkpoint(npz)


# ---------------------------------------------------------------------------
# npz SHA256 mismatch (npz modified after metadata written) -> reject
# ---------------------------------------------------------------------------

def test_npz_sha256_mismatch_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    meta = _load_metadata(js)
    meta["npz_sha256"] = "0" * 64  # bogus hash
    _save_metadata(js, meta)
    with pytest.raises(ValueError, match="SHA256 does not match"):
        load_converted_checkpoint(npz)


# ---------------------------------------------------------------------------
# npz SHA256 present and correct -> loads without warning
# ---------------------------------------------------------------------------

def test_npz_sha256_present_loads_clean(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    from nisqa_jax.checkpoint import _sha256
    meta = _load_metadata(js)
    meta["npz_sha256"] = _sha256(npz)
    _save_metadata(js, meta)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, _ = load_converted_checkpoint(npz)
    assert cfg.output_names
    assert not any("npz_sha256" in str(w.message) for w in caught), (
        "should not warn when npz_sha256 is present and correct"
    )


# ---------------------------------------------------------------------------
# Missing shape_manifest entirely -> reject
# ---------------------------------------------------------------------------

def test_missing_shape_manifest_rejected(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    npz, js = _copy_artifact(tmp_path)
    meta = _load_metadata(js)
    del meta["shape_manifest"]
    _save_metadata(js, meta)
    with pytest.raises(ValueError, match="missing a 'shape_manifest'"):
        load_converted_checkpoint(npz)
