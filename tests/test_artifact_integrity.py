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
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))
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


# ---------------------------------------------------------------------------
# Release verifier (scripts/verify_artifacts.py): strict mode + metadata gate
# ---------------------------------------------------------------------------

def _run_verifier(weights_dir: Path, *extra: str) -> tuple[int, str, str]:
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_artifacts.py"), "--weights-dir", str(weights_dir), *extra],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _copy_weights_to_tmp(tmp_path: Path, *, with_checksums: bool = True) -> Path:
    """Copy shipped NPZ + JSON + LICENSE (+ CHECKSUMS) into a temp weights dir."""
    _skip_if_weights_missing()
    wdir = tmp_path / "weights"
    wdir.mkdir()
    for p in WEIGHTS_ROOT.glob("*.npz"):
        shutil.copy2(p, wdir / p.name)
    for p in WEIGHTS_ROOT.glob("*.json"):
        shutil.copy2(p, wdir / p.name)
    shutil.copy2(WEIGHTS_ROOT / "LICENSE_model_weights", wdir / "LICENSE_model_weights")
    if with_checksums:
        shutil.copy2(WEIGHTS_ROOT / "CHECKSUMS.sha256", wdir / "CHECKSUMS.sha256")
    return wdir


def test_verifier_passes_on_shipped_artifacts() -> None:
    """The release verifier must pass cleanly on the shipped in-package artifacts."""
    from nisqa_jax.weights import WEIGHTS_DIR

    if not (WEIGHTS_DIR / "CHECKSUMS.sha256").exists():
        pytest.skip("shipped weights unavailable")
    rc, out, err = _run_verifier(WEIGHTS_DIR)
    assert rc == 0, f"verifier failed on shipped artifacts:\nstdout={out}\nstderr={err}"
    assert "metadata=valid" in out


def test_verifier_strict_fails_on_unknown_npz(tmp_path: Path) -> None:
    """Strict mode must fail when an npz on disk is not in CHECKSUMS.sha256."""
    wdir = _copy_weights_to_tmp(tmp_path)
    # Add an unlisted unknown npz -> strict must fail.
    shutil.copy2(WEIGHTS_ROOT / "nisqa_mos_only.npz", wdir / "_unknown.npz")
    rc, out, err = _run_verifier(wdir)
    assert rc != 0
    assert "_unknown.npz" in err


def test_verifier_strict_fails_on_unknown_json(tmp_path: Path) -> None:
    """Strict mode must fail when a JSON sidecar on disk is not in CHECKSUMS.sha256."""
    wdir = _copy_weights_to_tmp(tmp_path)
    # Add an unlisted unknown json -> strict must fail.
    (wdir / "_unknown.json").write_text('{"test": 1}')
    rc, out, err = _run_verifier(wdir)
    assert rc != 0
    assert "_unknown.json" in err


def test_verifier_checksums_includes_json_entries() -> None:
    """CHECKSUMS.sha256 must list SHA-256 for both NPZ and JSON files."""
    _skip_if_weights_missing()
    sums = (WEIGHTS_ROOT / "CHECKSUMS.sha256").read_text()
    npz_names = {p.name for p in WEIGHTS_ROOT.glob("*.npz")}
    json_names = {p.name for p in WEIGHTS_ROOT.glob("*.json")}
    listed = {line.split("  ", 1)[1].strip() for line in sums.splitlines() if line.strip() and not line.startswith("#")}
    assert npz_names <= listed, f"NPZ files missing from CHECKSUMS: {npz_names - listed}"
    assert json_names <= listed, f"JSON files missing from CHECKSUMS: {json_names - listed}"


def test_verifier_metadata_gate_catches_missing_field(tmp_path: Path) -> None:
    """The metadata gate must fail when a required JSON field is missing."""
    wdir = _copy_weights_to_tmp(tmp_path)
    # Remove a required field from one metadata sidecar (this also changes the
    # JSON checksum, so the checksum gate catches it too; the metadata gate
    # surfaces the specific missing-field diagnostic).
    js = wdir / "nisqa_mos_only.json"
    meta = _load_metadata(js)
    del meta["npz_sha256"]
    _save_metadata(js, meta)
    rc, out, err = _run_verifier(wdir)
    assert rc != 0
    assert "FAIL" in out
    assert "npz_sha256" in out


def test_verifier_update_checksums_roundtrip(tmp_path: Path) -> None:
    """--update-checksums rewrites CHECKSUMS.sha256 (NPZ + JSON) and then passes."""
    wdir = _copy_weights_to_tmp(tmp_path, with_checksums=False)
    # No CHECKSUMS file yet -> update must create it and then verify clean.
    rc, out, err = _run_verifier(wdir, "--update-checksums")
    assert rc == 0, f"update+verify failed:\nstdout={out}\nstderr={err}"
    assert (wdir / "CHECKSUMS.sha256").exists()
    assert "updated" in out
    # The rewritten CHECKSUMS must list both NPZ and JSON files.
    sums = (wdir / "CHECKSUMS.sha256").read_text()
    listed = {line.split("  ", 1)[1].strip() for line in sums.splitlines() if line.strip() and not line.startswith("#")}
    npz_names = {p.name for p in wdir.glob("*.npz")}
    json_names = {p.name for p in wdir.glob("*.json")}
    assert npz_names <= listed, f"NPZ missing from rewritten CHECKSUMS: {npz_names - listed}"
    assert json_names <= listed, f"JSON missing from rewritten CHECKSUMS: {json_names - listed}"


def test_verifier_strict_requires_metadata_sha256(tmp_path: Path) -> None:
    """Strict mode must fail when a JSON sidecar lacks metadata_sha256.

    metadata_sha256 is the canonical semantic checksum; its absence in strict
    mode (release CI) is a hard failure so a tampered/edited metadata payload
    cannot pass the gate. We strip only metadata_sha256 (and fix the JSON
    checksum so the checksum gate does not fire first) to isolate the
    metadata_sha256 requirement.
    """
    from nisqa_jax.checkpoint import _sha256

    wdir = _copy_weights_to_tmp(tmp_path)
    js = wdir / "nisqa_mos_only.json"
    meta = _load_metadata(js)
    meta.pop("metadata_sha256", None)
    _save_metadata(js, meta)
    # Rewrite CHECKSUMS so the JSON checksum gate does not fire first (the
    # metadata_sha256 requirement must be the diagnosed failure).
    sums = (wdir / "CHECKSUMS.sha256").read_text()
    lines = []
    for line in sums.splitlines():
        if line.strip() and not line.startswith("#") and line.endswith("nisqa_mos_only.json"):
            digest, _, name = line.partition("  ")
            lines.append(f"{_sha256(js)}  {name.strip()}")
        else:
            lines.append(line)
    (wdir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    rc, out, err = _run_verifier(wdir, "--strict")
    assert rc != 0, "strict verifier must fail on missing metadata_sha256"
    assert "metadata_sha256" in out


def test_verifier_strict_rejects_tampered_metadata_hash(tmp_path: Path) -> None:
    """Strict mode must recompute and reject a forged canonical metadata hash."""
    from nisqa_jax.checkpoint import _sha256

    wdir = _copy_weights_to_tmp(tmp_path)
    js = wdir / "nisqa_mos_only.json"
    meta = _load_metadata(js)
    meta["metadata_sha256"] = "0" * 64
    _save_metadata(js, meta)

    # Keep the raw JSON checksum consistent so this isolates the canonical
    # metadata checksum gate rather than failing at the byte-level manifest.
    sums = (wdir / "CHECKSUMS.sha256").read_text()
    lines = []
    for line in sums.splitlines():
        if line.strip() and not line.startswith("#") and line.endswith("nisqa_mos_only.json"):
            _digest, _, name = line.partition("  ")
            lines.append(f"{_sha256(js)}  {name.strip()}")
        else:
            lines.append(line)
    (wdir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")

    rc, out, err = _run_verifier(wdir, "--strict")
    assert rc != 0, "strict verifier must reject a forged metadata_sha256"
    assert "canonical metadata checksum does not match" in out
    assert err


def test_verifier_non_strict_warns_on_missing_metadata_sha256(tmp_path: Path) -> None:
    """Non-strict mode must warn (not fail) on missing metadata_sha256."""
    from nisqa_jax.checkpoint import _sha256

    wdir = _copy_weights_to_tmp(tmp_path)
    js = wdir / "nisqa_mos_only.json"
    meta = _load_metadata(js)
    meta.pop("metadata_sha256", None)
    _save_metadata(js, meta)
    sums = (wdir / "CHECKSUMS.sha256").read_text()
    lines = []
    for line in sums.splitlines():
        if line.strip() and not line.startswith("#") and line.endswith("nisqa_mos_only.json"):
            digest, _, name = line.partition("  ")
            lines.append(f"{_sha256(js)}  {name.strip()}")
        else:
            lines.append(line)
    (wdir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    rc, out, err = _run_verifier(wdir, "--no-strict")
    assert rc == 0, "non-strict verifier must not fail on missing metadata_sha256"
    assert "metadata_sha256" in err  # warned, not failed


def test_verifier_weights_dir_inspects_installed_wheel_path(tmp_path: Path) -> None:
    """--weights-dir must let the verifier inspect an arbitrary directory (wheel path).

    Simulates the CI installed-wheel verifier step: the weights dir is passed
    explicitly (resolved from the installed package in real CI) rather than
    auto-detected from the source tree.
    """
    wdir = _copy_weights_to_tmp(tmp_path)
    rc, out, err = _run_verifier(wdir, "--strict")
    assert rc == 0, f"verifier failed on copied weights dir:\nstdout={out}\nstderr={err}"
    assert "metadata=valid" in out
