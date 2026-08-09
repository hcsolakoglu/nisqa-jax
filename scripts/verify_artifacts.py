#!/usr/bin/env python
"""Release-gate artifact verifier for the bundled NISQA-JAX weight artifacts.

Performs three independent integrity checks for every ``.npz`` in the weights
directory (the in-package ``nisqa_jax/weights/`` tree, or the installed
``nisqa_jax.weights`` package for wheel-only environments):

  1. **Checksum gate** — recompute the SHA-256 of each ``.npz`` AND its sibling
     ``.json`` metadata sidecar and compare against the committed
     ``CHECKSUMS.sha256`` manifest (``sha256sum`` format, listing both NPZ and
     JSON files). Catches post-commit bit-rot / tampering / truncation of
     either the weights or the metadata.

  2. **Manifest gate** — load each ``.npz`` and confirm its tensor keys, per-
     tensor shapes, and float-dtype match the ``shape_manifest`` recorded in
     the sibling ``.json`` metadata sidecar. Catches npz/json desync.

  3. **Metadata gate** — validate the ``.json`` sidecar's required fields
     (conversion_version, source_sha256, npz_sha256, model_name, output_names,
     model_config, shape_manifest) and cross-check the embedded ``npz_sha256``
     against the recomputed npz hash and the ``source_path`` is a bare
     filename (no build-machine absolute path leak).

In **strict mode** (``--strict``, the default for release CI), any file on
disk (NPZ or JSON) that is not listed in ``CHECKSUMS.sha256`` -- or any entry
in ``CHECKSUMS.sha256`` that does not exist on disk -- causes a hard failure.
Without ``--strict`` unknown artifacts are reported as warnings (useful during
development).

``--update-checksums`` rewrites ``CHECKSUMS.sha256`` from the current ``.npz``
+ ``.json`` files (after a re-conversion); it does NOT skip the manifest/
metadata gates, so a regenerated artifact whose JSON sidecar is missing/
malformed still fails.

Exits ``0`` only if every artifact passes all gates; any failure exits ``1``
with a per-artifact diagnostic. Designed to run in CI with no optional deps
(jax/torch not required — only numpy + stdlib).

Usage::

    python scripts/verify_artifacts.py                 # strict release gate
    python scripts/verify_artifacts.py --no-strict     # dev (warn on unknown)
    python scripts/verify_artifacts.py --update-checksums   # after re-convert
    python scripts/verify_artifacts.py --weights-dir /path/to/weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nisqa_jax_metadata import canonical_metadata_checksum  # noqa: E402

CHECKSUMS_FILENAME = "CHECKSUMS.sha256"

# Required top-level fields in each .json metadata sidecar.
_REQUIRED_METADATA_FIELDS = (
    "conversion_version",
    "source_path",
    "source_sha256",
    "npz_sha256",
    "model_name",
    "output_names",
    "model_config",
    "shape_manifest",
)
# Required keys inside the nested model_config sub-dict.
_REQUIRED_MODEL_CONFIG_KEYS = (
    "model_name",
    "cnn_model",
    "td",
    "pool",
    "output_names",
    "source_path",
    "source_sha256",
    "feature",
)
# Current conversion version the loader expects (must match checkpoint.py).
_EXPECTED_CONVERSION_VERSION = 4
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX_DIGITS for char in value)


def _default_weights_dir() -> Path:
    """Locate the weights directory: explicit arg > env > repo root > installed package."""
    # Repo-tree layout: scripts/ is one level below the repo root that holds
    # nisqa_jax/weights/ (the canonical in-package location).
    repo_root = Path(__file__).resolve().parent.parent
    pkg_weights = repo_root / "nisqa_jax" / "weights"
    if (pkg_weights / CHECKSUMS_FILENAME).exists():
        return pkg_weights
    # Installed-wheel layout: the nisqa_jax.weights subpackage ships the data.
    try:
        from nisqa_jax.weights import WEIGHTS_DIR

        resolved = Path(WEIGHTS_DIR).resolve()
        if (resolved / CHECKSUMS_FILENAME).exists():
            return resolved
    except ImportError:
        pass
    raise FileNotFoundError(
        f"Could not locate {CHECKSUMS_FILENAME}; pass --weights-dir explicitly."
    )


def _load_checksums(weights_dir: Path) -> dict[str, str]:
    """Parse a sha256sum-format file into {basename: hexdigest}."""
    sums_file = weights_dir / CHECKSUMS_FILENAME
    if not sums_file.exists():
        raise FileNotFoundError(f"Missing checksum manifest: {sums_file}")
    out: dict[str, str] = {}
    for line in sums_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # sha256sum format: "<64-hex>  <filename>" (two spaces, binary/text marker).
        digest, _, name = line.partition("  ")
        if not _is_sha256_hex(digest):
            raise ValueError(f"Malformed checksum line in {sums_file}: {line!r}")
        out[name.strip()] = digest
    return out


def _write_checksums(weights_dir: Path, entries: dict[str, str]) -> None:
    """Write a sha256sum-format CHECKSUMS file (sorted by filename)."""
    lines = [f"{digest}  {name}" for name, digest in sorted(entries.items())]
    (weights_dir / CHECKSUMS_FILENAME).write_text("\n".join(lines) + "\n")


def _check_checksum(npz: Path, expected: str) -> list[str]:
    actual = _sha256(npz)
    if actual != expected:
        return [f"  checksum mismatch for {npz.name}: expected {expected}, got {actual}"]
    return []


def _check_manifest(npz: Path) -> tuple[list[str], dict | None]:
    """Validate the npz tensor keys/shapes/dtypes against its .json shape_manifest.

    Returns (errors, metadata) so the caller can run the metadata gate on the
    same parsed JSON without re-reading it. ``metadata`` is None if the JSON
    could not be parsed at all.
    """
    json_path = npz.with_suffix(".json")
    if not json_path.exists():
        return [f"  missing metadata sidecar: {json_path.name}"], None
    try:
        metadata = json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"  {json_path.name} is not valid JSON: {exc}"], None
    manifest = metadata.get("shape_manifest")
    if not isinstance(manifest, dict):
        return [f"  {json_path.name} missing a 'shape_manifest' dict"], metadata
    errors: list[str] = []
    with np.load(npz) as loaded:
        npz_keys = sorted(loaded.files)
        manifest_keys = sorted(manifest.keys())
        if npz_keys != manifest_keys:
            missing = sorted(set(manifest_keys) - set(npz_keys))
            extra = sorted(set(npz_keys) - set(manifest_keys))
            if missing:
                errors.append(f"  missing from npz: {missing}")
            if extra:
                errors.append(f"  extra in npz (not in manifest): {extra}")
            return errors, metadata
        for name in npz_keys:
            arr = loaded[name]
            expected_shape = tuple(manifest[name])
            if tuple(arr.shape) != expected_shape:
                errors.append(
                    f"  tensor {name!r} shape {tuple(arr.shape)} != manifest {expected_shape}"
                )
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"  tensor {name!r} dtype {arr.dtype} is not floating-point")
    return errors, metadata


def _check_metadata(npz: Path, metadata: dict, recomputed_npz_sha: str, *, strict: bool = True) -> list[str]:
    """Validate the JSON metadata sidecar's required fields and cross-checks.

    ``metadata_sha256`` (the canonical semantic checksum) is required only in
    strict mode: older artifacts predate the field and warn (not fail) in
    non-strict dev mode, but release CI (strict) rejects a sidecar lacking it
    so a tampered/edited metadata payload is caught independently of the
    structural key/shape/dtype checks.
    """
    errors: list[str] = []
    json_path = npz.with_suffix(".json")
    # Required top-level fields.
    for field in _REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errors.append(f"  {json_path.name} missing required field {field!r}")
    if errors:
        return errors  # further checks would cascade on missing keys.
    # conversion_version must match the loader's expected version.
    cv = metadata.get("conversion_version")
    if cv != _EXPECTED_CONVERSION_VERSION:
        errors.append(
            f"  {json_path.name} conversion_version {cv!r} != expected {_EXPECTED_CONVERSION_VERSION}"
        )
    # npz_sha256 must match the recomputed hash (independent of CHECKSUMS).
    embedded_sha = metadata.get("npz_sha256")
    if not isinstance(embedded_sha, str) or not _is_sha256_hex(embedded_sha):
        errors.append(f"  {json_path.name} npz_sha256 is not a 64-hex string: {embedded_sha!r}")
    elif embedded_sha != recomputed_npz_sha:
        errors.append(
            f"  {json_path.name} npz_sha256 {embedded_sha[:12]}… != recomputed {recomputed_npz_sha[:12]}…"
        )
    # source_path must be a bare filename (no absolute path / path separator).
    for src in (metadata.get("source_path"), metadata.get("model_config", {}).get("source_path")):
        if src is None:
            errors.append(f"  {json_path.name} source_path is missing")
        elif str(src).startswith("/") or "/" in str(src):
            errors.append(f"  {json_path.name} source_path is not a bare filename: {src!r}")
    # model_config sub-dict required keys.
    mc = metadata.get("model_config", {})
    if not isinstance(mc, dict):
        errors.append(f"  {json_path.name} model_config is not a dict")
    else:
        for key in _REQUIRED_MODEL_CONFIG_KEYS:
            if key not in mc:
                errors.append(f"  {json_path.name} model_config missing key {key!r}")
        # Cross-check: top-level and nested source_sha256 must agree.
        top_sha = metadata.get("source_sha256")
        nested_sha = mc.get("source_sha256")
        if top_sha != nested_sha:
            errors.append(
                f"  {json_path.name} source_sha256 mismatch: top={top_sha!r} vs model_config={nested_sha!r}"
            )
        # Cross-check: top-level and nested output_names must agree.
        top_out = tuple(metadata.get("output_names", []))
        nested_out = tuple(mc.get("output_names", []))
        if top_out != nested_out:
            errors.append(
                f"  {json_path.name} output_names mismatch: top={top_out} vs model_config={nested_out}"
            )
    # output_names must be a non-empty tuple/list of strings.
    out_names = metadata.get("output_names")
    if not isinstance(out_names, list | tuple) or not out_names or not all(isinstance(n, str) for n in out_names):
        errors.append(f"  {json_path.name} output_names must be a non-empty list of strings, got {out_names!r}")
    # metadata_sha256: the canonical semantic checksum of the metadata payload.
    # Required in strict mode (release CI) so a tampered/edited sidecar is
    # caught independently of the structural checks. Older artifacts predate
    # this field; in strict mode its absence is a failure (re-convert to embed
    # it), in non-strict mode it is reported but does not fail the gate.
    meta_sha = metadata.get("metadata_sha256")
    if "metadata_sha256" not in metadata:
        msg = (
            f"  {json_path.name} missing 'metadata_sha256' "
            "(canonical semantic checksum; re-convert the source checkpoint to embed it)"
        )
        if strict:
            errors.append(msg)
        else:
            print(f"warning:{msg}", file=sys.stderr)
    elif not isinstance(meta_sha, str) or not _is_sha256_hex(meta_sha):
        errors.append(f"  {json_path.name} metadata_sha256 is not a 64-hex string: {meta_sha!r}")
    elif (actual := canonical_metadata_checksum(metadata)) != meta_sha:
        errors.append(
            f"  {json_path.name} canonical metadata checksum does not match "
            f"'metadata_sha256': expected {meta_sha[:12]}… got {actual[:12]}…"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="weights directory (default: auto-detect repo or installed package)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="strict mode (default): fail on unknown/unlisted artifacts; --no-strict warns instead",
    )
    parser.add_argument(
        "--update-checksums",
        action="store_true",
        help="rewrite CHECKSUMS.sha256 from the current .npz + .json files before verifying",
    )
    args = parser.parse_args(argv)
    weights_dir = args.weights_dir.resolve() if args.weights_dir else _default_weights_dir()

    if not weights_dir.is_dir():
        print(f"error: weights dir not found: {weights_dir}", file=sys.stderr)
        return 1

    npz_files = sorted(weights_dir.glob("*.npz"))
    if not npz_files:
        print(f"error: no .npz artifacts in {weights_dir}", file=sys.stderr)
        return 1
    json_files = sorted(weights_dir.glob("*.json"))
    # All tracked files: NPZ artifacts + their JSON metadata sidecars.
    tracked_files = npz_files + json_files

    # --update-checksums: rewrite the manifest from the current artifacts, then
    # continue to the full verification (manifest + metadata gates still run).
    if args.update_checksums:
        entries = {p.name: _sha256(p) for p in tracked_files}
        _write_checksums(weights_dir, entries)
        print(f"updated {CHECKSUMS_FILENAME} ({len(entries)} entries)")

    try:
        checksums = _load_checksums(weights_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Every tracked file on disk (NPZ + JSON) must be in the checksum manifest,
    # and vice versa. This subsumes the old "orphan JSON" check: a JSON sidecar
    # without a matching NPZ is an unlisted file (not in CHECKSUMS) and fails.
    on_disk = {p.name for p in tracked_files}
    listed = set(checksums)
    unlisted = on_disk - listed
    untracked = listed - on_disk
    if unlisted or untracked:
        msg_parts = []
        if unlisted:
            msg_parts.append(f"files missing from {CHECKSUMS_FILENAME}: {sorted(unlisted)}")
        if untracked:
            msg_parts.append(f"{CHECKSUMS_FILENAME} lists non-existent files: {sorted(untracked)}")
        msg = "; ".join(msg_parts)
        if args.strict:
            print(f"error: {msg}", file=sys.stderr)
            return 1
        print(f"warning: {msg}", file=sys.stderr)

    failures = 0
    for npz in npz_files:
        errors: list[str] = []
        # Recompute the npz hash once; used by both the checksum gate and the
        # metadata gate's npz_sha256 cross-check.
        recomputed = _sha256(npz)
        if npz.name in checksums:
            if recomputed != checksums[npz.name]:
                errors.append(
                    f"  checksum mismatch for {npz.name}: expected {checksums[npz.name]}, got {recomputed}"
                )
        # Verify the JSON sidecar's checksum against CHECKSUMS (independent of
        # the metadata gate's internal field validation).
        json_path = npz.with_suffix(".json")
        if json_path.exists() and json_path.name in checksums:
            json_recomputed = _sha256(json_path)
            if json_recomputed != checksums[json_path.name]:
                errors.append(
                    f"  checksum mismatch for {json_path.name}: "
                    f"expected {checksums[json_path.name]}, got {json_recomputed}"
                )
        manifest_errors, metadata = _check_manifest(npz)
        errors += manifest_errors
        if metadata is not None:
            errors += _check_metadata(npz, metadata, recomputed, strict=args.strict)
        if errors:
            failures += 1
            print(f"FAIL {npz.name}")
            for e in errors:
                print(e)
        else:
            print(f"OK   {npz.name}  sha256={recomputed[:12]}…  manifest=valid  metadata=valid")

    if failures:
        print(f"\n{failures} artifact(s) failed verification", file=sys.stderr)
        return 1
    print(f"\nAll {len(npz_files)} artifact(s) verified (checksum + manifest + metadata).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
