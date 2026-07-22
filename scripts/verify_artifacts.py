#!/usr/bin/env python
"""Release-gate artifact verifier for the bundled NISQA-JAX weight artifacts.

Performs two independent integrity checks for every ``.npz`` in the weights
directory (the repo ``weights/`` tree, or the installed ``weights`` package for
wheel-only environments):

  1. **Checksum gate** — recompute the SHA-256 of each ``.npz`` and compare it
     against the committed ``weights/CHECKSUMS.sha256`` manifest (``sha256sum``
     format). Catches post-commit bit-rot / tampering / truncation.

  2. **Manifest gate** — load each ``.npz`` and confirm its tensor keys, per-
     tensor shapes, and float-dtype match the ``shape_manifest`` recorded in the
     sibling ``.json`` metadata sidecar. Catches npz/json desync.

Exits ``0`` only if every artifact passes both gates; any failure exits ``1``
with a per-artifact diagnostic. Designed to run in CI with no optional deps
(jax/torch not required — only numpy + stdlib).

Usage::

    python scripts/verify_artifacts.py
    python scripts/verify_artifacts.py --weights-dir /path/to/weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CHECKSUMS_FILENAME = "CHECKSUMS.sha256"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_weights_dir() -> Path:
    """Locate the weights directory: explicit arg > env > repo root > installed package."""
    # Repo-tree layout: scripts/ is one level below the repo root that holds weights/.
    repo_root = Path(__file__).resolve().parent.parent
    repo_weights = repo_root / "weights"
    if (repo_weights / CHECKSUMS_FILENAME).exists():
        return repo_weights
    # Installed-wheel layout: the ``weights`` package ships next to ``nisqa_jax``.
    try:
        import weights  # type: ignore[import-not-found]

        return Path(weights.__file__).resolve().parent
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
        if len(digest) != 64:
            raise ValueError(f"Malformed checksum line in {sums_file}: {line!r}")
        out[name.strip()] = digest
    return out


def _check_checksum(npz: Path, expected: str) -> list[str]:
    actual = _sha256(npz)
    if actual != expected:
        return [f"  checksum mismatch for {npz.name}: expected {expected}, got {actual}"]
    return []


def _check_manifest(npz: Path) -> list[str]:
    """Validate the npz tensor keys/shapes/dtypes against its .json shape_manifest."""
    json_path = npz.with_suffix(".json")
    if not json_path.exists():
        return [f"  missing metadata sidecar: {json_path.name}"]
    metadata = json.loads(json_path.read_text())
    manifest = metadata.get("shape_manifest")
    if not isinstance(manifest, dict):
        return [f"  {json_path.name} missing a 'shape_manifest' dict"]
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
            return errors
        for name in npz_keys:
            arr = loaded[name]
            expected_shape = tuple(manifest[name])
            if tuple(arr.shape) != expected_shape:
                errors.append(
                    f"  tensor {name!r} shape {tuple(arr.shape)} != manifest {expected_shape}"
                )
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"  tensor {name!r} dtype {arr.dtype} is not floating-point")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="weights directory (default: auto-detect repo or installed package)",
    )
    args = parser.parse_args(argv)
    weights_dir = args.weights_dir.resolve() if args.weights_dir else _default_weights_dir()

    if not weights_dir.is_dir():
        print(f"error: weights dir not found: {weights_dir}", file=sys.stderr)
        return 1

    try:
        checksums = _load_checksums(weights_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    npz_files = sorted(weights_dir.glob("*.npz"))
    if not npz_files:
        print(f"error: no .npz artifacts in {weights_dir}", file=sys.stderr)
        return 1

    # Every npz on disk must be in the checksum manifest, and vice versa.
    on_disk = {p.name for p in npz_files}
    listed = set(checksums)
    unlisted = on_disk - listed
    untracked = listed - on_disk
    if unlisted or untracked:
        if unlisted:
            print(f"error: npz files missing from {CHECKSUMS_FILENAME}: {sorted(unlisted)}", file=sys.stderr)
        if untracked:
            print(f"error: {CHECKSUMS_FILENAME} lists non-existent files: {sorted(untracked)}", file=sys.stderr)
        return 1

    failures = 0
    for npz in npz_files:
        errors: list[str] = []
        errors += _check_checksum(npz, checksums[npz.name])
        errors += _check_manifest(npz)
        if errors:
            failures += 1
            print(f"FAIL {npz.name}")
            for e in errors:
                print(e)
        else:
            print(f"OK   {npz.name}  sha256={checksums[npz.name][:12]}…  manifest=valid")

    if failures:
        print(f"\n{failures} artifact(s) failed verification", file=sys.stderr)
        return 1
    print(f"\nAll {len(npz_files)} artifact(s) verified (checksum + manifest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
