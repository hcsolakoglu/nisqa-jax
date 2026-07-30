"""Build-content gate: assert the wheel and sdist carry the required files.

Runs after ``python -m build`` in CI (and locally) to catch packaging
regressions where a needed artifact, script, test, or golden fixture is
silently dropped from the distribution. The wheel must include the bundled
weight artifacts (so a fresh install has zero-config inference); the sdist
must additionally carry the tests, scripts, golden fixtures, and docs so a
source install can reproduce the full CI gate.

This test does NOT build itself (that is CI's job); it inspects whatever
``dist/`` already contains. If ``dist/`` is empty it skips, so it is safe to
run in the normal pytest suite (it only activates after a build).
"""
from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(os.environ.get("NISQA_JAX_DIST_DIR", ROOT / "dist"))


def _find_wheel() -> Path | None:
    wheels = sorted(DIST.glob("nisqa_jax-*.whl"))
    return wheels[-1] if wheels else None


def _find_sdist() -> Path | None:
    sdists = sorted(DIST.glob("nisqa_jax-*.tar.gz"))
    return sdists[-1] if sdists else None


def _wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        return {name for name in zf.namelist() if not name.endswith("/")}


def _sdist_names(sdist: Path) -> set[str]:
    with tarfile.open(sdist, "r:gz") as tf:
        return {m.name for m in tf.getmembers() if m.isfile()}


# Required wheel data files: bundled weight artifacts inside nisqa_jax/weights.
_REQUIRED_WHEEL_WEIGHTS = [
    "nisqa_jax/weights/nisqa.npz",
    "nisqa_jax/weights/nisqa_mos_only.npz",
    "nisqa_jax/weights/nisqa_tts.npz",
    "nisqa_jax/weights/nisqa.json",
    "nisqa_jax/weights/nisqa_mos_only.json",
    "nisqa_jax/weights/nisqa_tts.json",
    "nisqa_jax/weights/CHECKSUMS.sha256",
    "nisqa_jax/weights/LICENSE_model_weights",
]
# Required wheel code modules.
_REQUIRED_WHEEL_CODE = [
    "nisqa_jax/__init__.py",
    "nisqa_jax/model.py",
    "nisqa_jax/checkpoint.py",
    "nisqa_jax/config.py",
    "nisqa_jax/features.py",
    "nisqa_jax/predict.py",
    "nisqa_jax/weights/__init__.py",
]
# Required sdist-only files (tests, scripts, golden, docs, CI).
_REQUIRED_SDIST_FILES = [
    "scripts/verify_artifacts.py",
    "scripts/generate_golden_fixtures.py",
    "tests/_testutil.py",
    "tests/test_jax_port.py",
    "tests/test_validation.py",
    "tests/test_artifact_integrity.py",
    "tests/test_prewarm.py",
    "tests/test_golden_parity.py",
    "tests/test_audio_frontend_regression.py",
    "tests/test_build_contents.py",
    "tests/fixtures/audio_frontend_regression.json",
    "tests/fixtures/audio_frontend_regression.npz.b64",
    "tests/fixtures/audio_frontend_regression.wav.gz.b64",
    "tests/golden/GOLDEN_MANIFEST.json",
    "tests/golden/nisqa.golden.json",
    "tests/golden/nisqa_mos_only.golden.json",
    "tests/golden/nisqa_tts.golden.json",
    "tests/golden/nisqa.golden.npz",
    "tests/golden/nisqa_mos_only.golden.npz",
    "tests/golden/nisqa_tts.golden.npz",
    # PyTorch reference trust chain: real PyTorch outputs committed alongside
    # the JAX golden vectors. The sdist must carry these so a source install can
    # reproduce the full golden parity gate (which compares current JAX outputs
    # directly against the PyTorch reference).
    "tests/golden/nisqa.ptref.npz",
    "tests/golden/nisqa_mos_only.ptref.npz",
    "tests/golden/nisqa_tts.ptref.npz",
    "MANIFEST.in",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "CITATION.cff",
    "requirements-jax.txt",
    "requirements-gpu.txt",
]


def _sdist_prefix(sdist_names: set[str]) -> str:
    """sdist files are prefixed with 'nisqa_jax-<version>/'; derive that prefix.

    Every file path starts with the top-level ``<name>-<version>/`` directory,
    so the prefix is the substring up to and including the first ``/``.
    """
    if not sdist_names:
        return ""
    sample = next(iter(sdist_names))
    return sample.split("/", 1)[0] + "/"


def test_wheel_includes_bundled_weights() -> None:
    wheel = _find_wheel()
    if wheel is None:
        pytest.skip(f"no wheel in {DIST}; run `python -m build` first")
    names = _wheel_names(wheel)
    missing = [f for f in _REQUIRED_WHEEL_WEIGHTS if f not in names]
    assert not missing, f"wheel missing weight data files: {missing}\nwheel contents (sample): {sorted(names)[:20]}"


def test_wheel_includes_code_modules() -> None:
    wheel = _find_wheel()
    if wheel is None:
        pytest.skip(f"no wheel in {DIST}; run `python -m build` first")
    names = _wheel_names(wheel)
    missing = [f for f in _REQUIRED_WHEEL_CODE if f not in names]
    assert not missing, f"wheel missing code modules: {missing}"


def test_wheel_excludes_tests_and_scripts() -> None:
    """Tests/scripts/golden must NOT ship in the wheel (they are sdist-only)."""
    wheel = _find_wheel()
    if wheel is None:
        pytest.skip(f"no wheel in {DIST}; run `python -m build` first")
    names = _wheel_names(wheel)
    leaked = [n for n in names if n.startswith("tests/") or n.startswith("scripts/") or "/golden/" in n]
    assert not leaked, f"wheel leaked sdist-only files: {leaked}"


def test_wheel_top_level_is_only_nisqa_jax() -> None:
    """The wheel must contain exactly one top-level package: nisqa_jax.

    No legacy ``weights`` package shim may leak into the wheel — consumers
    must use ``nisqa_jax.weights.WEIGHTS_DIR``.
    """
    wheel = _find_wheel()
    if wheel is None:
        pytest.skip(f"no wheel in {DIST}; run `python -m build` first")
    names = _wheel_names(wheel)
    # Derive top-level packages from the wheel file listing: the first path
    # component of every .py/.dist-info entry.
    top_level = {n.split("/")[0] for n in names if "/" in n and not n.startswith(".")}
    # The dist-info directory is expected (metadata, not a package).
    dist_info = {t for t in top_level if t.endswith(".dist-info")}
    packages = top_level - dist_info
    assert packages == {"nisqa_jax"}, (
        f"wheel top-level must be only nisqa_jax, got {sorted(packages)} "
        f"(dist-info: {sorted(dist_info)})"
    )
    # Explicitly assert no legacy weights package.
    assert "weights" not in packages, "wheel contains a legacy top-level `weights` package"


def test_sdist_includes_tests_scripts_golden_docs() -> None:
    sdist = _find_sdist()
    if sdist is None:
        pytest.skip(f"no sdist in {DIST}; run `python -m build` first")
    names = _sdist_names(sdist)
    prefix = _sdist_prefix(names)
    missing = [f for f in _REQUIRED_SDIST_FILES if f"{prefix}{f}" not in names]
    assert not missing, f"sdist missing required files: {missing}\nprefix={prefix!r}"


def test_sdist_includes_bundled_weights() -> None:
    """The sdist must carry the bundled weight artifacts for source installs."""
    sdist = _find_sdist()
    if sdist is None:
        pytest.skip(f"no sdist in {DIST}; run `python -m build` first")
    names = _sdist_names(sdist)
    prefix = _sdist_prefix(names)
    required = [
        f"nisqa_jax/weights/{p}"
        for p in ("nisqa.npz", "nisqa_mos_only.npz", "nisqa_tts.npz", "CHECKSUMS.sha256")
    ]
    missing = [f for f in required if f"{prefix}{f}" not in names]
    assert not missing, f"sdist missing bundled weight artifacts: {missing}"


def test_sdist_excludes_egg_info() -> None:
    """The sdist must not contain egg-info metadata directory contents.

    setuptools may leave a ``*.egg-info`` metadata directory (PKG-INFO,
    entry_points.txt, requires.txt, dependency_links.txt, top_level.txt) in the
    source tree during local builds; MANIFEST.in ``prune *.egg-info`` removes
    all of those from the sdist. A leaked egg-info is build-machine-specific
    cruft that does not belong in a reproducible source distribution.

    NOTE on SOURCES.txt: setuptools' ``sdist.run()`` unconditionally appends
    ``<pkg>.egg-info/SOURCES.txt`` to the file list *after* MANIFEST.in is
    processed (verified in setuptools 70.x source: ``self.filelist.append(
    os.path.join(ei_cmd.egg_info, 'SOURCES.txt'))``), so MANIFEST ``prune``
    cannot remove it. SOURCES.txt is the auto-generated source-file manifest
    (not build-machine-specific metadata) and is the sole permitted egg-info
    artifact; every other egg-info file must be pruned.
    """
    sdist = _find_sdist()
    if sdist is None:
        pytest.skip(f"no sdist in {DIST}; run `python -m build` first")
    names = _sdist_names(sdist)
    egg_info_entries = [n for n in names if ".egg-info/" in n or n.endswith(".egg-info")]
    # The only permitted egg-info artifact is the setuptools-mandated SOURCES.txt
    # (and its parent directory entry). Any other egg-info file is a prune leak.
    permitted = {n for n in egg_info_entries if n.endswith("/SOURCES.txt")}
    # The bare directory entry for the egg-info dir accompanies SOURCES.txt.
    permitted_dir = {n for n in egg_info_entries if n.endswith(".egg-info") and not n.endswith("/")}
    leaked = [n for n in egg_info_entries if n not in permitted and n not in permitted_dir]
    assert not leaked, (
        f"sdist leaked egg-info contents (only SOURCES.txt is permitted): {leaked}"
    )
    # Sanity: if any egg-info entry is present, it must be exactly SOURCES.txt
    # (+ its dir). If setuptools changes to stop force-adding SOURCES.txt, this
    # still passes (empty egg_info_entries); if it starts adding other files,
    # the leaked check above catches them.
    non_sources = [n for n in egg_info_entries if not n.endswith("/SOURCES.txt") and not n.endswith(".egg-info")]
    assert not non_sources, f"sdist leaked non-SOURCES.txt egg-info contents: {non_sources}"
