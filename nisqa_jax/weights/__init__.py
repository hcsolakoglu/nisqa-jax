"""Bundled pre-converted NISQA weight artifacts (shipped inside the package).

Exposes :data:`WEIGHTS_DIR` so installed-wheel users can locate the bundled
``.npz`` checkpoints without a repo checkout, via :mod:`importlib.resources`
so it resolves correctly inside a wheel/sdist install (not just an editable
checkout)::

    from nisqa_jax.weights import WEIGHTS_DIR
    from nisqa_jax import load_model
    model = load_model(WEIGHTS_DIR / "nisqa_mos_only.npz")

There is no top-level ``weights`` package; consumers must import from
``nisqa_jax.weights``.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

# Resolve the on-disk location of this package's data files. `files()` returns
# a Traversable rooted at the package directory; `Path()` on its resolved form
# works for the normal filesystem layout used by both editable installs and
# wheels (the .npz files are read via random-access by numpy, so the wheel is
# installed unpacked on disk, not as a zip).
WEIGHTS_DIR: Path = Path(str(resources.files(__name__)))
