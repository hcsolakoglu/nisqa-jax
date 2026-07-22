"""Bundled pre-converted NISQA weight artifacts (shipped with the package).

Exposes :data:`WEIGHTS_DIR` so installed-wheel users can locate the bundled
``.npz`` checkpoints without a repo checkout::

    from weights import WEIGHTS_DIR
    from nisqa_jax import load_model
    model = load_model(WEIGHTS_DIR / "nisqa_mos_only.npz")
"""
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent
