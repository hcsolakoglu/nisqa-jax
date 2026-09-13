from __future__ import annotations

import numpy as np

from nisqa_jax.double_ended import DoubleEndedConfig, forward_double_ended
from test_double_ended import _full_params


def test_invalid_reference_tail_cannot_change_full_graph_output() -> None:
    """Static JAX padding must preserve upstream packed-reference semantics."""
    rng = np.random.default_rng(41)
    params = _full_params(seed=43)
    baseline = rng.normal(0, 0.2, (1, 4, 2, 48, 15)).astype(np.float32)
    lengths = np.asarray([[4, 2]], dtype=np.int32)

    adversarial = baseline.copy()
    adversarial[:, 2:, 1] = rng.normal(0, 1e4, adversarial[:, 2:, 1].shape).astype(np.float32)

    expected = forward_double_ended(params, baseline, lengths, cfg=DoubleEndedConfig())
    actual = forward_double_ended(params, adversarial, lengths, cfg=DoubleEndedConfig())

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=1e-6)
