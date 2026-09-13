from __future__ import annotations

import numpy as np
import pytest

from nisqa_jax.double_ended_features import (
    canonical_double_ended_feature_config,
    pack_double_ended_segments,
)


def test_canonical_double_ended_frontend_contract() -> None:
    cfg = canonical_double_ended_feature_config()
    assert cfg.n_fft == 4096
    assert cfg.n_mels == 48
    assert cfg.seg_length == 15
    assert cfg.seg_hop_length == 4
    assert cfg.max_segments == 1300
    assert cfg.fmax == 20000


def test_pair_assembly_preserves_channel_order_and_independent_lengths() -> None:
    degraded = np.arange(3 * 1 * 2 * 3, dtype=np.float32).reshape(3, 1, 2, 3)
    reference = (100 + np.arange(2 * 1 * 2 * 3, dtype=np.float32)).reshape(2, 1, 2, 3)
    pair, lengths = pack_double_ended_segments(degraded, reference)
    assert pair.shape == (1, 3, 2, 2, 3)
    np.testing.assert_array_equal(lengths, np.asarray([[3, 2]], np.int32))
    np.testing.assert_array_equal(pair[0, :, 0], degraded[:, 0])
    np.testing.assert_array_equal(pair[0, :2, 1], reference[:, 0])
    np.testing.assert_array_equal(pair[0, 2, 1], np.zeros((2, 3), np.float32))


def test_pair_assembly_rejects_shape_and_nonfinite_drift() -> None:
    valid = np.ones((2, 1, 2, 3), np.float32)
    with pytest.raises(ValueError, match="feature shapes differ"):
        pack_double_ended_segments(valid, np.ones((2, 1, 3, 3), np.float32))
    bad = valid.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pack_double_ended_segments(bad, valid)
