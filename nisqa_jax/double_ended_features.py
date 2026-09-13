from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import FeatureConfig
from .features import preprocess_file


def canonical_double_ended_feature_config() -> FeatureConfig:
    """Frontend contract from upstream ``train_nisqa_double_ended.yaml``."""
    return FeatureConfig(
        sr=None,
        n_fft=4096,
        hop_length_seconds=0.01,
        win_length_seconds=0.02,
        n_mels=48,
        fmax=20000,
        seg_length=15,
        seg_hop_length=4,
        max_segments=1300,
    )


def pack_double_ended_segments(
    degraded: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad and concatenate independently segmented degraded/reference features.

    Inputs are real (unpadded) segment tensors ``[steps, 1, mels, segment]``.
    The returned pair follows upstream channel ordering and has shape
    ``[1, max_steps, 2, mels, segment]`` with lengths ``[1, 2]``.
    """
    for name, value in (("degraded", degraded), ("reference", reference)):
        if not isinstance(value, np.ndarray) or value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{name} must be [steps, 1, mels, segment]")
        if value.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one segment")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point features")
    if degraded.shape[2:] != reference.shape[2:]:
        raise ValueError(
            f"degraded/reference feature shapes differ: {degraded.shape[2:]} vs {reference.shape[2:]}"
        )

    n_deg, n_ref = degraded.shape[0], reference.shape[0]
    steps = max(n_deg, n_ref)
    pair = np.zeros((steps, 2, degraded.shape[2], degraded.shape[3]), dtype=np.float32)
    pair[:n_deg, 0] = degraded[:, 0].astype(np.float32, copy=False)
    pair[:n_ref, 1] = reference[:, 0].astype(np.float32, copy=False)
    lengths = np.asarray([[n_deg, n_ref]], dtype=np.int32)
    return np.ascontiguousarray(pair[None]), lengths


def preprocess_pair(
    degraded_path: str | Path,
    reference_path: str | Path,
    cfg: FeatureConfig | None = None,
    *,
    degraded_channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create NISQA_DE input from a degraded/reference audio pair.

    Upstream applies its optional ``ms_channel`` only to the degraded file; the
    reference path is loaded in default mono mode. This function preserves that
    asymmetric behavior intentionally.
    """
    feature_cfg = canonical_double_ended_feature_config() if cfg is None else cfg
    degraded, _ = preprocess_file(degraded_path, feature_cfg, channel=degraded_channel)
    reference, _ = preprocess_file(reference_path, feature_cfg, channel=None)
    return pack_double_ended_segments(degraded, reference)
