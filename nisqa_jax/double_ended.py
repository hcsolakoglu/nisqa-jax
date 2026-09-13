from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .model import ArrayTree, _dense

AlignmentMethod = Literal["bahd", "luong", "dot", "cosine", "distance", "none"]
AlignmentApply = Literal["soft", "hard"]
FusionMethod = Literal["x/y/-", "+/-", "x/y"]


@dataclass(frozen=True)
class DoubleEndedConfig:
    """Runtime contract for upstream NISQA_DE inference profile.

    This intentionally starts with graph variants exercised by upstream
    ``train_nisqa_double_ended.yaml`` and NISQA_DE defaults. Extending arbitrary
    training architectures remains a separate compatibility decision.
    """

    de_align: AlignmentMethod = "cosine"
    de_align_apply: AlignmentApply = "hard"
    de_fuse: FusionMethod = "x/y/-"
    de_fuse_dim: int | None = None

    def __post_init__(self) -> None:
        if self.de_align not in {"bahd", "luong", "dot", "cosine", "distance", "none"}:
            raise ValueError(f"unsupported de_align: {self.de_align!r}")
        if self.de_align_apply not in {"soft", "hard"}:
            raise ValueError(f"unsupported de_align_apply: {self.de_align_apply!r}")
        if self.de_fuse not in {"x/y/-", "+/-", "x/y"}:
            raise ValueError(f"unsupported de_fuse: {self.de_fuse!r}")
        if self.de_fuse_dim is not None and self.de_fuse_dim < 1:
            raise ValueError("de_fuse_dim must be positive when provided")


def alignment_scores(
    params: ArrayTree,
    query: jnp.ndarray,
    y: jnp.ndarray,
    method: AlignmentMethod,
) -> jnp.ndarray:
    """Return upstream-compatible alignment logits ``[batch, query_t, y_t]``."""
    if method == "dot":
        return jnp.einsum("bqd,byd->bqy", query, y)
    if method == "cosine":
        # torch.nn.CosineSimilarity clamps each L2 norm independently at eps.
        eps = jnp.asarray(1e-8, dtype=query.dtype)
        q_norm = jnp.maximum(jnp.linalg.norm(query, axis=-1), eps)
        y_norm = jnp.maximum(jnp.linalg.norm(y, axis=-1), eps)
        dot = jnp.einsum("bqd,byd->bqy", query, y)
        return dot / (q_norm[:, :, None] * y_norm[:, None, :])
    if method == "distance":
        # Upstream AttDistance defaults to dist_norm=1 and weight_norm=1.
        dist = jnp.mean(jnp.abs(query[:, None, :, :] - y[:, :, None, :]), axis=-1)
        return -jnp.swapaxes(dist, 1, 2)
    if method == "bahd":
        q = _dense(query, params["wq"])
        y_proj = _dense(y, params["wy"])
        hidden = jnp.tanh(q[:, None, :, :] + y_proj[:, :, None, :])
        return jnp.swapaxes(_dense(hidden, params["v"]).squeeze(-1), 1, 2)
    if method == "luong":
        return jnp.einsum("bqd,byd->bqy", query, _dense(y, params["w"]))
    raise ValueError("alignment_scores is undefined for de_align='none'")


def align(
    params: ArrayTree,
    query: jnp.ndarray,
    y: jnp.ndarray,
    n_wins_y: jnp.ndarray,
    *,
    method: AlignmentMethod,
    apply: AlignmentApply,
) -> jnp.ndarray:
    """Align ``y`` to query using upstream masking and hard/soft application."""
    if method == "none":
        return y

    scores = alignment_scores(params, query, y, method)
    mask = jnp.arange(y.shape[1], dtype=n_wins_y.dtype)[None, :] < n_wins_y[:, None]
    scores = jnp.where(mask[:, None, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    if apply == "soft":
        return jnp.einsum("bqy,byd->bqd", weights, y)
    if apply == "hard":
        index = jnp.argmax(weights, axis=-1)
        return y[jnp.arange(y.shape[0])[:, None], index]
    raise ValueError(f"unsupported de_align_apply: {apply!r}")


def fuse(
    params: ArrayTree,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    method: FusionMethod,
    fuse_dim: int | None,
) -> jnp.ndarray:
    """Fuse aligned feature sequences exactly like upstream ``Fusion``."""
    if method == "x/y/-":
        out = jnp.concatenate((x, y, x - y), axis=-1)
    elif method == "+/-":
        out = jnp.concatenate((x + y, x - y), axis=-1)
    elif method == "x/y":
        out = jnp.concatenate((x, y), axis=-1)
    else:  # pragma: no cover - DoubleEndedConfig validates this boundary.
        raise ValueError(f"unsupported de_fuse: {method!r}")
    if fuse_dim is not None:
        out = _dense(out, params["linear"])
    return out
