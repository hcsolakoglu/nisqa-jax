from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .model import (
    ArrayTree,
    _bidirectional_lstm,
    _cnn_adapt,
    _cnn_standard,
    _dense,
    _pool_att_ff,
    _pool_last_step_bi,
    _self_attention,
)

AlignmentMethod = Literal["bahd", "luong", "dot", "cosine", "distance", "none"]
AlignmentApply = Literal["soft", "hard"]
FusionMethod = Literal["x/y/-", "+/-", "x/y"]
TimeDependency = Literal["self_att", "lstm", "skip"]
PoolingMethod = Literal["att", "last_step_bi"]
CnnMethod = Literal["adapt", "standard"]


@dataclass(frozen=True)
class DoubleEndedConfig:
    """Native JAX runtime contract for the upstream ``NISQA_DE`` graph.

    Defaults mirror ``config/train_nisqa_double_ended.yaml``. The graph is
    intentionally parameter-driven: source checkpoint conversion is a separate
    concern, so random-weight architecture parity can be tested without trusting
    serialized checkpoints.
    """

    cnn_model: CnnMethod = "adapt"
    cnn_pool_1: tuple[int, int] | None = (24, 7)
    cnn_pool_2: tuple[int, int] | None = (12, 5)
    cnn_pool_3: tuple[int, int] | None = (6, 3)
    td: TimeDependency = "self_att"
    td_2: TimeDependency = "self_att"
    pool: PoolingMethod = "att"
    de_align: AlignmentMethod = "cosine"
    de_align_apply: AlignmentApply = "hard"
    de_fuse: FusionMethod = "x/y/-"
    de_fuse_dim: int | None = None

    def __post_init__(self) -> None:
        if self.cnn_model not in {"adapt", "standard"}:
            raise ValueError(f"unsupported cnn_model: {self.cnn_model!r}")
        if self.cnn_model == "adapt" and None in (self.cnn_pool_1, self.cnn_pool_2, self.cnn_pool_3):
            raise ValueError("adaptive CNN requires cnn_pool_1/2/3")
        if self.td not in {"self_att", "lstm", "skip"} or self.td_2 not in {"self_att", "lstm", "skip"}:
            raise ValueError("td and td_2 must be self_att, lstm, or skip")
        if self.pool not in {"att", "last_step_bi"}:
            raise ValueError(f"unsupported pool: {self.pool!r}")
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
        eps = jnp.asarray(1e-8, dtype=query.dtype)
        q_norm = jnp.maximum(jnp.linalg.norm(query, axis=-1), eps)
        y_norm = jnp.maximum(jnp.linalg.norm(y, axis=-1), eps)
        dot = jnp.einsum("bqd,byd->bqy", query, y)
        return dot / (q_norm[:, :, None] * y_norm[:, None, :])
    if method == "distance":
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
    else:  # pragma: no cover
        raise ValueError(f"unsupported de_fuse: {method!r}")
    if fuse_dim is not None:
        out = _dense(out, params["linear"])
    return out


def _run_cnn(params: ArrayTree, x: jnp.ndarray, cfg: DoubleEndedConfig) -> jnp.ndarray:
    if cfg.cnn_model == "adapt":
        return _cnn_adapt(params, x, cfg)  # type: ignore[arg-type]
    return _cnn_standard(params, x)


def _run_td(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray, kind: TimeDependency) -> jnp.ndarray:
    if kind == "self_att":
        return _self_attention(params, x, n_wins)
    if kind == "lstm":
        return _bidirectional_lstm(params, x, n_wins)
    if kind == "skip":
        return x
    raise ValueError(kind)  # pragma: no cover


def _mask_steps(x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
    valid = jnp.arange(x.shape[1], dtype=n_wins.dtype)[None, :] < n_wins[:, None]
    return jnp.where(valid[:, :, None], x, jnp.zeros_like(x))


def forward_double_ended_stages(
    params: ArrayTree,
    x: jnp.ndarray,
    n_wins: jnp.ndarray,
    *,
    cfg: DoubleEndedConfig,
) -> dict[str, jnp.ndarray]:
    """Run source graph: shared CNN/TD, align, fuse, TD2, pool."""
    n_wins = n_wins.astype(jnp.int32)
    x_deg, x_ref = x[:, :, :1], x[:, :, 1:2]
    n_deg, n_ref = n_wins[:, 0], n_wins[:, 1]

    cnn_deg = _mask_steps(_run_cnn(params["cnn"], x_deg, cfg), n_deg)
    cnn_ref = _mask_steps(_run_cnn(params["cnn"], x_ref, cfg), n_ref)
    td_deg = _run_td(params["time_dependency"], cnn_deg, n_deg, cfg.td)
    td_ref = _run_td(params["time_dependency"], cnn_ref, n_ref, cfg.td)
    aligned_ref = align(
        params.get("align", {}),
        td_deg,
        td_ref,
        n_ref,
        method=cfg.de_align,
        apply=cfg.de_align_apply,
    )
    fused = fuse(
        params.get("fuse", {}),
        td_deg,
        aligned_ref,
        method=cfg.de_fuse,
        fuse_dim=cfg.de_fuse_dim,
    )
    td2 = _run_td(params.get("time_dependency_2", {}), fused, n_deg, cfg.td_2)
    if cfg.pool == "att":
        out = _pool_att_ff(params["pool"], td2, n_deg)
    else:
        out = _pool_last_step_bi(params["pool"], td2, n_deg)
    return {
        "cnn_deg": cnn_deg,
        "cnn_ref": cnn_ref,
        "time_dependency_deg": td_deg,
        "time_dependency_ref": td_ref,
        "aligned_ref": aligned_ref,
        "fused": fused,
        "time_dependency_2": td2,
        "pool": out.astype(jnp.float32),
    }


def forward_double_ended(
    params: ArrayTree,
    x: jnp.ndarray,
    n_wins: jnp.ndarray,
    *,
    cfg: DoubleEndedConfig,
) -> jnp.ndarray:
    return forward_double_ended_stages(params, x, n_wins, cfg=cfg)["pool"]


@dataclass
class NisqaDeJaxModel:
    """JIT wrapper for native NISQA_DE parameters."""

    config: DoubleEndedConfig
    params: ArrayTree
    device: jax.Device

    def __post_init__(self) -> None:
        self.params = jax.device_put(self.params, self.device)

        def strict_forward(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
            with jax.default_matmul_precision("float32"):
                return forward_double_ended(params, x, n_wins, cfg=self.config)

        self._forward = jax.jit(strict_forward)

    def __call__(self, x: np.ndarray, n_wins: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray) or x.ndim != 5 or x.shape[2] != 2:
            raise ValueError("x must be [batch, steps, 2, n_mels, segment_length]")
        if not isinstance(n_wins, np.ndarray) or n_wins.ndim != 2 or n_wins.shape != (x.shape[0], 2):
            raise ValueError("n_wins must be integer [batch, 2] counts")
        if not np.issubdtype(n_wins.dtype, np.integer) or np.issubdtype(n_wins.dtype, np.bool_):
            raise ValueError("n_wins must have an integer, non-bool dtype")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values")
        if np.any(n_wins < 1) or np.any(n_wins > x.shape[1]):
            raise ValueError("every n_wins value must be in [1, steps]")
        x_device = jax.device_put(x, self.device)
        n_wins_device = jax.device_put(n_wins, self.device)
        return np.asarray(self._forward(self.params, x_device, n_wins_device))
