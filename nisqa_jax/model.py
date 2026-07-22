from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig

ArrayTree = dict[str, Any]
Precision = Literal["float32", "bf16"]


def _validate_precision(precision: str) -> Precision:
    if precision not in {"float32", "bf16"}:
        raise ValueError(f"precision must be 'float32' or 'bf16', got {precision!r}")
    return precision  # type: ignore[return-value]


def _compute_dtype(precision: Precision) -> jnp.dtype:
    return jnp.bfloat16 if precision == "bf16" else jnp.float32


def _cast_tree_for_compute(tree: Any, dtype: jnp.dtype) -> Any:
    if dtype == jnp.float32:
        return tree
    if isinstance(tree, dict):
        return {key: _cast_tree_for_compute(value, dtype) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_cast_tree_for_compute(value, dtype) for value in tree)
    if isinstance(tree, list):
        return [_cast_tree_for_compute(value, dtype) for value in tree]
    arr = jnp.asarray(tree)
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return arr.astype(dtype)
    return arr


def _same_dtype_scalar(value: float, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(value, dtype=dtype)


def _dense(x: jnp.ndarray, p: ArrayTree) -> jnp.ndarray:
    return jnp.matmul(x, p["w"]) + p["b"]


def _conv2d_nhwc(x: jnp.ndarray, p: ArrayTree, padding: tuple[tuple[int, int], tuple[int, int]]) -> jnp.ndarray:
    y = jax.lax.conv_general_dilated(
        x,
        p["w"],
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + p["b"]


def _layer_norm(x: jnp.ndarray, p: ArrayTree, eps: float = 1e-5) -> jnp.ndarray:
    out_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
    y = (x32 - mean) * jax.lax.rsqrt(var + eps) * p["scale"].astype(jnp.float32) + p["bias"].astype(jnp.float32)
    return y.astype(out_dtype)


def _adaptive_max_pool2d_nhwc(x: jnp.ndarray, output_size: tuple[int, int]) -> jnp.ndarray:
    in_h, in_w = x.shape[1], x.shape[2]
    out_h, out_w = output_size

    def regular_bins(in_size: int, out_size: int) -> tuple[int, int] | None:
        starts = [int(np.floor(i * in_size / out_size)) for i in range(out_size)]
        ends = [int(np.ceil((i + 1) * in_size / out_size)) for i in range(out_size)]
        widths = [end - start for start, end in zip(starts, ends)]
        strides = [starts[i + 1] - starts[i] for i in range(out_size - 1)]
        if len(set(widths)) == 1 and (not strides or len(set(strides)) == 1):
            return widths[0], strides[0] if strides else widths[0]
        return None

    h_regular = regular_bins(in_h, out_h)
    if h_regular is not None:
        kernel_h, stride_h = h_regular
        x = jax.lax.reduce_window(
            x,
            _same_dtype_scalar(-jnp.inf, x.dtype),
            jax.lax.max,
            window_dimensions=(1, kernel_h, 1, 1),
            window_strides=(1, stride_h, 1, 1),
            padding="VALID",
        )
    else:
        rows = []
        for oh in range(out_h):
            h0 = int(np.floor(oh * in_h / out_h))
            h1 = int(np.ceil((oh + 1) * in_h / out_h))
            rows.append(jnp.max(x[:, h0:h1, :, :], axis=1))
        x = jnp.stack(rows, axis=1)

    w_regular = regular_bins(in_w, out_w)
    if w_regular is not None:
        kernel_w, stride_w = w_regular
        return jax.lax.reduce_window(
            x,
            _same_dtype_scalar(-jnp.inf, x.dtype),
            jax.lax.max,
            window_dimensions=(1, 1, kernel_w, 1),
            window_strides=(1, 1, stride_w, 1),
            padding="VALID",
        )

    cols = []
    for ow in range(out_w):
        w0 = int(np.floor(ow * in_w / out_w))
        w1 = int(np.ceil((ow + 1) * in_w / out_w))
        cols.append(jnp.max(x[:, :, w0:w1, :], axis=2))
    return jnp.stack(cols, axis=2)


def _max_pool2d_nhwc(
    x: jnp.ndarray,
    *,
    kernel: int,
    stride: int,
    padding: tuple[tuple[int, int], tuple[int, int]],
) -> jnp.ndarray:
    x = jnp.pad(
        x,
        ((0, 0), padding[0], padding[1], (0, 0)),
        mode="constant",
        constant_values=_same_dtype_scalar(-jnp.inf, x.dtype),
    )
    return jax.lax.reduce_window(
        x,
        _same_dtype_scalar(-jnp.inf, x.dtype),
        jax.lax.max,
        window_dimensions=(1, kernel, kernel, 1),
        window_strides=(1, stride, stride, 1),
        padding="VALID",
    )


def _cnn_adapt(params: ArrayTree, x: jnp.ndarray, cfg: ModelConfig) -> jnp.ndarray:
    bsz, steps = x.shape[0], x.shape[1]
    x = x.reshape((bsz * steps, x.shape[2], x.shape[3], x.shape[4]))
    x = jnp.transpose(x, (0, 2, 3, 1))
    pad = ((1, 1), (1, 1))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv1"], pad))
    x = _adaptive_max_pool2d_nhwc(x, cfg.cnn_pool_1)
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv2"], pad))
    x = _adaptive_max_pool2d_nhwc(x, cfg.cnn_pool_2)
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv3"], pad))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv4"], pad))
    x = _adaptive_max_pool2d_nhwc(x, cfg.cnn_pool_3)
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv5"], pad))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv6"], ((1, 1), (0, 0))))
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape((bsz, steps, -1))
    return x


def _cnn_standard(params: ArrayTree, x: jnp.ndarray) -> jnp.ndarray:
    bsz, steps = x.shape[0], x.shape[1]
    x = x.reshape((bsz * steps, x.shape[2], x.shape[3], x.shape[4]))
    x = jnp.transpose(x, (0, 2, 3, 1))
    pad = ((1, 1), (1, 1))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv1"], pad))
    x = _max_pool2d_nhwc(x, kernel=2, stride=2, padding=((0, 0), (1, 1)))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv2"], pad))
    x = _max_pool2d_nhwc(x, kernel=2, stride=2, padding=((0, 0), (0, 0)))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv3"], pad))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv4"], pad))
    x = _max_pool2d_nhwc(x, kernel=2, stride=2, padding=((0, 0), (0, 0)))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv5"], pad))
    x = jax.nn.relu(_conv2d_nhwc(x, params["conv6"], pad))
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape((bsz, steps, -1))
    return _dense(x, params["fc_out"])


def _self_attention_layer(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
    q, k, v = jnp.split(_dense(x, params["in_proj"]), 3, axis=-1)
    dim = q.shape[-1]
    scores = jnp.einsum("btd,bsd->bts", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = scores / jnp.sqrt(jnp.asarray(dim, dtype=jnp.float32))
    key_mask = jnp.arange(x.shape[1], dtype=n_wins.dtype)[None, :] >= n_wins[:, None]
    scores = jnp.where(key_mask[:, None, :], -jnp.inf, scores)
    att = jax.nn.softmax(scores, axis=-1)
    y = _dense(jnp.einsum("bts,bsd->btd", att, v.astype(jnp.float32)).astype(x.dtype), params["out"])
    x = _layer_norm(x + y, params["norm1"])
    y = _dense(jax.nn.relu(_dense(x, params["linear1"])), params["linear2"])
    return _layer_norm(x + y, params["norm2"])


def _self_attention(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
    x = _dense(x, params["input"])
    x = _layer_norm(x, params["norm1"])
    for layer in params["layers"]:
        x = _self_attention_layer(layer, x, n_wins)
    return x


def _lstm_direction(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray, *, reverse: bool) -> jnp.ndarray:
    bsz, steps = x.shape[0], x.shape[1]
    hidden_size = params["w_hh"].shape[0]

    def step(carry: tuple[jnp.ndarray, jnp.ndarray], item: tuple[jnp.ndarray, jnp.ndarray]):
        h, c = carry
        x_t, valid = item
        gates = (
            jnp.matmul(x_t, params["w_ih"])
            + jnp.matmul(h, params["w_hh"])
            + params["b_ih"]
            + params["b_hh"]
        )
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        c_new = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)
        valid = valid[:, None]
        h = jnp.where(valid, h_new, h)
        c = jnp.where(valid, c_new, c)
        out = jnp.where(valid, h, jnp.zeros_like(h))
        return (h, c), out

    time = jnp.arange(steps, dtype=n_wins.dtype)
    valid = time[None, :] < n_wins[:, None]
    seq = jnp.swapaxes(x, 0, 1)
    valid_seq = jnp.swapaxes(valid, 0, 1)
    if reverse:
        seq = seq[::-1]
        valid_seq = valid_seq[::-1]
    init = (
        jnp.zeros((bsz, hidden_size), dtype=x.dtype),
        jnp.zeros((bsz, hidden_size), dtype=x.dtype),
    )
    _, out = jax.lax.scan(step, init, (seq, valid_seq), unroll=32)
    if reverse:
        out = out[::-1]
    return jnp.swapaxes(out, 0, 1).astype(x.dtype)


def _bidirectional_lstm(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
    fw = _lstm_direction(params["forward"], x, n_wins, reverse=False)
    bw = _lstm_direction(params["reverse"], x, n_wins, reverse=True)
    return jnp.concatenate([fw, bw], axis=-1)


def _pool_att_ff(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
    att = _dense(jax.nn.relu(_dense(x, params["linear1"])), params["linear2"]).astype(jnp.float32).squeeze(-1)
    mask = jnp.arange(x.shape[1], dtype=n_wins.dtype)[None, :] >= n_wins[:, None]
    att = jnp.where(mask, -jnp.inf, att)
    weights = jax.nn.softmax(att, axis=-1)
    pooled = jnp.einsum("bt,btd->bd", weights, x.astype(jnp.float32)).astype(x.dtype)
    return _dense(pooled, params["linear3"]).astype(jnp.float32)


def _pool_last_step_bi(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
    hidden = x.shape[-1] // 2
    idx = jnp.maximum(n_wins.astype(jnp.int32) - 1, 0)
    forward_last = x[jnp.arange(x.shape[0]), idx, :hidden]
    backward_first = x[:, 0, hidden:]
    return _dense(jnp.concatenate([forward_last, backward_first], axis=-1), params["linear"]).astype(jnp.float32)


def forward_stages(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray, *, cfg: ModelConfig) -> dict[str, jnp.ndarray]:
    n_wins = n_wins.astype(jnp.int32)
    if cfg.cnn_model == "adapt":
        cnn = _cnn_adapt(params["cnn"], x, cfg)
    elif cfg.cnn_model == "standard":
        cnn = _cnn_standard(params["cnn"], x)
    else:  # pragma: no cover - guarded during config load.
        raise NotImplementedError(cfg.cnn_model)
    valid_steps = jnp.arange(cnn.shape[1], dtype=n_wins.dtype)[None, :] < n_wins[:, None]
    cnn = jnp.where(valid_steps[:, :, None], cnn, jnp.zeros_like(cnn))

    if cfg.td == "self_att":
        td = _self_attention(params["time_dependency"], cnn, n_wins)
    elif cfg.td == "lstm":
        td = _bidirectional_lstm(params["time_dependency"], cnn, n_wins)
    else:  # pragma: no cover - guarded during config load.
        raise NotImplementedError(cfg.td)

    if cfg.pool == "att":
        if cfg.is_dimensional:
            out = jnp.concatenate([_pool_att_ff(pool, td, n_wins) for pool in params["pool_layers"]], axis=1)
        else:
            out = _pool_att_ff(params["pool"], td, n_wins)
    elif cfg.pool == "last_step_bi":
        out = _pool_last_step_bi(params["pool"], td, n_wins)
    else:
        raise NotImplementedError(cfg.pool)  # pragma: no cover
    return {"cnn": cnn, "time_dependency": td, "pool": out.astype(jnp.float32)}


def forward(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray, *, cfg: ModelConfig) -> jnp.ndarray:
    return forward_stages(params, x, n_wins, cfg=cfg)["pool"]


@dataclass
class NisqaJaxModel:
    config: ModelConfig
    params: ArrayTree
    device: jax.Device
    precision: Precision = "float32"
    cache_path: Path | None = None

    def __post_init__(self) -> None:
        self.precision = _validate_precision(self.precision)
        self.params = jax.device_put(self.params, self.device)
        compute_params = _cast_tree_for_compute(self.params, _compute_dtype(self.precision))

        def strict_forward(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> jnp.ndarray:
            with jax.default_matmul_precision("float32"):
                return forward(params, x, n_wins, cfg=self.config)

        def strict_forward_stages(params: ArrayTree, x: jnp.ndarray, n_wins: jnp.ndarray) -> dict[str, jnp.ndarray]:
            with jax.default_matmul_precision("float32"):
                return forward_stages(params, x, n_wins, cfg=self.config)

        self._compute_params = compute_params
        self._forward = jax.jit(strict_forward)
        self._forward_stages = jax.jit(strict_forward_stages)

    @property
    def compute_dtype(self) -> jnp.dtype:
        return _compute_dtype(self.precision)

    def device_segments(self, x: np.ndarray, n_wins: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Fail-fast validation at the API boundary, before any device transfer.
        # Rejects malformed shapes/dtypes and zero/negative/over-long windows early with
        # precise messages, instead of producing NaN (softmax over all -inf) or late
        # broadcasting/conv errors deep inside the jitted forward pass.
        if not isinstance(x, np.ndarray) or x.ndim != 5:
            raise ValueError(
                f"x must be a 5-D ndarray [batch, steps, 1, n_mels, seg_length], got "
                f"shape {getattr(x, 'shape', None)}"
            )
        if not isinstance(n_wins, np.ndarray) or n_wins.ndim != 1:
            raise ValueError(f"n_wins must be a 1-D ndarray, got shape {getattr(n_wins, 'shape', None)}")
        if n_wins.shape[0] != x.shape[0]:
            raise ValueError(
                f"len(n_wins)={n_wins.shape[0]} must equal batch size x.shape[0]={x.shape[0]}"
            )
        if x.shape[0] == 0:
            raise ValueError("batch size must be greater than 0")
        if not np.issubdtype(n_wins.dtype, np.integer):
            raise ValueError(f"n_wins must have an integer dtype, got {n_wins.dtype}")
        feat = self.config.feature
        expected_tail = (1, feat.n_mels, feat.seg_length)
        if tuple(x.shape[2:]) != expected_tail:
            raise ValueError(
                f"x.shape[2:] must be {expected_tail} (1, n_mels, seg_length), got {tuple(x.shape[2:])}"
            )
        if int(n_wins.min()) < 1:
            raise ValueError(f"all n_wins must be >= 1, got min={int(n_wins.min())}")
        if int(n_wins.max()) > x.shape[1]:
            raise ValueError(f"all n_wins must be <= x.shape[1]={x.shape[1]}, got max={int(n_wins.max())}")

        max_steps = int(n_wins.max())
        x = x[:, :max_steps]
        x_dev = jax.device_put(jnp.asarray(x, dtype=self.compute_dtype), self.device)
        n_dev = jax.device_put(jnp.asarray(n_wins, dtype=jnp.int32), self.device)
        return x_dev, n_dev

    def predict_segments(self, x: np.ndarray, n_wins: np.ndarray) -> np.ndarray:
        x_dev, n_dev = self.device_segments(x, n_wins)
        out = self._forward(self._compute_params, x_dev, n_dev)
        return np.asarray(out.block_until_ready())

    def predict_stages(self, x: np.ndarray, n_wins: np.ndarray) -> dict[str, np.ndarray]:
        x_dev, n_dev = self.device_segments(x, n_wins)
        out = self._forward_stages(self._compute_params, x_dev, n_dev)
        return {key: np.asarray(value.block_until_ready()) for key, value in out.items()}
