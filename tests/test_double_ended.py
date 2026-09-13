from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from nisqa_jax.double_ended import (
    DoubleEndedConfig,
    align,
    alignment_scores,
    forward_double_ended,
    fuse,
)
from nisqa_jax.double_ended_checkpoint import (
    convert_double_ended_state_dict,
    expected_double_ended_parameter_shapes,
    expected_double_ended_source_keys,
    validate_double_ended_checkpoint_args,
    validate_double_ended_parameter_shapes,
)


def _linear(rng: np.random.Generator, inputs: int, outputs: int) -> dict[str, jnp.ndarray]:
    return {
        "w": jnp.asarray(rng.normal(0, 0.03, (inputs, outputs)).astype(np.float32)),
        "b": jnp.asarray(rng.normal(0, 0.03, (outputs,)).astype(np.float32)),
    }


def _norm(size: int) -> dict[str, jnp.ndarray]:
    return {"scale": jnp.ones((size,), jnp.float32), "bias": jnp.zeros((size,), jnp.float32)}


def _attention(rng: np.random.Generator, inputs: int) -> dict:
    layers = []
    for _ in range(2):
        layers.append(
            {
                "in_proj": _linear(rng, 64, 192),
                "out": _linear(rng, 64, 64),
                "linear1": _linear(rng, 64, 64),
                "linear2": _linear(rng, 64, 64),
                "norm1": _norm(64),
                "norm2": _norm(64),
            }
        )
    return {"input": _linear(rng, inputs, 64), "norm1": _norm(64), "layers": tuple(layers)}


def _full_params(seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    channels = ((1, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64))
    cnn = {
        f"conv{index}": {
            "w": jnp.asarray(rng.normal(0, 0.01, (3, 3, cin, cout)).astype(np.float32)),
            "b": jnp.asarray(rng.normal(0, 0.01, (cout,)).astype(np.float32)),
        }
        for index, (cin, cout) in enumerate(channels, 1)
    }
    return {
        "cnn": cnn,
        "time_dependency": _attention(rng, 384),
        "align": {},
        "fuse": {},
        "time_dependency_2": _attention(rng, 192),
        "pool": {
            "linear1": _linear(rng, 64, 128),
            "linear2": _linear(rng, 128, 1),
            "linear3": _linear(rng, 64, 1),
        },
    }


def _canonical_args() -> dict:
    return {
        "model": "NISQA_DE",
        "cnn_model": "adapt",
        "cnn_pool_1": [24, 7],
        "cnn_pool_2": [12, 5],
        "cnn_pool_3": [6, 3],
        "cnn_fc_out_h": None,
        "td": "self_att",
        "td_sa_d_model": 64,
        "td_sa_nhead": 1,
        "td_sa_pos_enc": False,
        "td_sa_num_layers": 2,
        "td_sa_h": 64,
        "td_2": "self_att",
        "td_2_sa_d_model": 64,
        "td_2_sa_nhead": 1,
        "td_2_sa_pos_enc": False,
        "td_2_sa_num_layers": 2,
        "td_2_sa_h": 64,
        "pool": "att",
        "pool_att_h": 128,
        "de_align": "cosine",
        "de_align_apply": "hard",
        "de_fuse": "x/y/-",
        "de_fuse_dim": None,
        "ms_n_fft": 4096,
        "ms_hop_length": 0.01,
        "ms_win_length": 0.02,
        "ms_n_mels": 48,
        "ms_seg_length": 15,
        "ms_seg_hop_length": 4,
        "ms_max_segments": 1300,
        "ms_fmax": 20000,
    }


def _fake_source_state(seed: int = 17) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    channels = ((1, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64))
    for index, (cin, cout) in enumerate(channels, 1):
        state[f"cnn.model.conv{index}.weight"] = rng.normal(0, 0.02, (cout, cin, 3, 3)).astype(np.float32)
        state[f"cnn.model.conv{index}.bias"] = rng.normal(0, 0.02, (cout,)).astype(np.float32)
        state[f"cnn.model.bn{index}.weight"] = rng.normal(1, 0.02, (cout,)).astype(np.float32)
        state[f"cnn.model.bn{index}.bias"] = rng.normal(0, 0.02, (cout,)).astype(np.float32)
        state[f"cnn.model.bn{index}.running_mean"] = rng.normal(0, 0.02, (cout,)).astype(np.float32)
        state[f"cnn.model.bn{index}.running_var"] = rng.uniform(0.5, 1.5, (cout,)).astype(np.float32)
        state[f"cnn.model.bn{index}.num_batches_tracked"] = np.asarray(3, np.int64)

    for base, inputs in (("time_dependency.model", 384), ("time_dependency_2.model", 192)):
        state[f"{base}.linear.weight"] = rng.normal(0, 0.02, (64, inputs)).astype(np.float32)
        state[f"{base}.linear.bias"] = rng.normal(0, 0.02, (64,)).astype(np.float32)
        state[f"{base}.norm1.weight"] = np.ones((64,), np.float32)
        state[f"{base}.norm1.bias"] = np.zeros((64,), np.float32)
        for index in range(2):
            prefix = f"{base}.layers.{index}"
            state[f"{prefix}.self_attn.in_proj_weight"] = rng.normal(0, 0.02, (192, 64)).astype(np.float32)
            state[f"{prefix}.self_attn.in_proj_bias"] = rng.normal(0, 0.02, (192,)).astype(np.float32)
            for name, shape in (
                ("self_attn.out_proj", (64, 64)),
                ("linear1", (64, 64)),
                ("linear2", (64, 64)),
            ):
                state[f"{prefix}.{name}.weight"] = rng.normal(0, 0.02, shape).astype(np.float32)
                state[f"{prefix}.{name}.bias"] = rng.normal(0, 0.02, (shape[0],)).astype(np.float32)
            for name in ("norm1", "norm2"):
                state[f"{prefix}.{name}.weight"] = np.ones((64,), np.float32)
                state[f"{prefix}.{name}.bias"] = np.zeros((64,), np.float32)

    for name, shape in (("linear1", (128, 64)), ("linear2", (1, 128)), ("linear3", (1, 64))):
        state[f"pool.model.{name}.weight"] = rng.normal(0, 0.02, shape).astype(np.float32)
        state[f"pool.model.{name}.bias"] = rng.normal(0, 0.02, (shape[0],)).astype(np.float32)
    return state


@pytest.mark.parametrize("method", ["dot", "cosine", "distance"])
def test_parameter_free_alignment_scores_match_numpy(method: str) -> None:
    rng = np.random.default_rng(2)
    query = rng.normal(size=(2, 3, 4)).astype(np.float32)
    y = rng.normal(size=(2, 5, 4)).astype(np.float32)
    if method == "dot":
        expected = np.einsum("bqd,byd->bqy", query, y)
    elif method == "cosine":
        expected = np.einsum("bqd,byd->bqy", query, y) / (
            np.maximum(np.linalg.norm(query, axis=-1), 1e-8)[:, :, None]
            * np.maximum(np.linalg.norm(y, axis=-1), 1e-8)[:, None, :]
        )
    else:
        expected = -np.swapaxes(np.mean(np.abs(query[:, None, :, :] - y[:, :, None, :]), axis=-1), 1, 2)
    actual = alignment_scores({}, jnp.asarray(query), jnp.asarray(y), method)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=2e-6)


def test_soft_alignment_masks_reference_tail() -> None:
    query = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    y = jnp.asarray([[[1.0, 0.0], [0.0, 1.0], [1000.0, 1000.0]]], dtype=jnp.float32)
    actual = align({}, query, y, jnp.asarray([2], jnp.int32), method="dot", apply="soft")
    expected_scores = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], np.float32)
    weights = np.exp(expected_scores - expected_scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("bqy,byd->bqd", weights, np.asarray(y)[:, :2])
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("method,width", [("x/y/-", 12), ("+/-", 8), ("x/y", 8)])
def test_fusion_contract(method: str, width: int) -> None:
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    y = x * 0.25
    result = fuse({}, x, y, method=method, fuse_dim=None)
    assert result.shape == (2, 3, width)
    if method == "x/y/-":
        np.testing.assert_array_equal(np.asarray(result[..., -4:]), np.asarray(x - y))


def test_canonical_config_and_converter_are_exhaustive() -> None:
    cfg = validate_double_ended_checkpoint_args(_canonical_args())
    assert cfg == DoubleEndedConfig()
    state = _fake_source_state()
    required, ignored = expected_double_ended_source_keys()
    assert set(state) == required | ignored
    params = convert_double_ended_state_dict(state)
    validate_double_ended_parameter_shapes(params)
    assert expected_double_ended_parameter_shapes()["time_dependency_2/input/w"] == [192, 64]


def test_converter_rejects_missing_and_unexpected_source_tensor() -> None:
    state = _fake_source_state()
    state.pop("time_dependency_2.model.linear.weight")
    with pytest.raises(ValueError, match="missing required keys"):
        convert_double_ended_state_dict(state)
    state = _fake_source_state()
    state["oracle.output"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="unexpected keys"):
        convert_double_ended_state_dict(state)


def test_config_rejects_architecture_drift() -> None:
    args = _canonical_args()
    args["td_2_sa_nhead"] = 2
    with pytest.raises(NotImplementedError, match="td_2_sa_nhead"):
        validate_double_ended_checkpoint_args(args)


def test_full_default_graph_is_finite_and_differentiable() -> None:
    params = _full_params()
    rng = np.random.default_rng(23)
    x = rng.normal(0, 0.2, (1, 3, 2, 48, 15)).astype(np.float32)
    n_wins = jnp.asarray([[3, 2]], dtype=jnp.int32)
    result = forward_double_ended(params, jnp.asarray(x), n_wins, cfg=DoubleEndedConfig())
    assert result.shape == (1, 1)
    assert bool(jnp.all(jnp.isfinite(result)))

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        scaled = dict(params)
        scaled["pool"] = dict(params["pool"])
        scaled["pool"]["linear3"] = dict(params["pool"]["linear3"])
        scaled["pool"]["linear3"]["w"] = params["pool"]["linear3"]["w"] * scale
        return jnp.sum(forward_double_ended(scaled, jnp.asarray(x), n_wins, cfg=DoubleEndedConfig()))

    grad = jax.grad(loss)(jnp.asarray(1.0, jnp.float32))
    assert bool(jnp.isfinite(grad))
    assert float(jnp.abs(grad)) > 0
