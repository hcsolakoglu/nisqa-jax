from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import checkpoint as _checkpoint
from .double_ended import DoubleEndedConfig


_DE_PROFILE: dict[str, Any] = {
    "model": "NISQA_DE",
    "cnn_model": "adapt",
    "cnn_pool_1": (24, 7),
    "cnn_pool_2": (12, 5),
    "cnn_pool_3": (6, 3),
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


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def validate_double_ended_checkpoint_args(args: dict[str, Any]) -> DoubleEndedConfig:
    """Fail closed unless source args match canonical upstream NISQA_DE graph."""
    for key, expected in _DE_PROFILE.items():
        if key not in args:
            raise NotImplementedError(f"NISQA_DE checkpoint is missing required argument {key!r}")
        actual = _normalize(args[key])
        if actual != expected:
            raise NotImplementedError(
                f"Unsupported NISQA_DE {key}={args[key]!r}; canonical upstream profile requires {expected!r}"
            )
    return DoubleEndedConfig()


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"model tensor must be floating point, got {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("model tensor contains non-finite values")
    return np.array(array, copy=True)


def _linear(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {"w": _array(sd[f"{prefix}.weight"]).T, "b": _array(sd[f"{prefix}.bias"])}


def _conv(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {
        "w": np.transpose(_array(sd[f"{prefix}.weight"]), (2, 3, 1, 0)),
        "b": _array(sd[f"{prefix}.bias"]),
    }


def _fold_conv_bn(sd: dict[str, Any], index: int) -> dict[str, np.ndarray]:
    conv = _conv(sd, f"cnn.model.conv{index}")
    prefix = f"cnn.model.bn{index}"
    scale = _array(sd[f"{prefix}.weight"])
    bias = _array(sd[f"{prefix}.bias"])
    mean = _array(sd[f"{prefix}.running_mean"])
    var = _array(sd[f"{prefix}.running_var"])
    factor = scale / np.sqrt(var + np.float32(1e-5))
    return {
        "w": (conv["w"] * factor[None, None, None, :]).astype(np.float32),
        "b": ((conv["b"] - mean) * factor + bias).astype(np.float32),
    }


def _norm(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {"scale": _array(sd[f"{prefix}.weight"]), "bias": _array(sd[f"{prefix}.bias"])}


def _attention_layer(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "in_proj": {
            "w": _array(sd[f"{prefix}.self_attn.in_proj_weight"]).T,
            "b": _array(sd[f"{prefix}.self_attn.in_proj_bias"]),
        },
        "out": _linear(sd, f"{prefix}.self_attn.out_proj"),
        "linear1": _linear(sd, f"{prefix}.linear1"),
        "linear2": _linear(sd, f"{prefix}.linear2"),
        "norm1": _norm(sd, f"{prefix}.norm1"),
        "norm2": _norm(sd, f"{prefix}.norm2"),
    }


def _attention(sd: dict[str, Any], base: str, *, layers: int = 2) -> dict[str, Any]:
    return {
        "input": _linear(sd, f"{base}.linear"),
        "norm1": _norm(sd, f"{base}.norm1"),
        "layers": tuple(_attention_layer(sd, f"{base}.layers.{index}") for index in range(layers)),
    }


def expected_double_ended_source_keys() -> tuple[set[str], set[str]]:
    required: set[str] = set()
    ignored: set[str] = set()
    for index in range(1, 7):
        for suffix in ("weight", "bias"):
            required.add(f"cnn.model.conv{index}.{suffix}")
            required.add(f"cnn.model.bn{index}.{suffix}")
        required.add(f"cnn.model.bn{index}.running_mean")
        required.add(f"cnn.model.bn{index}.running_var")
        ignored.add(f"cnn.model.bn{index}.num_batches_tracked")

    for base in ("time_dependency.model", "time_dependency_2.model"):
        required.update(
            {
                f"{base}.linear.weight",
                f"{base}.linear.bias",
                f"{base}.norm1.weight",
                f"{base}.norm1.bias",
            }
        )
        for index in range(2):
            prefix = f"{base}.layers.{index}"
            required.add(f"{prefix}.self_attn.in_proj_weight")
            required.add(f"{prefix}.self_attn.in_proj_bias")
            for layer in ("self_attn.out_proj", "linear1", "linear2", "norm1", "norm2"):
                required.add(f"{prefix}.{layer}.weight")
                required.add(f"{prefix}.{layer}.bias")

    for layer in ("linear1", "linear2", "linear3"):
        required.add(f"pool.model.{layer}.weight")
        required.add(f"pool.model.{layer}.bias")
    return required, ignored


def _validate_state_dict(sd: Any) -> None:
    if not hasattr(sd, "keys"):
        raise ValueError("NISQA_DE model_state_dict must be a tensor mapping")
    required, ignored = expected_double_ended_source_keys()
    actual = set(sd.keys())
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - ignored)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing required keys: {missing}")
        if unexpected:
            detail.append(f"unexpected keys: {unexpected}")
        raise ValueError("NISQA_DE source state is not exhaustive: " + "; ".join(detail))


def _flatten(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        out: dict[str, np.ndarray] = {}
        for key, value in tree.items():
            out.update(_flatten(value, f"{prefix}{key}/"))
        return out
    if isinstance(tree, tuple | list):
        out = {}
        for index, value in enumerate(tree):
            out.update(_flatten(value, f"{prefix}{index}/"))
        return out
    return {prefix[:-1]: np.asarray(tree)}


def convert_double_ended_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert every tensor in canonical upstream NISQA_DE state into JAX layout."""
    _validate_state_dict(sd)
    params: dict[str, Any] = {
        "cnn": {f"conv{index}": _fold_conv_bn(sd, index) for index in range(1, 7)},
        "time_dependency": _attention(sd, "time_dependency.model"),
        "time_dependency_2": _attention(sd, "time_dependency_2.model"),
        "pool": {
            "linear1": _linear(sd, "pool.model.linear1"),
            "linear2": _linear(sd, "pool.model.linear2"),
            "linear3": _linear(sd, "pool.model.linear3"),
        },
        "align": {},
        "fuse": {},
    }
    return params


def expected_double_ended_parameter_shapes() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}

    def linear(prefix: str, inputs: int, outputs: int) -> None:
        shapes[f"{prefix}/w"] = [inputs, outputs]
        shapes[f"{prefix}/b"] = [outputs]

    channels = ((1, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64))
    for index, (inputs, outputs) in enumerate(channels, 1):
        shapes[f"cnn/conv{index}/w"] = [3, 3, inputs, outputs]
        shapes[f"cnn/conv{index}/b"] = [outputs]

    for name, inputs in (("time_dependency", 384), ("time_dependency_2", 192)):
        linear(f"{name}/input", inputs, 64)
        shapes[f"{name}/norm1/scale"] = [64]
        shapes[f"{name}/norm1/bias"] = [64]
        for index in range(2):
            prefix = f"{name}/layers/{index}"
            linear(f"{prefix}/in_proj", 64, 192)
            linear(f"{prefix}/out", 64, 64)
            linear(f"{prefix}/linear1", 64, 64)
            linear(f"{prefix}/linear2", 64, 64)
            for norm in ("norm1", "norm2"):
                shapes[f"{prefix}/{norm}/scale"] = [64]
                shapes[f"{prefix}/{norm}/bias"] = [64]
    linear("pool/linear1", 64, 128)
    linear("pool/linear2", 128, 1)
    linear("pool/linear3", 64, 1)
    return dict(sorted(shapes.items()))


def validate_double_ended_parameter_shapes(params: dict[str, Any]) -> None:
    actual = {name: list(value.shape) for name, value in sorted(_flatten(params).items())}
    expected = expected_double_ended_parameter_shapes()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        raise ValueError(
            "converted NISQA_DE parameter contract mismatch: "
            f"missing={missing}, extra={extra}, wrong={wrong}"
        )


def convert_double_ended_checkpoint(checkpoint_path: str | Path) -> tuple[DoubleEndedConfig, dict[str, Any], str]:
    """Safely convert canonical upstream NISQA_DE .tar checkpoint in memory."""
    path = Path(checkpoint_path).expanduser().resolve()
    torch = _checkpoint._torch()
    checkpoint = _checkpoint._load_torch_checkpoint(torch, path)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise ValueError("NISQA_DE checkpoint must contain an args dict")
    if "model_state_dict" not in checkpoint:
        raise ValueError("NISQA_DE checkpoint is missing model_state_dict")
    cfg = validate_double_ended_checkpoint_args(checkpoint["args"])
    params = convert_double_ended_state_dict(checkpoint["model_state_dict"])
    validate_double_ended_parameter_shapes(params)
    return cfg, params, _checkpoint._sha256(path)
