from __future__ import annotations

import hashlib
import json
import threading
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from .config import (
    FeatureConfig,
    ModelConfig,
    config_from_checkpoint_args,
    derive_output_names,
    validate_model_config,
)
from .model import NisqaJaxModel, Precision, _validate_precision

CONVERSION_VERSION = 4

# Persistent compilation-cache configuration is process-global. JAX 0.4.x keeps
# an internal cache singleton after first use, so conflicting reconfiguration is
# rejected instead of claiming a later directory has become active.
# `is_initialized`/`initialize_cache` are deprecated in JAX 0.4.30 and removed in
# newer releases; `jax_compilation_cache_dir` is the stable, version-tolerant knob.
_persistent_cache_path: Path | None = None
_persistent_cache_lock = threading.Lock()


def _configure_persistent_cache(cache_dir: str | Path) -> None:
    """Idempotently enable JAX's persistent compilation cache.

    JAX defaults ``jax_persistent_cache_min_compile_time_secs=1.0``, which silently
    drops the sub-second compiles this repo produces, leaving the cache empty. We
    lower the threshold to 0 and disable the entry-size floor so every compiled
    program is persisted. Must run before the first compilation of the target
    program (import order is irrelevant). Repeated use of the same path is
    idempotent; a different explicit path is rejected because JAX may continue
    writing to the first directory after its internal cache is initialized.
    """
    global _persistent_cache_path
    path = Path(cache_dir).expanduser().resolve()
    with _persistent_cache_lock:
        active_raw = getattr(jax.config, "jax_compilation_cache_dir", None)
        active = Path(active_raw).expanduser().resolve() if active_raw else _persistent_cache_path
        if active is not None:
            # The directory may have been supplied externally (for example via
            # JAX_COMPILATION_CACHE_DIR), so still apply this project's policy
            # for short/small compilations instead of assuming it was set.
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            if active != path:
                raise ValueError(
                    "JAX persistent compilation cache is process-global and is already "
                    f"configured for {active}; refusing conflicting directory {path}. "
                    "Reuse the first cache directory or start a new process."
                )
            _persistent_cache_path = path
            return
        path.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(path))
        # Persist short (<1s) compiles instead of filtering them out.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # No minimum entry size: cache every program regardless of size.
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        _persistent_cache_path = path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Fields excluded from the canonical metadata checksum: the two file-reference
# hashes are not part of the semantic metadata (npz_sha256 pins the .npz bytes;
# metadata_sha256 is this checksum itself — including it would be circular).
_CANONICAL_HASH_EXCLUDED_FIELDS: frozenset[str] = frozenset({"npz_sha256", "metadata_sha256"})


def canonical_metadata_checksum(metadata: dict[str, Any]) -> str:
    """Deterministic SHA-256 of an artifact's semantic metadata payload.

    The canonical form is ``json.dumps(metadata, sort_keys=True, separators=(",", ":"))``
    over a copy of ``metadata`` with the file-reference hash fields
    (``npz_sha256``, ``metadata_sha256``) removed. This is:

      * **Stable** — independent of pretty-printing/whitespace, so re-serializing
        the same logical metadata yields the same digest.
      * **Non-circular** — the checksum never covers itself (``metadata_sha256``
        is stripped before hashing), so embedding it in the JSON is safe and
        verifiable by recomputing over the stripped payload.
      * **Externally verifiable** — a release ``CHECKSUMS`` manifest can list the
        raw ``.json`` file SHA-256 (catches any byte change) AND/OR this canonical
        digest (catches semantic changes regardless of formatting). The loader
        itself cross-checks ``npz_sha256`` (JSON) against the ``.npz`` file for
        internal JSON<->NPZ consistency.

    Older artifacts (pre-``metadata_sha256``) lack this field; the loader warns
    (not fails) for backward compatibility, matching the ``npz_sha256`` policy.
    """
    payload = {k: v for k, v in metadata.items() if k not in _CANONICAL_HASH_EXCLUDED_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError(
            "Loading original .tar checkpoints requires the optional PyTorch dependency. "
            "Use a pre-converted JAX .npz artifact for standalone inference."
        ) from exc
    return torch


def _torch_version_lt(version: str, major: int, minor: int) -> bool:
    """Return True if ``version`` is strictly older than ``major.minor``.

    Parses the leading numeric major/minor numerically (not lexicographically)
    so suffixes like ``+cu128``, ``.dev0``, ``.rc1`` and multi-digit minor
    numbers are handled correctly. Examples::

        _torch_version_lt("1.12.1+cpu", 1, 13) -> True
        _torch_version_lt("1.13.0", 1, 13)     -> False
        _torch_version_lt("2.11.0+cu128", 1, 13) -> False
        _torch_version_lt("1.9.0", 1, 13)      -> True
        _torch_version_lt("1.10.2.dev0", 1, 13) -> True

    A non-parseable version is treated as 0.0 (safely "old") so the caller's
    "too old" branch fires rather than silently passing an unsupported torch.
    """
    # Strip any local/build suffix after '+' and pre-release suffix after a
    # non-numeric component, then take the leading numeric dotted parts.
    base = version.split("+", 1)[0]
    parts: list[int] = []
    for part in base.split("."):
        # Stop at the first non-numeric component (e.g. "0" in "1.13.0.dev0"
        # is numeric; "dev0" is not -> we keep [1,13,0]).
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    ver_major = parts[0] if len(parts) > 0 else 0
    ver_minor = parts[1] if len(parts) > 1 else 0
    return (ver_major, ver_minor) < (major, minor)


def _load_torch_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    """Load a NISQA .tar checkpoint with ``weights_only=True`` (safe unpickle).

    The shipped NISQA .tar checkpoints contain only an ``args`` dict of plain
    Python scalars/strings and a ``model_state_dict`` of tensors, all of which
    are on PyTorch's ``weights_only=True`` safelist. Forcing ``weights_only=True``
    prevents arbitrary-code-execution via a maliciously crafted checkpoint
    (pickle deserialization of untrusted data).

    We never silently fall back to unsafe pickle (``weights_only=False``). If the
    installed torch is too old to support ``weights_only`` (pre-1.13), we raise an
    actionable error rather than downgrading to unsafe loading. If a checkpoint
    genuinely requires unsafe loading (it carries a non-safelist type), that is a
    strong signal it is not a stock NISQA checkpoint and conversion is refused.
    """
    # `weights_only` was added in torch 1.13 and became the default in 2.6. We
    # pass it explicitly to be explicit about the safety contract on every torch
    # version that supports it.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Distinguish "torch too old for weights_only kwarg" from a TypeError
        # raised inside the unpickler. The kwarg-missing case is detected by
        # numerically parsing the torch major.minor version: <1.13 lacks
        # weights_only entirely. Parsing numerically (not lexicographic string
        # compare) is robust to suffixes like "2.11.0+cu128", "1.13.0.dev0",
        # "1.12.1+cpu", and multi-digit minor numbers (e.g. "1.13" vs "1.9").
        version = getattr(torch, "__version__", "0")
        if _torch_version_lt(version, 1, 13):
            raise RuntimeError(
                f"Installed torch ({version}) does not support weights_only=True "
                "(requires torch>=1.13). Upgrade torch before converting checkpoints, "
                "or use a pre-converted JAX .npz artifact. Unsafe pickle loading of "
                "untrusted checkpoints is refused by design."
            ) from None
        raise
    except OSError:
        # Filesystem-level errors (missing file, permission, truncation) propagate
        # unchanged — they are not safety-policy violations.
        raise
    except Exception as exc:
        # weights_only=True rejected the checkpoint payload (non-safelist type,
        # e.g. a custom class). This is expected for non-stock checkpoints; do
        # NOT retry with weights_only=False.
        raise RuntimeError(
            f"Checkpoint {path.name} could not be loaded with weights_only=True "
            f"(safe unpickle). It may carry non-safelist Python objects, which is not "
            f"expected for a stock NISQA checkpoint. Unsafe pickle loading is refused "
            f"by design. Use a pre-converted JAX .npz artifact. Underlying error: {exc}"
        ) from exc


def _np(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _linear(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {"w": _np(sd[f"{prefix}.weight"]).T, "b": _np(sd[f"{prefix}.bias"])}


def _conv(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {"w": np.transpose(_np(sd[f"{prefix}.weight"]), (2, 3, 1, 0)), "b": _np(sd[f"{prefix}.bias"])}


def _bn(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {
        "scale": _np(sd[f"{prefix}.weight"])[None, None, None, :],
        "bias": _np(sd[f"{prefix}.bias"])[None, None, None, :],
        "mean": _np(sd[f"{prefix}.running_mean"])[None, None, None, :],
        "var": _np(sd[f"{prefix}.running_var"])[None, None, None, :],
    }


def _fold_conv_bn(conv: dict[str, np.ndarray], bn: dict[str, np.ndarray], eps: float = 1e-5) -> dict[str, np.ndarray]:
    scale = bn["scale"].reshape(-1)
    bias = bn["bias"].reshape(-1)
    mean = bn["mean"].reshape(-1)
    var = bn["var"].reshape(-1)
    factor = scale / np.sqrt(var + eps)
    return {
        "w": (conv["w"] * factor[None, None, None, :]).astype(np.float32),
        "b": ((conv["b"] - mean) * factor + bias).astype(np.float32),
    }


def _ln(sd: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {"scale": _np(sd[f"{prefix}.weight"]), "bias": _np(sd[f"{prefix}.bias"])}


def _cnn(sd: dict[str, Any], cfg: ModelConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for i in range(1, 7):
        conv = _conv(sd, f"cnn.model.conv{i}")
        bn = _bn(sd, f"cnn.model.bn{i}")
        params[f"conv{i}"] = _fold_conv_bn(conv, bn)
    if cfg.cnn_model == "standard":
        params["fc_out"] = _linear(sd, "cnn.model.fc_out")
    return params


def _mha_layer(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    in_w = _np(sd[f"{prefix}.self_attn.in_proj_weight"])
    in_b = _np(sd[f"{prefix}.self_attn.in_proj_bias"])
    return {
        "in_proj": {"w": in_w.T, "b": in_b},
        "out": _linear(sd, f"{prefix}.self_attn.out_proj"),
        "linear1": _linear(sd, f"{prefix}.linear1"),
        "linear2": _linear(sd, f"{prefix}.linear2"),
        "norm1": _ln(sd, f"{prefix}.norm1"),
        "norm2": _ln(sd, f"{prefix}.norm2"),
    }


def _self_attention(sd: dict[str, Any], cfg: ModelConfig) -> dict[str, Any]:
    base = "time_dependency.model"
    return {
        "input": _linear(sd, f"{base}.linear"),
        "norm1": _ln(sd, f"{base}.norm1"),
        "layers": tuple(_mha_layer(sd, f"{base}.layers.{i}") for i in range(int(cfg.td_sa_num_layers or 0))),
    }


def _lstm_direction(sd: dict[str, Any], suffix: str) -> dict[str, np.ndarray]:
    base = "time_dependency.model.lstm"
    return {
        "w_ih": _np(sd[f"{base}.weight_ih_l0{suffix}"]).T,
        "w_hh": _np(sd[f"{base}.weight_hh_l0{suffix}"]).T,
        "b_ih": _np(sd[f"{base}.bias_ih_l0{suffix}"]),
        "b_hh": _np(sd[f"{base}.bias_hh_l0{suffix}"]),
    }


def _lstm(sd: dict[str, Any]) -> dict[str, Any]:
    return {"forward": _lstm_direction(sd, ""), "reverse": _lstm_direction(sd, "_reverse")}


def _pool_att(sd: dict[str, Any], prefix: str) -> dict[str, dict[str, np.ndarray]]:
    # Each entry is a _linear() sub-dict {"w": ..., "b": ...}, not a bare array.
    return {
        "linear1": _linear(sd, f"{prefix}.model.linear1"),
        "linear2": _linear(sd, f"{prefix}.model.linear2"),
        "linear3": _linear(sd, f"{prefix}.model.linear3"),
    }


def _pool_last_step_bi(sd: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {"linear": _linear(sd, "pool.model.linear")}


def _flatten(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        out: dict[str, np.ndarray] = {}
        for key, value in tree.items():
            out.update(_flatten(value, f"{prefix}{key}/"))
        return out
    if isinstance(tree, tuple | list):
        out = {}
        for idx, value in enumerate(tree):
            out.update(_flatten(value, f"{prefix}{idx}/"))
        return out
    return {prefix[:-1]: np.asarray(tree)}


def _shape_manifest(params: dict[str, Any]) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in sorted(_flatten(params).items())}


def _expected_artifact_manifest(cfg: ModelConfig) -> dict[str, list[int]]:
    """Return the exact parameter contract implemented by the shipped kernels."""
    shapes: dict[str, list[int]] = {}

    def linear(prefix: str, in_features: int, out_features: int) -> None:
        shapes[f"{prefix}/w"] = [in_features, out_features]
        shapes[f"{prefix}/b"] = [out_features]

    def conv(prefix: str, in_channels: int, out_channels: int) -> None:
        shapes[f"{prefix}/w"] = [3, 3, in_channels, out_channels]
        shapes[f"{prefix}/b"] = [out_channels]

    channels = ((1, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64))
    for index, (in_channels, out_channels) in enumerate(channels, start=1):
        conv(f"cnn/conv{index}", in_channels, out_channels)

    if cfg.cnn_model == "adapt":
        linear("time_dependency/input", 384, 64)
        shapes["time_dependency/norm1/scale"] = [64]
        shapes["time_dependency/norm1/bias"] = [64]
        for index in range(2):
            prefix = f"time_dependency/layers/{index}"
            linear(f"{prefix}/in_proj", 64, 192)
            linear(f"{prefix}/out", 64, 64)
            linear(f"{prefix}/linear1", 64, 64)
            linear(f"{prefix}/linear2", 64, 64)
            for norm in ("norm1", "norm2"):
                shapes[f"{prefix}/{norm}/scale"] = [64]
                shapes[f"{prefix}/{norm}/bias"] = [64]

        pool_prefixes = [f"pool_layers/{index}" for index in range(5)] if cfg.is_dimensional else ["pool"]
        for prefix in pool_prefixes:
            linear(f"{prefix}/linear1", 64, 128)
            linear(f"{prefix}/linear2", 128, 1)
            linear(f"{prefix}/linear3", 64, 1)
    else:
        linear("cnn/fc_out", 768, 20)
        for direction in ("forward", "reverse"):
            shapes[f"time_dependency/{direction}/w_ih"] = [20, 512]
            shapes[f"time_dependency/{direction}/w_hh"] = [128, 512]
            shapes[f"time_dependency/{direction}/b_ih"] = [512]
            shapes[f"time_dependency/{direction}/b_hh"] = [512]
        linear("pool/linear", 256, 1)
    return dict(sorted(shapes.items()))


def _expected_source_state_keys(cfg: ModelConfig) -> tuple[set[str], set[str]]:
    """Return required source tensors and the only ignorable state entries."""
    required: set[str] = set()
    ignored: set[str] = set()
    for index in range(1, 7):
        required.update(
            {
                f"cnn.model.conv{index}.weight",
                f"cnn.model.conv{index}.bias",
                f"cnn.model.bn{index}.weight",
                f"cnn.model.bn{index}.bias",
                f"cnn.model.bn{index}.running_mean",
                f"cnn.model.bn{index}.running_var",
            }
        )
        ignored.add(f"cnn.model.bn{index}.num_batches_tracked")
    if cfg.cnn_model == "standard":
        required.update({"cnn.model.fc_out.weight", "cnn.model.fc_out.bias"})

    if cfg.td == "self_att":
        base = "time_dependency.model"
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
            required.update(
                {
                    f"{prefix}.self_attn.in_proj_weight",
                    f"{prefix}.self_attn.in_proj_bias",
                }
            )
            for layer_name in (
                "self_attn.out_proj",
                "linear1",
                "linear2",
                "norm1",
                "norm2",
            ):
                required.update({f"{prefix}.{layer_name}.weight", f"{prefix}.{layer_name}.bias"})
    else:
        base = "time_dependency.model.lstm"
        for suffix in ("", "_reverse"):
            required.update(
                {
                    f"{base}.weight_ih_l0{suffix}",
                    f"{base}.weight_hh_l0{suffix}",
                    f"{base}.bias_ih_l0{suffix}",
                    f"{base}.bias_hh_l0{suffix}",
                }
            )

    if cfg.pool == "att":
        prefixes = [f"pool_layers.{index}" for index in range(5)] if cfg.is_dimensional else ["pool"]
        for prefix in prefixes:
            for layer in ("linear1", "linear2", "linear3"):
                required.update(
                    {
                        f"{prefix}.model.{layer}.weight",
                        f"{prefix}.model.{layer}.bias",
                    }
                )
    else:
        required.update({"pool.model.linear.weight", "pool.model.linear.bias"})
    return required, ignored


def _validate_source_state_dict(sd: Any, cfg: ModelConfig) -> None:
    if not hasattr(sd, "keys"):
        raise ValueError("Checkpoint model_state_dict must be a mapping of tensor names to tensors")
    required, ignored = _expected_source_state_keys(cfg)
    actual = set(sd.keys())
    if not all(isinstance(name, str) for name in actual):
        raise ValueError("Checkpoint model_state_dict keys must all be strings")
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - ignored)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise ValueError(
            "Checkpoint state_dict does not match the validated shipped architecture "
            f"({'; '.join(details)}). Only the six expected BatchNorm "
            "num_batches_tracked entries may be ignored."
        )


def _validate_parameter_manifest(cfg: ModelConfig, manifest: dict[str, list[int]], *, context: str) -> None:
    expected = _expected_artifact_manifest(cfg)
    if manifest != expected:
        missing = sorted(set(expected) - set(manifest))
        extra = sorted(set(manifest) - set(expected))
        wrong = sorted(
            name
            for name in set(expected).intersection(manifest)
            if manifest[name] != expected[name]
        )
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"unexpected keys: {extra}")
        if wrong:
            details.append(
                "wrong shapes: "
                + repr({name: {"actual": manifest[name], "expected": expected[name]} for name in wrong})
            )
        raise ValueError(
            f"{context} does not match the exact shipped JAX parameter contract "
            f"({'; '.join(details)})."
        )


def _convert_state_dict(sd: dict[str, Any], cfg: ModelConfig) -> dict[str, Any]:
    params: dict[str, Any] = {"cnn": _cnn(sd, cfg)}
    if cfg.td == "self_att":
        params["time_dependency"] = _self_attention(sd, cfg)
    elif cfg.td == "lstm":
        params["time_dependency"] = _lstm(sd)
    else:
        raise NotImplementedError(cfg.td)

    if cfg.pool == "att":
        if cfg.is_dimensional:
            params["pool_layers"] = tuple(_pool_att(sd, f"pool_layers.{i}") for i in range(5))
        else:
            params["pool"] = _pool_att(sd, "pool")
    elif cfg.pool == "last_step_bi":
        params["pool"] = _pool_last_step_bi(sd)
    else:
        raise NotImplementedError(cfg.pool)
    return params


def _config_metadata(cfg: ModelConfig) -> dict[str, Any]:
    data = asdict(cfg)
    # Store ONLY the source filename, not the absolute build-machine path, so
    # artifacts are portable and do not leak the converter's filesystem layout.
    # `cache_key`/`_csv_row` use `source_path.stem`, which is identical for a
    # bare filename; `_config_from_metadata` rewraps it in Path() unchanged.
    data["source_path"] = cfg.source_path.name
    return data


def _config_from_metadata(data: dict[str, Any]) -> ModelConfig:
    feature = FeatureConfig(**data["feature"])
    # Backward-compatible fallback: pre-source-name artifacts (conversion
    # version < this field's introduction) lack 'source_name'. Default to None
    # so older shipped JSON still loads; the CSV lane treats None as "unknown".
    raw_name = data.get("source_name")
    source_name = raw_name if isinstance(raw_name, str) and raw_name else None
    return ModelConfig(
        source_path=Path(data["source_path"]),
        source_sha256=data["source_sha256"],
        model_name=data["model_name"],
        source_name=source_name,
        cnn_model=data["cnn_model"],
        td=data["td"],
        td_2=data["td_2"],
        pool=data["pool"],
        output_names=tuple(data["output_names"]),
        feature=feature,
        cnn_pool_1=tuple(data["cnn_pool_1"]) if data["cnn_pool_1"] is not None else None,
        cnn_pool_2=tuple(data["cnn_pool_2"]) if data["cnn_pool_2"] is not None else None,
        cnn_pool_3=tuple(data["cnn_pool_3"]) if data["cnn_pool_3"] is not None else None,
        td_sa_d_model=data["td_sa_d_model"],
        td_sa_nhead=data["td_sa_nhead"],
        td_sa_num_layers=data["td_sa_num_layers"],
        td_sa_h=data["td_sa_h"],
        td_lstm_h=data["td_lstm_h"],
        td_lstm_bidirectional=data["td_lstm_bidirectional"],
    )


def _metadata_for_artifact(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    if path.suffix == ".npz":
        return path.with_suffix(".json")
    raise ValueError(f"Expected .npz or .json converted artifact, got {path}")


def _npz_for_artifact(path: Path) -> Path:
    if path.suffix == ".npz":
        return path
    if path.suffix == ".json":
        return path.with_suffix(".npz")
    raise ValueError(f"Expected .npz or .json converted artifact, got {path}")


def _insert_flat(root: dict[str, Any], name: str, value: np.ndarray) -> None:
    node = root
    parts = name.split("/")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _tuplify_numeric_dicts(tree: Any) -> Any:
    if isinstance(tree, dict):
        if tree and all(key.isdigit() for key in tree):
            return tuple(_tuplify_numeric_dicts(tree[key]) for key in sorted(tree, key=int))
        return {key: _tuplify_numeric_dicts(value) for key, value in tree.items()}
    return tree


def convert_checkpoint(
    checkpoint_path: str | Path, *, cache_dir: str | Path | None = None
) -> tuple[ModelConfig, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    digest = _sha256(path)
    torch = _torch()
    checkpoint = _load_torch_checkpoint(torch, path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dict containing 'args' and 'model_state_dict'")
    if not isinstance(checkpoint.get("args"), dict):
        raise ValueError("Checkpoint 'args' must be a dict")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing required 'model_state_dict'")
    cfg = config_from_checkpoint_args(checkpoint["args"], path, digest)
    state_dict = checkpoint["model_state_dict"]
    _validate_source_state_dict(state_dict, cfg)
    params = _convert_state_dict(state_dict, cfg)
    _validate_parameter_manifest(cfg, _shape_manifest(params), context="Converted source checkpoint")

    if cache_dir is not None:
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        stem = f"{cfg.cache_key}.v{CONVERSION_VERSION}"
        flat = _flatten(params)
        npz_path = cache / f"{stem}.npz"
        # numpy's `savez` stubs do not model `**dict[str, ndarray]` unpacking and
        # mis-resolve the kwargs against an unrelated overload (arg-type on a
        # `bool` param); the runtime call is correct. Justified narrow ignore.
        np.savez(npz_path, **flat)  # type: ignore[arg-type]
        metadata = {
            "conversion_version": CONVERSION_VERSION,
            "source_path": path.name,
            "source_sha256": digest,
            # Integrity hash of the .npz bytes; verified on load to detect
            # tampering/corruption of the weight artifact independent of the
            # source checkpoint hash. Older artifacts lack this field and the
            # loader warns (not fails) for backward compatibility.
            "npz_sha256": _sha256(npz_path),
            # Canonical checksum of the metadata payload (everything except the
            # two file-reference hashes: npz_sha256 and metadata_sha256 itself).
            # Externally verifiable: a release CHECKSUMS manifest can list this
            # value for the .json sidecar so downstream consumers can confirm the
            # metadata content is byte-for-byte the converted canonical form,
            # independent of pretty-printing whitespace. See
            # ``canonical_metadata_checksum`` for the exact algorithm.
            "metadata_sha256": canonical_metadata_checksum(
                {
                    "conversion_version": CONVERSION_VERSION,
                    "source_path": path.name,
                    "source_sha256": digest,
                    "model_name": cfg.model_name,
                    "source_name": cfg.source_name,
                    "output_names": list(cfg.output_names),
                    "model_config": _config_metadata(cfg),
                    "shape_manifest": _shape_manifest(params),
                }
            ),
            "model_name": cfg.model_name,
            # Stable original training-run identity from args['name']; consumed
            # by the CSV lane. None for pre-source-name artifacts.
            "source_name": cfg.source_name,
            "output_names": cfg.output_names,
            "model_config": _config_metadata(cfg),
            "shape_manifest": _shape_manifest(params),
        }
        (cache / f"{stem}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return cfg, params


def _validate_flat_parameter_names(names: list[str]) -> None:
    """Validate flat names can be rebuilt unambiguously into dicts/tuples."""
    leaf = object()
    root: dict[str | object, Any] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Artifact parameter name must be a non-empty string, got {name!r}")
        parts = name.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise ValueError(f"Artifact parameter name has an invalid path component: {name!r}")
        node = root
        for part in parts:
            if leaf in node:
                raise ValueError(f"Artifact parameter paths collide at {name!r}")
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Artifact parameter paths collide at {name!r}")
            node = child
        if node:
            raise ValueError(f"Artifact parameter paths collide at {name!r}")
        node[leaf] = True

    def validate_node(node: dict[str | object, Any], path: str) -> None:
        keys = [key for key in node if key is not leaf]
        if leaf in node and keys:
            raise ValueError(f"Artifact parameter paths collide at {path!r}")
        if not keys:
            return
        string_keys = [str(key) for key in keys]
        numeric = [key.isdigit() for key in string_keys]
        if any(numeric):
            if not all(numeric):
                raise ValueError(
                    f"Artifact tuple node {path!r} mixes numeric and named children: {sorted(string_keys)}"
                )
            expected = [str(index) for index in range(len(string_keys))]
            if sorted(string_keys, key=int) != expected:
                raise ValueError(
                    f"Artifact tuple node {path!r} must use contiguous canonical indices "
                    f"{expected}, got {sorted(string_keys, key=int)}"
                )
        for key in keys:
            child_path = f"{path}/{key}" if path else str(key)
            validate_node(node[key], child_path)

    validate_node(root, "")


def _verify_artifact_integrity(
    npz_path: Path,
    metadata: dict[str, Any],
    cfg: ModelConfig,
) -> dict[str, np.ndarray]:
    """Validate the .npz weight artifact against the JSON ``shape_manifest``.

    Checks (all hard-fail on mismatch — a divergence means the npz and its
    metadata sidecar are out of sync, which corrupts inference):
      1. Exact tensor-key equality with ``shape_manifest`` (no missing/extra).
      2. Each tensor's shape matches the manifest entry.
      3. Each tensor's dtype is float32-compatible (floating-point).
      4. Each tensor is finite (no NaN/Inf) — non-finite weights corrupt every
         forward pass and usually signal conversion corruption or bit-rot.
      5. The npz SHA256 matches ``npz_sha256`` when present (older artifacts
         lack this field -> warn, do not fail, for backward compatibility).
    """
    manifest = metadata.get("shape_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Artifact metadata for {npz_path.name} is missing a 'shape_manifest'; "
            "the JSON sidecar is malformed. Re-convert the source checkpoint."
        )
    if not all(isinstance(name, str) for name in manifest):
        raise ValueError("Artifact shape_manifest keys must all be strings")
    for name, shape in manifest.items():
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in shape
        ):
            raise ValueError(
                f"Artifact shape_manifest entry {name!r} must be a list of positive integer dimensions, "
                f"got {shape!r}"
            )
    _validate_flat_parameter_names(list(manifest))
    # np.load as a context manager so the underlying file handle is closed even
    # on early validation errors (previously leaked on every load).
    with np.load(npz_path, allow_pickle=False) as loaded:
        npz_keys = sorted(loaded.files)
        manifest_keys = sorted(manifest.keys())
        if npz_keys != manifest_keys:
            missing = sorted(set(manifest_keys) - set(npz_keys))
            extra = sorted(set(npz_keys) - set(manifest_keys))
            details = []
            if missing:
                details.append(f"missing from npz: {missing}")
            if extra:
                details.append(f"extra in npz (not in manifest): {extra}")
            raise ValueError(
                f"Artifact {npz_path.name} tensor keys do not match shape_manifest ({'; '.join(details)}). "
                "The npz and JSON sidecar are out of sync; re-convert the source checkpoint."
            )
        # Materialize arrays inside the context manager while the file is open.
        arrays: dict[str, np.ndarray] = {}
        for name in npz_keys:
            arr = loaded[name]
            expected_shape = tuple(manifest[name])
            if tuple(arr.shape) != expected_shape:
                raise ValueError(
                    f"Artifact {npz_path.name} tensor {name!r} shape {tuple(arr.shape)} does not "
                    f"match manifest {expected_shape}. The npz and JSON sidecar are out of sync; "
                    "re-convert the source checkpoint."
                )
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(
                    f"Artifact {npz_path.name} tensor {name!r} has non-float32-compatible dtype "
                    f"{arr.dtype}; expected a floating-point dtype. The artifact may be corrupted."
                )
            arr32 = arr.astype(np.float32, copy=True)
            # Non-finite weights (NaN/Inf) silently turn every prediction into
            # NaN/Inf. Reject up front so corruption is attributed to the artifact,
            # not reported as a model bug downstream.
            if not np.isfinite(arr32).all():
                bad = np.argwhere(~np.isfinite(arr32))
                first = tuple(int(x) for x in bad[0]) if len(bad) else ()
                raise ValueError(
                    f"Artifact {npz_path.name} tensor {name!r} contains non-finite values "
                    f"(NaN/Inf) at {len(bad)} position(s), first at index {first}. "
                    "The weight file is corrupted; re-convert the source checkpoint."
                )
            arrays[name] = arr32

    # Run the implementation-specific contract after the ordinary NPZ-vs-
    # manifest checks. This preserves their more actionable diagnostics while
    # still rejecting a self-consistent NPZ/sidecar pair whose tensor contract
    # cannot be consumed safely by the shipped kernels.
    _validate_parameter_manifest(cfg, manifest, context=f"Artifact {npz_path.name} shape_manifest")

    expected_hash = metadata.get("npz_sha256")
    if expected_hash is None:
        # Backward compat: shipped artifacts predate the npz_sha256 field.
        # Warn loudly but still load — only key/shape/dtype mismatches are fatal.
        warnings.warn(
            f"Artifact {npz_path.name} metadata lacks 'npz_sha256'; skipping integrity hash "
            "verification. Re-convert the source checkpoint to embed the hash.",
            stacklevel=2,
        )
    elif _sha256(npz_path) != expected_hash:
        raise ValueError(
            f"Artifact {npz_path.name} SHA256 does not match metadata 'npz_sha256'. "
            "The weight file has been modified or corrupted; re-convert the source checkpoint."
        )
    return arrays


def _validate_loaded_metadata(metadata: dict[str, Any], cfg: ModelConfig, json_path: Path) -> None:
    """Semantic validation of the loaded JSON metadata against the rebuilt config.

    Catches metadata tampering/corruption that structural (key/shape/dtype) checks
    cannot: top-level fields inconsistent with the embedded ``model_config``,
    output_names relabeled to lie about the architecture, and unsupported
    architecture combos smuggled past the source-args audit (``_config_from_metadata``
    builds the dataclass directly without re-running that audit). The canonical
    ``metadata_sha256`` is verified separately in ``load_converted_checkpoint`` as a
    final blanket guard, after the more specific structural checks run.
    """
    # Top-level scalar fields must agree with the embedded model_config.
    top_model_name = metadata.get("model_name")
    if top_model_name != cfg.model_name:
        raise ValueError(
            f"Artifact {json_path.name} top-level model_name {top_model_name!r} does not match "
            f"model_config.model_name {cfg.model_name!r}. The metadata is inconsistent; "
            "re-convert the source checkpoint."
        )
    top_outputs = tuple(metadata.get("output_names") or ())
    if top_outputs != tuple(cfg.output_names):
        raise ValueError(
            f"Artifact {json_path.name} top-level output_names {top_outputs!r} do not match "
            f"model_config output_names {tuple(cfg.output_names)!r}. The metadata is inconsistent; "
            "re-convert the source checkpoint."
        )
    top_source_path = metadata.get("source_path")
    if top_source_path != str(cfg.source_path):
        raise ValueError(
            f"Artifact {json_path.name} top-level source_path {top_source_path!r} does not match "
            f"model_config.source_path {str(cfg.source_path)!r}. The metadata is inconsistent; "
            "re-convert the source checkpoint."
        )
    top_source_sha256 = metadata.get("source_sha256")
    if top_source_sha256 != cfg.source_sha256:
        raise ValueError(
            f"Artifact {json_path.name} top-level source_sha256 {top_source_sha256!r} does not match "
            f"model_config.source_sha256 {cfg.source_sha256!r}. The metadata is inconsistent; "
            "re-convert the source checkpoint."
        )
    if (
        not isinstance(cfg.source_sha256, str)
        or len(cfg.source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in cfg.source_sha256.lower())
    ):
        raise ValueError(
            f"Artifact {json_path.name} has invalid source_sha256 {cfg.source_sha256!r}; "
            "expected 64 hexadecimal characters"
        )
    # source_name: only cross-check when the top-level field is present (older
    # artifacts lack it); the model_config value is the source of truth.
    if "source_name" in metadata:
        top_src = metadata.get("source_name")
        if top_src != cfg.source_name:
            raise ValueError(
                f"Artifact {json_path.name} top-level source_name {top_src!r} does not match "
                f"model_config.source_name {cfg.source_name!r}. The metadata is inconsistent; "
                "re-convert the source checkpoint."
            )
    # Re-derive output names from the architecture combo and confirm the config
    # agrees — catches a tampered model_config whose output_names were relabeled
    # to misrepresent the head (e.g. a mos checkpoint relabeled as naturalness).
    expected = derive_output_names(cfg.model_name, (cfg.model_name, cfg.cnn_model, cfg.td, cfg.pool))
    if tuple(cfg.output_names) != expected:
        raise ValueError(
            f"Artifact {json_path.name} output_names {tuple(cfg.output_names)!r} are inconsistent "
            f"with its architecture (expected {expected!r}). The metadata may be tampered; "
            "re-convert the source checkpoint."
        )
    # Full structural/td-specific audit on the rebuilt config.
    validate_model_config(cfg)


def _verify_metadata_checksum(metadata: dict[str, Any], json_path: Path) -> None:
    """Final blanket guard: verify the canonical ``metadata_sha256``.

    Runs after the specific semantic + structural checks so those emit precise
    diagnostics for the cases they cover; this catches any remaining metadata
    field edit (e.g. ``source_sha256`` tampering) not otherwise guarded. Older
    artifacts (pre-``metadata_sha256``) warn, not fail, for backward compatibility.
    """
    expected_meta_hash = metadata.get("metadata_sha256")
    if expected_meta_hash is None:
        warnings.warn(
            f"Artifact {json_path.name} metadata lacks 'metadata_sha256'; skipping canonical "
            "metadata checksum verification. Re-convert the source checkpoint to embed the hash.",
            stacklevel=3,
        )
        return
    actual = canonical_metadata_checksum(metadata)
    if actual != expected_meta_hash:
        raise ValueError(
            f"Artifact {json_path.name} canonical metadata checksum does not match "
            "'metadata_sha256'. The metadata has been edited or corrupted; re-convert "
            "the source checkpoint."
        )


def _validate_metadata_schema(metadata: dict[str, Any], json_path: Path) -> dict[str, Any]:
    required = {
        "conversion_version",
        "source_path",
        "source_sha256",
        "model_name",
        "output_names",
        "model_config",
        "shape_manifest",
    }
    optional = {"source_name", "npz_sha256", "metadata_sha256"}
    missing = sorted(required - set(metadata))
    extra = sorted(set(metadata) - required - optional)
    if "shape_manifest" in missing:
        raise ValueError(
            f"Artifact metadata for {json_path.with_suffix('.npz').name} is missing a 'shape_manifest'; "
            "the JSON sidecar is malformed. Re-convert the source checkpoint."
        )
    if missing or extra:
        raise ValueError(
            f"Artifact metadata {json_path.name} has an invalid field set "
            f"(missing={missing}, unexpected={extra})"
        )
    model_config = metadata["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError(f"Artifact metadata {json_path.name} is missing a valid 'model_config' object")
    config_required = {
        "source_path",
        "source_sha256",
        "model_name",
        "cnn_model",
        "td",
        "td_2",
        "pool",
        "output_names",
        "feature",
        "cnn_pool_1",
        "cnn_pool_2",
        "cnn_pool_3",
        "td_sa_d_model",
        "td_sa_nhead",
        "td_sa_num_layers",
        "td_sa_h",
        "td_lstm_h",
        "td_lstm_bidirectional",
    }
    config_optional = {"source_name"}
    config_missing = sorted(config_required - set(model_config))
    config_extra = sorted(set(model_config) - config_required - config_optional)
    if config_missing or config_extra:
        raise ValueError(
            f"Artifact metadata {json_path.name} model_config has an invalid field set "
            f"(missing={config_missing}, unexpected={config_extra})"
        )
    feature = model_config["feature"]
    feature_fields = {
        "sr",
        "n_fft",
        "hop_length_seconds",
        "win_length_seconds",
        "n_mels",
        "fmax",
        "seg_length",
        "seg_hop_length",
        "max_segments",
    }
    if not isinstance(feature, dict) or set(feature) != feature_fields:
        actual = sorted(feature) if isinstance(feature, dict) else type(feature).__name__
        raise ValueError(
            f"Artifact metadata {json_path.name} model_config.feature has invalid fields {actual!r}; "
            f"expected {sorted(feature_fields)!r}"
        )
    for field in ("npz_sha256", "metadata_sha256"):
        value = metadata.get(field)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())
        ):
            raise ValueError(
                f"Artifact metadata {json_path.name} has invalid {field}={value!r}; "
                "expected 64 hexadecimal characters"
            )
    return model_config


def load_converted_checkpoint(artifact_path: str | Path) -> tuple[ModelConfig, dict[str, Any]]:
    path = Path(artifact_path).expanduser().resolve()
    metadata_path = _metadata_for_artifact(path)
    npz_path = _npz_for_artifact(path)
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"Artifact metadata {metadata_path.name} must contain a JSON object")
    version = metadata.get("conversion_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != CONVERSION_VERSION:
        raise ValueError(
            f"Converted artifact version {version!r} is incompatible with "
            f"conversion version {CONVERSION_VERSION}. Re-convert the source checkpoint."
        )
    model_config = _validate_metadata_schema(metadata, metadata_path)
    try:
        cfg = _config_from_metadata(model_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Artifact metadata {metadata_path.name} contains an invalid model_config: {exc}"
        ) from exc
    # 1. Semantic validation: reject tampered/inconsistent metadata early so a
    #    bad sidecar never reaches the weight loader (specific diagnostics).
    _validate_loaded_metadata(metadata, cfg, metadata_path)
    # 2. Structural validation: npz keys/shapes/dtypes/finiteness + npz_sha256.
    arrays = _verify_artifact_integrity(npz_path, metadata, cfg)
    # 3. Canonical metadata checksum: blanket guard for any remaining field edit.
    _verify_metadata_checksum(metadata, metadata_path)
    params: dict[str, Any] = {}
    for name, value in arrays.items():
        _insert_flat(params, name, value)
    return cfg, _tuplify_numeric_dicts(params)


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    precision: Precision = "float32",
) -> NisqaJaxModel:
    precision = _validate_precision(precision)
    if cache_dir is not None:
        # Configure before any JIT compilation of the model (load_converted_checkpoint
        # below does not compile; the first predict_* call does). Idempotent.
        _configure_persistent_cache(Path(cache_dir).expanduser().resolve() / "jax_compilation_cache")
    path = Path(checkpoint_path).expanduser()
    if path.suffix in {".npz", ".json"}:
        cfg, params = load_converted_checkpoint(path)
    else:
        cfg, params = convert_checkpoint(path, cache_dir=cache_dir)
    devices = jax.devices(device) if device else jax.devices()
    if not devices:
        raise RuntimeError(f"No JAX devices available for device selector: {device}")
    return NisqaJaxModel(config=cfg, params=params, device=devices[0], precision=precision)


def prewarm(
    model: NisqaJaxModel,
    batch_sizes: Sequence[int],
    bucket_lengths: Sequence[int],
    cache_dir: str | Path | None = None,
) -> None:
    """Pre-compile the JIT forward for each ``(batch_size, bucket_length)`` shape.

    Runs a tiny dummy ``predict_segments`` (zeros, no audio) for every shape so
    the persistent compilation cache is hot before real traffic — the first
    real call for a given shape otherwise pays the XLA compile cost. Call after
    ``load_model``. ``cache_dir`` configures the persistent cache if the model
    was not already loaded with one (idempotent; safe to pass the same value).
    Cheap: dummy zeros only, output is discarded.
    """
    if cache_dir is not None:
        # Idempotent: no-op if load_model already configured the cache.
        _configure_persistent_cache(Path(cache_dir).expanduser().resolve() / "jax_compilation_cache")
    feat = model.config.feature
    for bs in batch_sizes:
        if bs < 1:
            raise ValueError(f"batch_sizes must be >= 1, got {bs}")
        for bl in bucket_lengths:
            if bl < 1:
                raise ValueError(f"bucket_lengths must be >= 1, got {bl}")
            # Dummy zeros of the exact compiled shape [bs, bl, 1, n_mels, seg_length];
            # n_wins all = bl so the whole time axis is "valid" (no masking edge cases).
            x = np.zeros((bs, bl, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
            n_wins = np.full((bs,), bl, dtype=np.int32)
            model.predict_segments(x, n_wins)
