"""JAX inference port for NISQA speech quality models."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .checkpoint import convert_checkpoint, load_converted_checkpoint, load_model, prewarm, prewarm_pairs
from .config import FeatureConfig, ModelConfig
from .double_ended import DoubleEndedConfig, NisqaDeJaxModel, forward_double_ended
from .double_ended_checkpoint import convert_double_ended_checkpoint
from .double_ended_features import (
    canonical_double_ended_feature_config,
    pack_double_ended_segments,
    preprocess_pair,
)
from .model import Precision

try:
    __version__ = _pkg_version("nisqa-jax")
except PackageNotFoundError:  # editable/source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "DoubleEndedConfig",
    "FeatureConfig",
    "ModelConfig",
    "NisqaDeJaxModel",
    "Precision",
    "__version__",
    "canonical_double_ended_feature_config",
    "convert_checkpoint",
    "convert_double_ended_checkpoint",
    "forward_double_ended",
    "load_converted_checkpoint",
    "load_model",
    "pack_double_ended_segments",
    "preprocess_pair",
    "prewarm",
    "prewarm_pairs",
    "predict_batch",
    "predict_file",
]


def __getattr__(name: str):
    if name in {"predict_batch", "predict_file"}:
        from .predict import predict_batch, predict_file

        return {"predict_batch": predict_batch, "predict_file": predict_file}[name]
    raise AttributeError(name)
