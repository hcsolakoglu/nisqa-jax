"""JAX inference port for the shipped NISQA checkpoints."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .checkpoint import convert_checkpoint, load_converted_checkpoint, load_model, prewarm, prewarm_pairs
from .config import FeatureConfig, ModelConfig
from .model import Precision

try:
    __version__ = _pkg_version("nisqa-jax")
except PackageNotFoundError:  # editable/source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "FeatureConfig",
    "ModelConfig",
    "Precision",
    "__version__",
    "convert_checkpoint",
    "load_converted_checkpoint",
    "load_model",
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
