"""JAX inference port for the shipped NISQA checkpoints."""

from .checkpoint import convert_checkpoint, load_converted_checkpoint, load_model, prewarm
from .config import FeatureConfig, ModelConfig
from .model import Precision

__all__ = [
    "FeatureConfig",
    "ModelConfig",
    "Precision",
    "convert_checkpoint",
    "load_converted_checkpoint",
    "load_model",
    "prewarm",
    "predict_batch",
    "predict_file",
]


def __getattr__(name: str):
    if name in {"predict_batch", "predict_file"}:
        from .predict import predict_batch, predict_file

        return {"predict_batch": predict_batch, "predict_file": predict_file}[name]
    raise AttributeError(name)
