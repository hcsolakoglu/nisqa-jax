from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureConfig:
    sr: int | None
    n_fft: int
    hop_length_seconds: float
    win_length_seconds: float
    n_mels: int
    fmax: int | float | None
    seg_length: int
    seg_hop_length: int
    max_segments: int


@dataclass(frozen=True)
class ModelConfig:
    source_path: Path
    source_sha256: str
    model_name: str
    cnn_model: str
    td: str
    td_2: str | None
    pool: str
    output_names: tuple[str, ...]
    feature: FeatureConfig
    cnn_pool_1: tuple[int, int] | None
    cnn_pool_2: tuple[int, int] | None
    cnn_pool_3: tuple[int, int] | None
    td_sa_d_model: int | None
    td_sa_nhead: int | None
    td_sa_num_layers: int | None
    td_sa_h: int | None
    td_lstm_h: int | None
    td_lstm_bidirectional: bool | None

    @property
    def is_dimensional(self) -> bool:
        return self.model_name == "NISQA_DIM"

    @property
    def cache_key(self) -> str:
        return f"{self.source_path.stem}-{self.source_sha256[:16]}"


def _tuple2(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    return int(value[0]), int(value[1])


def config_from_checkpoint_args(args: dict[str, Any], source_path: Path, sha256: str) -> ModelConfig:
    model_name = args["model"]
    if model_name == "NISQA_DIM":
        output_names = ("mos", "noi", "dis", "col", "loud")
    elif source_path.name == "nisqa_tts.tar":
        output_names = ("naturalness",)
    elif model_name == "NISQA":
        output_names = ("mos",)
    else:
        raise NotImplementedError(f"Unsupported model architecture for JAX inference v1: {model_name}")

    if model_name not in {"NISQA", "NISQA_DIM"}:
        raise NotImplementedError(f"Unsupported model architecture for JAX inference v1: {model_name}")
    if args.get("double_ended") or model_name == "NISQA_DE":
        raise NotImplementedError("NISQA_DE is unsupported in JAX inference v1")
    if args.get("td_2") not in {None, "skip"}:
        raise NotImplementedError("Only checkpoints with td_2=skip are supported in JAX inference v1")
    # JAX inference v1 implements single-head self-attention only (see _self_attention_layer:
    # scale 1/sqrt(d_model), no head reshape). Silently accepting nhead>1 would yield wrong
    # results for custom multi-head checkpoints, so reject it at the API boundary.
    if args.get("td_sa_nhead") not in (None, 1):
        raise NotImplementedError(
            "multi-head self-attention (td_sa_nhead>1) is not supported in JAX inference v1"
        )

    supported = {
        ("NISQA_DIM", "adapt", "self_att", "att"),
        ("NISQA", "adapt", "self_att", "att"),
        ("NISQA", "standard", "lstm", "last_step_bi"),
    }
    combo = (model_name, args.get("cnn_model"), args.get("td"), args.get("pool"))
    if combo not in supported:
        raise NotImplementedError(f"Unsupported shipped-checkpoint architecture: {combo}")

    feature = FeatureConfig(
        sr=args.get("ms_sr"),
        n_fft=int(args["ms_n_fft"]),
        hop_length_seconds=float(args["ms_hop_length"]),
        win_length_seconds=float(args["ms_win_length"]),
        n_mels=int(args["ms_n_mels"]),
        fmax=args.get("ms_fmax"),
        seg_length=int(args["ms_seg_length"]),
        seg_hop_length=int(args["ms_seg_hop_length"]),
        max_segments=int(args["ms_max_segments"]),
    )
    return ModelConfig(
        source_path=source_path,
        source_sha256=sha256,
        model_name=model_name,
        cnn_model=args["cnn_model"],
        td=args["td"],
        td_2=args.get("td_2"),
        pool=args["pool"],
        output_names=output_names,
        feature=feature,
        cnn_pool_1=_tuple2(args.get("cnn_pool_1")),
        cnn_pool_2=_tuple2(args.get("cnn_pool_2")),
        cnn_pool_3=_tuple2(args.get("cnn_pool_3")),
        td_sa_d_model=args.get("td_sa_d_model"),
        td_sa_nhead=args.get("td_sa_nhead"),
        td_sa_num_layers=args.get("td_sa_num_layers"),
        td_sa_h=args.get("td_sa_h"),
        td_lstm_h=args.get("td_lstm_h"),
        td_lstm_bidirectional=args.get("td_lstm_bidirectional"),
    )
