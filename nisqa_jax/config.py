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
    # Stable original model identity from the source checkpoint's ``args['name']``
    # training-run label (e.g. 'NISQA_TTS_v1', 'NISQAv2_mos_only'). Propagated
    # into converted JSON metadata so downstream consumers (CSV lane) can report
    # the original run identity without re-reading the .tar. ``None`` for
    # pre-source-name artifacts (backward-compat fallback). This is a free-form
    # label, NOT a structural guarantee — architecture discrimination still
    # derives from the (model, cnn_model, td, pool) combo below.
    source_name: str | None
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


def _expect_arg(args: dict[str, Any], key: str, expected: Any, *, required: bool = True) -> None:
    """Reject source-checkpoint arguments outside the shipped model profiles.

    Some old callers construct a minimal args mapping rather than preserving
    every training-only field. Optional fields are therefore checked whenever
    present, while all arguments that affect the implemented inference graph or
    audio frontend are required.
    """
    if key not in args:
        if required:
            raise NotImplementedError(f"Required shipped-checkpoint argument {key!r} is missing")
        return
    value = args[key]
    if key.startswith("cnn_pool_") or key == "cnn_kernel_size":
        value = tuple(value) if value is not None else None
        expected = tuple(expected) if expected is not None else None
    if value != expected:
        raise NotImplementedError(
            f"Unsupported {key}={args[key]!r}; the shipped JAX inference profile requires {expected!r}"
        )


def _validate_shipped_profile_args(
    args: dict[str, Any],
    combo: tuple[str, str, str, str],
) -> None:
    """Validate every fixed assumption made by the JAX kernels/frontend."""
    is_tts = combo == ("NISQA", "standard", "lstm", "last_step_bi")
    exact_common = {
        "ms_sr": None,
        "ms_n_fft": 4096,
        "ms_hop_length": 0.01,
        "ms_win_length": 0.02,
        "ms_n_mels": 48,
        "ms_seg_length": 15,
        "cnn_c_out_1": 16,
        "cnn_c_out_2": 32,
        "cnn_c_out_3": 64,
        "cnn_kernel_size": (3, 3),
        "pool_output_size": 1,
    }
    for key, expected in exact_common.items():
        # Shape-defining fields absent from legacy hand-built argument mappings
        # are still enforced independently by the strict state/manifest audit.
        _expect_arg(args, key, expected, required=key.startswith("ms_"))

    exact: dict[str, Any]
    if is_tts:
        exact = {
            "ms_fmax": 8000,
            "ms_seg_hop_length": 1,
            "ms_max_segments": 6000,
            "cnn_pool_1": None,
            "cnn_pool_2": None,
            "cnn_pool_3": None,
            "cnn_fc_out_h": 20,
            "td_lstm_h": 128,
            "td_lstm_num_layers": 1,
            "td_lstm_bidirectional": True,
            "td_sa_d_model": None,
            "td_sa_num_layers": None,
            "td_sa_h": None,
            "td_sa_pos_enc": None,
            "pool_att_h": None,
        }
    else:
        exact = {
            "ms_fmax": 20000,
            "ms_seg_hop_length": 4,
            "ms_max_segments": 1300,
            "cnn_pool_1": (24, 7),
            "cnn_pool_2": (12, 5),
            "cnn_pool_3": (6, 3),
            "cnn_fc_out_h": None,
            "td_sa_d_model": 64,
            "td_sa_nhead": 1,
            "td_sa_num_layers": 2,
            "td_sa_h": 64,
            "td_sa_pos_enc": False,
            "td_lstm_h": None,
            "td_lstm_num_layers": None,
            "td_lstm_bidirectional": None,
            "pool_att_h": 128,
        }
    for key, expected in exact.items():
        # Existing programmatic callers may omit inactive/training-only knobs.
        # Active graph and frontend fields are always required.
        active_graph_fields = (
            {
                "cnn_fc_out_h",
                "td_lstm_h",
                "td_lstm_num_layers",
                "td_lstm_bidirectional",
            }
            if is_tts
            else {"td_sa_d_model", "td_sa_nhead", "td_sa_num_layers", "td_sa_h"}
        )
        required = key.startswith("ms_") or key.startswith("cnn_pool_") or key in active_graph_fields
        _expect_arg(args, key, expected, required=required)


# The full set of (model, cnn_model, td, pool) architecture combinations the
# JAX port implements. Used both at conversion time (reject unsupported source
# checkpoints) and at load time (reject tampered/unsupported converted metadata).
SUPPORTED_ARCH_COMBOS: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        ("NISQA_DIM", "adapt", "self_att", "att"),
        ("NISQA", "adapt", "self_att", "att"),
        ("NISQA", "standard", "lstm", "last_step_bi"),
    }
)

# Output names that may legitimately appear in a converted artifact. The loader
# rejects any output name outside this set to catch metadata tampering.
_KNOWN_OUTPUT_NAMES: frozenset[str] = frozenset({"mos", "noi", "dis", "col", "loud", "naturalness"})


def derive_output_names(model_name: str, combo: tuple[str, str, str, str]) -> tuple[str, ...]:
    """Output names derived from the validated architecture combo, not the filename.

    The TTS/naturalness model is the unique ("NISQA", "standard", "lstm",
    "last_step_bi") checkpoint — its head is named `naturalness`. This is robust
    to renaming: a renamed nisqa_tts.tar still loads as naturalness, and a
    renamed nisqa_mos_only.tar (which is ("NISQA","adapt","self_att","att"))
    does NOT become naturalness. The `name` training-run label ('NISQA_TTS_v1')
    is NOT used — it is a free-form string, not a structural guarantee.
    """
    if model_name == "NISQA_DIM":
        return ("mos", "noi", "dis", "col", "loud")
    if combo == ("NISQA", "standard", "lstm", "last_step_bi"):
        return ("naturalness",)
    return ("mos",)


def validate_model_config(cfg: ModelConfig) -> None:
    """Semantic validation of a loaded/converted ModelConfig.

    Rejects metadata tampering that ``_config_from_metadata`` would otherwise
    accept blindly (it constructs the dataclass without re-running the source-
    args audit): unsupported architecture combos, output_names inconsistent with
    the combo, unknown output names, and td-specific field impossibilities.
    Raises ``ValueError`` on any inconsistency so corruption fails early at load
    rather than producing silently wrong inference.
    """
    combo = (cfg.model_name, cfg.cnn_model, cfg.td, cfg.pool)
    if combo not in SUPPORTED_ARCH_COMBOS:
        raise ValueError(
            f"Loaded artifact has unsupported architecture combo {combo}; supported: "
            f"{sorted(SUPPORTED_ARCH_COMBOS)}. The metadata may be tampered or corrupted; "
            "re-convert the source checkpoint."
        )
    if cfg.model_name not in {"NISQA", "NISQA_DIM"}:
        raise ValueError(f"Loaded artifact has unsupported model_name {cfg.model_name!r}")
    if cfg.td_2 not in {None, "skip"}:
        raise ValueError(f"Loaded artifact has unsupported td_2={cfg.td_2!r}")
    expected = derive_output_names(cfg.model_name, combo)
    if tuple(cfg.output_names) != expected:
        raise ValueError(
            f"Loaded artifact output_names {tuple(cfg.output_names)!r} are inconsistent with "
            f"architecture combo {combo} (expected {expected!r}). The metadata may be tampered; "
            "re-convert the source checkpoint."
        )
    unknown = [n for n in cfg.output_names if n not in _KNOWN_OUTPUT_NAMES]
    if unknown:
        raise ValueError(f"Loaded artifact has unknown output names {unknown!r}")
    is_tts = combo == ("NISQA", "standard", "lstm", "last_step_bi")
    expected_feature = (
        FeatureConfig(
            sr=None,
            n_fft=4096,
            hop_length_seconds=0.01,
            win_length_seconds=0.02,
            n_mels=48,
            fmax=8000,
            seg_length=15,
            seg_hop_length=1,
            max_segments=6000,
        )
        if is_tts
        else FeatureConfig(
            sr=None,
            n_fft=4096,
            hop_length_seconds=0.01,
            win_length_seconds=0.02,
            n_mels=48,
            fmax=20000,
            seg_length=15,
            seg_hop_length=4,
            max_segments=1300,
        )
    )
    if cfg.feature != expected_feature:
        raise ValueError(
            f"Loaded artifact feature config {cfg.feature!r} does not match the exact shipped "
            f"profile {expected_feature!r}"
        )
    expected_pools = (None, None, None) if is_tts else ((24, 7), (12, 5), (6, 3))
    actual_pools = (cfg.cnn_pool_1, cfg.cnn_pool_2, cfg.cnn_pool_3)
    if actual_pools != expected_pools:
        raise ValueError(
            f"Loaded artifact CNN pools {actual_pools!r} do not match the exact shipped "
            f"profile {expected_pools!r}"
        )
    # td-specific field consistency (mirrors the source-args audit).
    if cfg.td == "self_att":
        attention_profile = (
            cfg.td_sa_d_model,
            cfg.td_sa_nhead,
            cfg.td_sa_num_layers,
            cfg.td_sa_h,
            cfg.td_lstm_h,
            cfg.td_lstm_bidirectional,
        )
        expected_attention = (64, 1, 2, 64, None, None)
        if attention_profile != expected_attention:
            raise ValueError(
                f"Loaded self_att artifact profile {attention_profile!r} does not match the "
                f"exact shipped profile {expected_attention!r}"
            )
    elif cfg.td == "lstm":
        lstm_profile = (
            cfg.td_lstm_h,
            cfg.td_lstm_bidirectional,
            cfg.td_sa_d_model,
            cfg.td_sa_nhead,
            cfg.td_sa_num_layers,
            cfg.td_sa_h,
        )
        expected_lstm = (128, True, None, None, None, None)
        if lstm_profile != expected_lstm:
            raise ValueError(
                f"Loaded lstm artifact profile {lstm_profile!r} does not match the exact "
                f"shipped profile {expected_lstm!r}"
            )


def config_from_checkpoint_args(args: dict[str, Any], source_path: Path, sha256: str) -> ModelConfig:
    model_name = args["model"]
    if model_name not in {"NISQA", "NISQA_DIM"}:
        raise NotImplementedError(f"Unsupported model architecture for JAX inference v1: {model_name}")
    if args.get("double_ended") or model_name == "NISQA_DE":
        raise NotImplementedError("NISQA_DE is unsupported in JAX inference v1")
    if args.get("td_2") not in {None, "skip"}:
        raise NotImplementedError("Only checkpoints with td_2=skip are supported in JAX inference v1")

    combo = (model_name, args.get("cnn_model"), args.get("td"), args.get("pool"))
    if combo not in SUPPORTED_ARCH_COMBOS:
        raise NotImplementedError(f"Unsupported shipped-checkpoint architecture: {combo}")
    # Membership in SUPPORTED_ARCH_COMBOS (all-str tuples) is verified above; the
    # cast narrows the args.get Any|None components for the type checker.
    combo_typed: tuple[str, str, str, str] = combo  # type: ignore[assignment]

    td = args.get("td")
    cnn_model = args.get("cnn_model")

    # --- Per-architecture arg audit: reject unsupported combinations clearly. ---
    # The JAX port implements a fixed subset of the PyTorch NISQA architecture
    # space. Accepting a checkpoint whose args select an unimplemented path would
    # either crash deep in conversion or silently produce wrong numbers, so each
    # unsupported knob is rejected at the API boundary with an actionable error.

    # JAX inference v1 implements single-head self-attention only (see
    # _self_attention_layer: scale 1/sqrt(d_model), no head reshape). Silently
    # accepting nhead>1 would yield wrong results for custom multi-head
    # checkpoints, so reject it at the API boundary.
    if args.get("td_sa_nhead") not in (None, 1):
        raise NotImplementedError("multi-head self-attention (td_sa_nhead>1) is not supported in JAX inference v1")

    if td == "self_att":
        # Positional encodings are not implemented in the JAX self-attention path
        # (_self_attention_layer applies no positional embedding). Accepting a
        # checkpoint trained with pos_enc would drop the encoding and corrupt
        # outputs, so reject any truthy td_sa_pos_enc.
        if args.get("td_sa_pos_enc"):
            raise NotImplementedError(
                "td_sa_pos_enc is enabled but positional encodings are not supported in JAX inference v1"
            )
        # The self-attention path iterates `range(td_sa_num_layers)` transformer
        # layers; require a positive int so conversion does not silently produce
        # an empty layer tuple.
        n_layers = args.get("td_sa_num_layers")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise NotImplementedError(f"td_sa_num_layers must be a positive int for self-attention, got {n_layers!r}")
        for key in ("td_sa_d_model", "td_sa_h"):
            val = args.get(key)
            if not isinstance(val, int) or val < 1:
                raise NotImplementedError(f"{key} must be a positive int for self-attention, got {val!r}")
        # self_att checkpoints use the adapt CNN, which has no fc_out head.
        if args.get("td_lstm_num_layers") is not None:
            raise NotImplementedError(
                "td_lstm_num_layers is set on a self-attention checkpoint; this mixed "
                "configuration is not supported in JAX inference v1"
            )

    if td == "lstm":
        # The JAX LSTM path (_bidirectional_lstm) implements exactly one BiLSTM
        # layer per direction (it reads weight_ih_l0/weight_hh_l0 only). Multi-
        # layer LSTMs (td_lstm_num_layers>1) would need l1/l2... weights that the
        # converter never extracts, so reject anything other than a single layer.
        n_lstm_layers = args.get("td_lstm_num_layers")
        if n_lstm_layers != 1:
            raise NotImplementedError(
                f"td_lstm_num_layers must be 1 for LSTM checkpoints (JAX implements a single "
                f"BiLSTM layer), got {n_lstm_layers!r}"
            )
        if not args.get("td_lstm_bidirectional"):
            raise NotImplementedError(
                "td_lstm_bidirectional must be True; only the BiLSTM path is implemented in JAX inference v1"
            )
        if not isinstance(args.get("td_lstm_h"), int) or args["td_lstm_h"] < 1:
            raise NotImplementedError(
                f"td_lstm_h must be a positive int for LSTM checkpoints, got {args.get('td_lstm_h')!r}"
            )

    # The adapt CNN has no fc_out head (_cnn only emits fc_out for cnn_model ==
    # 'standard'). A checkpoint that sets cnn_fc_out_h on an adapt CNN would have
    # an fc_out module the converter ignores, masking an architecture mismatch.
    if cnn_model == "adapt" and args.get("cnn_fc_out_h") is not None:
        raise NotImplementedError(
            f"cnn_fc_out_h={args.get('cnn_fc_out_h')!r} is set on an adapt-CNN checkpoint, but "
            "the adapt path has no fc_out head; this configuration is not supported"
        )

    # After the general unsupported-path checks above have emitted their more
    # specific diagnostics, enforce the exact dimensions/frontend settings of
    # the three shipped profiles.
    _validate_shipped_profile_args(args, combo_typed)

    # Output naming is derived from the model IDENTITY (the validated architecture
    # combo), NOT the checkpoint filename. See ``derive_output_names`` for the
    # rationale (robust to checkpoint renaming; the `name` training-run label is
    # a free-form string, not a structural guarantee).
    output_names: tuple[str, ...] = derive_output_names(model_name, combo_typed)

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
    # Stable original training-run label from args['name']; coerce to str|None
    # so the field type is stable regardless of what the source checkpoint stored.
    raw_name = args.get("name")
    source_name = raw_name if isinstance(raw_name, str) and raw_name else None
    return ModelConfig(
        source_path=source_path,
        source_sha256=sha256,
        model_name=model_name,
        source_name=source_name,
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
