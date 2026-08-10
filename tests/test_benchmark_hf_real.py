from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_hf_real.py"


@pytest.fixture(scope="module")
def benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_hf_real", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_self_attention_window_guard_boundary(benchmark_module) -> None:
    # The shipped self-attention profile uses 10 ms frames, 15-frame windows,
    # and a four-window hop. These are the exact boundary samples for 1300.
    assert benchmark_module._self_att_n_wins(417_040, 8_000) == 1_300
    assert benchmark_module._self_att_n_wins(417_120, 8_000) == 1_301


def test_parser_uses_serial_decode_by_default(benchmark_module) -> None:
    args = benchmark_module._build_parser().parse_args(["--torch-source-root", "/tmp/torch-source"])
    assert args.decode_threads == 0
    assert args.frontend_mode == "native"
    assert args.batch_mode == "fixed"


def test_schedule_uses_feature_config_and_td_for_default_bucket(benchmark_module, monkeypatch) -> None:
    benchmark_module.estimate_n_wins = lambda path, cfg: {"short": 8, "long": 140}[path]
    benchmark_module._fixed_chunks = lambda order, batch_size: [order]
    cfg = SimpleNamespace(max_segments=1300)
    chunks, estimates, bucket = benchmark_module._schedule(
        ["short", "long"], cfg, "self_att", 2, "fixed", None, None
    )
    assert chunks == [([1, 0], 2)]
    assert estimates == [8, 140]
    assert bucket == 128


def test_model_argument_validation(benchmark_module) -> None:
    assert benchmark_module._parse_models("nisqa_mos_only,nisqa_tts") == [
        "nisqa_mos_only",
        "nisqa_tts",
    ]
    with pytest.raises(ValueError, match="must not contain duplicates"):
        benchmark_module._parse_models("nisqa,nisqa")
    with pytest.raises(ValueError, match="unknown model"):
        benchmark_module._parse_models("not-a-model")
