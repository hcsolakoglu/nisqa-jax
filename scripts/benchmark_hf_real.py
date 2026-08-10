#!/usr/bin/env python3
"""Benchmark NISQA-JAX on a bounded, streamed Hugging Face audio sample.

The harness deliberately keeps dataset acquisition, preprocessing, model-forward
latency, framework comparison, and correctness evidence separate. It materializes
only the requested examples as temporary WAV files, runs the JAX and upstream
PyTorch models over the same length-aware batches, and records machine-readable
results. The audio and Hugging Face cache are task-owned scratch data and are
removed by default.

Example::

    python scripts/benchmark_hf_real.py \
        --torch-source-root /path/to/NISQA \
        --jax-device gpu --torch-device cuda \
        --samples 2000 --output docs/benchmarks/results/hf-minds14-2k.json

The default dataset is PolyAI/minds14. Its small language-specific Parquet
configurations are streamed in sequence to avoid touching the much larger
aggregate ``all`` shards. It is an open CC BY 4.0 audio dataset with
variable-duration speech. The resolved Hub commit is recorded in the output.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import os
import platform
import pstats
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These imports are intentionally populated after HF sample materialization.
# Importing predict/bench_compare initializes JAX's CUDA plugin, which can make
# torchcodec's Parquet audio decoder contend with the accelerator during setup.
_model_args: Any = None
_load_torch_checkpoint: Any = None
estimate_n_wins: Any = None
preprocess_file: Any = None
_cost_aware_chunks: Any = None
_cost_exponent: Any = None
_fixed_chunks: Any = None
_round_up: Any = None
WEIGHTS_DIR: Path


def _load_runtime_helpers() -> None:
    global _model_args, _load_torch_checkpoint, estimate_n_wins, preprocess_file
    global _cost_aware_chunks, _cost_exponent, _fixed_chunks, _round_up, WEIGHTS_DIR
    from nisqa_jax.bench_compare import _model_args as model_args
    from nisqa_jax.checkpoint import _load_torch_checkpoint as load_torch_checkpoint
    from nisqa_jax.features import estimate_n_wins as estimate_windows
    from nisqa_jax.features import preprocess_file as preprocess_audio
    from nisqa_jax.predict import _cost_aware_chunks as cost_aware_chunks
    from nisqa_jax.predict import _cost_exponent as cost_exponent
    from nisqa_jax.predict import _fixed_chunks as fixed_chunks
    from nisqa_jax.predict import _round_up as round_up
    from nisqa_jax.weights import WEIGHTS_DIR as weights_dir

    _model_args = model_args
    _load_torch_checkpoint = load_torch_checkpoint
    estimate_n_wins = estimate_windows
    preprocess_file = preprocess_audio
    _cost_aware_chunks = cost_aware_chunks
    _cost_exponent = cost_exponent
    _fixed_chunks = fixed_chunks
    _round_up = round_up
    WEIGHTS_DIR = weights_dir

MODEL_FILES = {
    "nisqa_mos_only": "nisqa_mos_only.npz",
    "nisqa": "nisqa.npz",
    "nisqa_tts": "nisqa_tts.npz",
}
MODEL_TAR_FILES = {
    "nisqa_mos_only": "nisqa_mos_only.tar",
    "nisqa": "nisqa.tar",
    "nisqa_tts": "nisqa_tts.tar",
}
DEFAULT_DATASET = "PolyAI/minds14"
DEFAULT_CONFIGS = "en-US,en-GB,de-DE,fr-FR,es-ES"
DEFAULT_SEED = 20260810
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "results" / "hf-minds14-2k.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_value) + "\n")


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_clean": run("status", "--porcelain") == "",
    }


def _nvidia_identity() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip() or "nvidia-smi failed"}
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"name": fields[0], "driver": fields[1], "memory_mib": fields[2]})
    return {"available": bool(rows), "gpus": rows}


def _parse_models(raw: str) -> list[str]:
    names = list(MODEL_FILES) if raw == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(names) - set(MODEL_FILES))
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}; choose from {sorted(MODEL_FILES)} or all")
    if not names:
        raise ValueError("at least one model is required")
    if len(set(names)) != len(names):
        raise ValueError("--models must not contain duplicates")
    return names


def _set_hf_cache(work_dir: Path) -> dict[str, str]:
    # Keep Hub/Arrow cache inside the task-owned scratch directory. This prevents
    # a streaming benchmark from silently filling the user's shared cache and lets
    # the default cleanup remove exactly what this run created.
    hf_home = work_dir / "hf-home"
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    for path in (hf_home, hub_cache, datasets_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    return {"HF_HOME": str(hf_home), "HF_HUB_CACHE": str(hub_cache), "HF_DATASETS_CACHE": str(datasets_cache)}


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _decode_audio(audio: Any) -> tuple[np.ndarray, int]:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        sample_rate = int(samples.sample_rate)
    elif isinstance(audio, dict) and "array" in audio:
        data = audio["array"]
        sample_rate = int(audio["sampling_rate"])
    else:
        raise TypeError(f"unsupported Hugging Face audio value: {type(audio).__name__}")

    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    if data.ndim == 1:
        waveform = data
    elif data.ndim == 2:
        # datasets' AudioDecoder returns [channels, samples]. SoundFile writes
        # [samples, channels], which preserves the channels for NISQA/librosa to
        # downmix using its normal production path.
        waveform = data.T if data.shape[0] <= 32 else data
    else:
        raise ValueError(f"decoded audio must be 1-D or 2-D, got {data.shape}")
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("decoded audio is empty or contains NaN/Inf")
    return waveform, sample_rate


def _self_att_n_wins(num_samples: int, sample_rate: int) -> int:
    hop_length = int(sample_rate * 0.01)
    n_frames = 1 + num_samples // hop_length
    n_wins_raw = n_frames - (15 - 1)
    return max(1, math.ceil(n_wins_raw / 4))


def _resolve_dataset_revision(dataset_id: str, requested_revision: str | None) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_id, revision=requested_revision)
    tags = list(getattr(info, "tags", []) or [])
    license_tags = sorted(tag for tag in tags if tag.startswith("license:"))
    return str(info.sha), {
        "id": dataset_id,
        "requested_revision": requested_revision or "main",
        "resolved_revision": str(info.sha),
        "gated": bool(getattr(info, "gated", False)),
        "license_tags": license_tags,
    }


def _download_hf_shard(repo_id: str, filename: str, revision: str, target: Path) -> Path:
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    quoted_filename = quote(filename, safe="/.")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_filename}"
    request = Request(url, headers={"User-Agent": "nisqa-jax-real-benchmark/1"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def _collect_streamed_audio(args: argparse.Namespace, work_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    from datasets import load_dataset

    configs = [item.strip() for item in args.config.split(",") if item.strip()]
    if not configs:
        raise ValueError("--config must contain at least one dataset configuration")
    revision, dataset_meta = _resolve_dataset_revision(args.dataset, args.revision)

    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "samples.jsonl"
    paths: list[Path] = []
    durations: list[float] = []
    sample_rates: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    config_counts: Counter[str] = Counter()
    source_ids: list[str] = []
    started = time.perf_counter()
    rejected = 0
    rejected_length = 0
    streamed_examples = 0
    downloaded_shards: list[dict[str, Any]] = []

    with manifest_path.open("w") as manifest:
        for config_index, config in enumerate(configs):
            if len(paths) >= args.samples:
                break
            if args.download_mode == "download_shards":
                filename = f"{config}/{args.split}-00000-of-00001.parquet"
                shard_path = _download_hf_shard(
                    args.dataset,
                    filename,
                    revision,
                    work_dir / "shards" / config / f"{args.split}-00000-of-00001.parquet",
                )
                downloaded_shards.append(
                    {"config": config, "filename": filename, "bytes": shard_path.stat().st_size}
                )
                dataset = load_dataset(
                    "parquet",
                    data_files={args.split: str(shard_path)},
                    split=args.split,
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    args.dataset,
                    config,
                    split=args.split,
                    revision=revision,
                    streaming=True,
                )
            dataset = dataset.shuffle(seed=args.seed + config_index, buffer_size=args.shuffle_buffer)
            if args.decode_threads > 0:
                dataset = dataset.decode(num_threads=args.decode_threads)
            for example in dataset:
                if len(paths) >= args.samples:
                    break
                streamed_examples += 1
                try:
                    waveform, sample_rate = _decode_audio(example["audio"])
                except (TypeError, ValueError, KeyError):
                    rejected += 1
                    continue
                if args.max_self_att_wins > 0 and _self_att_n_wins(
                    int(waveform.shape[0]), sample_rate
                ) > args.max_self_att_wins:
                    rejected_length += 1
                    continue
                path = audio_dir / f"{len(paths):06d}.wav"
                sf.write(path, waveform, sample_rate, format="WAV", subtype="FLOAT")
                source_id = str(example.get("path", example.get("id", len(paths))))
                record = {
                    "index": len(paths),
                    "config": config,
                    "source_id": source_id,
                    "source_id_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
                    "audio_path": path.name,
                    "sample_rate": sample_rate,
                    "num_samples": int(waveform.shape[0]),
                    "channels": int(waveform.shape[1]) if waveform.ndim == 2 else 1,
                    "duration_seconds": float(waveform.shape[0] / sample_rate),
                    "language": example.get("lang_id"),
                    "intent": example.get("intent_class"),
                }
                manifest.write(json.dumps(record, sort_keys=True, default=_json_value) + "\n")
                paths.append(path)
                source_ids.append(source_id)
                config_counts[config] += 1
                durations.append(record["duration_seconds"])
                sample_rates[str(sample_rate)] += 1
                languages[str(record["language"])] += 1
                intents[str(record["intent"])] += 1

    if len(paths) != args.samples:
        raise RuntimeError(
            f"stream ended after {len(paths)} valid examples, requested {args.samples}; "
            f"streamed={streamed_examples} rejected_decode={rejected} rejected_length={rejected_length}"
        )
    durations_np = np.asarray(durations, dtype=np.float64)
    dataset_meta.update(
        {
            "configs": configs,
            "split": args.split,
            "samples_requested": args.samples,
            "samples_materialized": len(paths),
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "decode_threads": args.decode_threads,
            "rejected_decode_examples": rejected,
            "rejected_length_examples": rejected_length,
            "streamed_examples": streamed_examples,
            "max_self_att_wins": args.max_self_att_wins,
            "stream_seconds": time.perf_counter() - started,
            "materialized_audio_bytes": sum(path.stat().st_size for path in paths),
            "stream_cache_bytes": _tree_size(work_dir / "hf-home"),
            "download_mode": args.download_mode,
            "downloaded_shards": downloaded_shards,
            "downloaded_shard_bytes": sum(item["bytes"] for item in downloaded_shards),
            "sample_manifest_sha256": _sha256(manifest_path),
            "duration_seconds": {
                "min": float(durations_np.min()),
                "p25": float(np.percentile(durations_np, 25)),
                "median": float(np.median(durations_np)),
                "p75": float(np.percentile(durations_np, 75)),
                "p95": float(np.percentile(durations_np, 95)),
                "max": float(durations_np.max()),
                "mean": float(durations_np.mean()),
                "total": float(durations_np.sum()),
            },
            "sample_rates": dict(sample_rates),
            "config_counts": dict(config_counts),
            "language_counts": dict(languages),
            "intent_counts": dict(intents),
            "unique_source_ids": len(set(source_ids)) if source_ids else len(paths),
        }
    )
    return paths, dataset_meta


def _load_torch_model(torch: Any, source_root: Path, checkpoint: Path, device_name: str):
    source_module = source_root / "nisqa" / "NISQA_lib.py"
    if not source_module.is_file():
        raise FileNotFoundError(f"PyTorch NISQA source not found at {source_module}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from nisqa import NISQA_lib as nl

    checkpoint_data = _load_torch_checkpoint(torch, checkpoint)
    args = checkpoint_data["args"]
    cls = {"NISQA": nl.NISQA, "NISQA_DIM": nl.NISQA_DIM}[args["model"]]
    model = cls(**_model_args(args))
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    device = torch.device(device_name)
    return model.to(device).eval(), device


def _sync_torch(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _schedule(
    paths: Sequence[Path],
    cfg: Any,
    td: str,
    batch_size: int,
    batch_mode: str,
    length_bucket: int | None,
):
    if not paths:
        raise ValueError("cannot schedule an empty path list")
    estimates = [estimate_n_wins(path, cfg) for path in paths]
    order = sorted(range(len(paths)), key=lambda idx: estimates[idx], reverse=True)
    effective_bs = min(batch_size, len(paths))
    if length_bucket is None:
        length_bucket = 32 if td == "self_att" else 64
    if batch_mode == "cost_aware":
        chunks = _cost_aware_chunks(order, estimates, effective_bs, exponent=_cost_exponent(td))
    else:
        chunks = [(chunk, effective_bs) for chunk in _fixed_chunks(order, effective_bs)]
    return chunks, estimates, length_bucket


def _assemble_batch(prepared: Sequence[tuple[np.ndarray, np.ndarray]], cfg: Any, requested_bs: int, bucket: int):
    live_n = np.asarray([int(item[1]) for item in prepared], dtype=np.int32)
    max_steps = min(_round_up(int(live_n.max()), bucket), cfg.max_segments)
    padded = list(prepared)
    if len(padded) < requested_bs:
        last_seg = padded[-1][0][:1]
        padded.extend((last_seg, np.asarray(1, dtype=np.int32)) for _ in range(requested_bs - len(padded)))
        all_n = np.concatenate([live_n, np.ones(requested_bs - len(prepared), dtype=np.int32)])
    else:
        all_n = live_n
    x = np.zeros((len(padded), max_steps, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
    for index, (segments, n_wins) in enumerate(padded):
        x[index, : int(n_wins)] = segments[: int(n_wins)]
    return x, all_n, len(prepared), max_steps


def _diff_metrics(jax_out: np.ndarray, torch_out: np.ndarray, tolerance: float) -> dict[str, Any]:
    if jax_out.shape != torch_out.shape:
        return {
            "shape_equal": False,
            "jax_shape": list(jax_out.shape),
            "torch_shape": list(torch_out.shape),
            "passed": False,
        }
    diff = np.abs(jax_out.astype(np.float64) - torch_out.astype(np.float64))
    finite = np.isfinite(diff).all()
    max_abs = float(diff.max()) if diff.size else 0.0
    return {
        "shape_equal": True,
        "finite": bool(finite),
        "max_abs": max_abs,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "p95_abs": float(np.percentile(diff, 95)) if diff.size else 0.0,
        "p99_abs": float(np.percentile(diff, 99)) if diff.size else 0.0,
        "tolerance": tolerance,
        "passed": bool(finite and max_abs <= tolerance),
    }


def _run_engine(
    kind: str,
    model: Any,
    paths: Sequence[Path],
    *,
    cfg: Any,
    td: str,
    batch_size: int,
    batch_mode: str,
    length_bucket: int | None,
    preprocess_workers: int,
    torch: Any | None = None,
    torch_device: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    chunks, estimates, bucket = _schedule(paths, cfg, td, batch_size, batch_mode, length_bucket)
    predictions: list[np.ndarray | None] = [None] * len(paths)
    shape_counts: Counter[str] = Counter()
    stats = {
        "engine": kind,
        "batch_size": batch_size,
        "batch_mode": batch_mode,
        "length_bucket": bucket,
        "preprocess_workers": preprocess_workers,
        "file_count": len(paths),
        "batch_count": len(chunks),
        "n_wins_min": int(min(estimates)),
        "n_wins_max": int(max(estimates)),
        "n_wins_mean": float(np.mean(estimates)),
        "preprocess_wait_seconds": 0.0,
        "preprocess_worker_seconds": 0.0,
        "input_transfer_seconds": 0.0,
        "forward_seconds": 0.0,
        "first_forward_seconds": 0.0,
        "first_forward_calls": 0,
        "warmed_forward_seconds": 0.0,
        "warmed_forward_calls": 0,
        "output_transfer_seconds": 0.0,
        "real_segments": 0,
        "padded_segments": 0,
        "total_seconds": 0.0,
    }
    seen_shapes: set[tuple[int, int]] = set()
    total_start = time.perf_counter()

    def submit(executor: ThreadPoolExecutor | None, chunk: list[int]):
        if executor is None:
            return [_preprocess_timed(paths[index], cfg) for index in chunk]
        return [executor.submit(_preprocess_timed, paths[index], cfg) for index in chunk]

    executor = ThreadPoolExecutor(max_workers=preprocess_workers) if preprocess_workers > 1 else None
    try:
        futures_or_values = submit(executor, chunks[0][0])
        for chunk_number, (chunk, requested_bs) in enumerate(chunks):
            wait_start = time.perf_counter()
            if executor is None:
                prepared_with_times = futures_or_values
            else:
                prepared_with_times = [future.result() for future in futures_or_values]
            stats["preprocess_wait_seconds"] += time.perf_counter() - wait_start
            prepared = [item[0] for item in prepared_with_times]
            stats["preprocess_worker_seconds"] += sum(item[1] for item in prepared_with_times)
            if chunk_number + 1 < len(chunks):
                futures_or_values = submit(executor, chunks[chunk_number + 1][0])

            x, n_wins, real_bs, max_steps = _assemble_batch(prepared, cfg, requested_bs, bucket)
            stats["real_segments"] += int(n_wins[:real_bs].sum())
            stats["padded_segments"] += int(n_wins.shape[0] * max_steps)
            shape_key = (int(x.shape[0]), max_steps)
            shape_counts[str(shape_key)] += 1

            if kind == "jax":
                transfer_start = time.perf_counter()
                x_dev, n_dev = model.device_segments(x, n_wins, padded_steps=max_steps)
                x_dev.block_until_ready()
                n_dev.block_until_ready()
                stats["input_transfer_seconds"] += time.perf_counter() - transfer_start
                forward_start = time.perf_counter()
                output = model._forward(model._compute_params, x_dev, n_dev)
                output.block_until_ready()
                elapsed = time.perf_counter() - forward_start
                host_start = time.perf_counter()
                output_np = np.asarray(output)
                stats["output_transfer_seconds"] += time.perf_counter() - host_start
            else:
                assert torch is not None and torch_device is not None
                _sync_torch(torch, torch_device)
                transfer_start = time.perf_counter()
                x_dev = torch.from_numpy(x).to(torch_device)
                n_dev = torch.from_numpy(n_wins).to(torch_device)
                _sync_torch(torch, torch_device)
                stats["input_transfer_seconds"] += time.perf_counter() - transfer_start
                forward_start = time.perf_counter()
                with torch.inference_mode():
                    output = model(x_dev, n_dev)
                _sync_torch(torch, torch_device)
                elapsed = time.perf_counter() - forward_start
                host_start = time.perf_counter()
                output_np = output.detach().cpu().numpy()
                stats["output_transfer_seconds"] += time.perf_counter() - host_start

            stats["forward_seconds"] += elapsed
            if shape_key in seen_shapes:
                stats["warmed_forward_seconds"] += elapsed
                stats["warmed_forward_calls"] += 1
            else:
                seen_shapes.add(shape_key)
                stats["first_forward_seconds"] += elapsed
                stats["first_forward_calls"] += 1
            for row, original_index in enumerate(chunk):
                predictions[original_index] = np.asarray(output_np[row], dtype=np.float32)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    stats["total_seconds"] = time.perf_counter() - total_start
    stats["shape_counts"] = dict(shape_counts)
    stats["shape_count"] = len(shape_counts)
    stats["padded_to_real_segment_ratio"] = stats["padded_segments"] / stats["real_segments"]
    stats["files_per_second"] = len(paths) / stats["total_seconds"]
    stats["model_forward_files_per_second"] = len(paths) / stats["forward_seconds"]
    if kind == "torch" and torch_device is not None and getattr(torch_device, "type", None) == "cuda":
        stats["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(torch_device))
        stats["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(torch_device))
    return np.stack([prediction for prediction in predictions if prediction is not None]), stats


def _preprocess_timed(path: Path, cfg: Any) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    start = time.perf_counter()
    result = preprocess_file(path, cfg)
    return result, time.perf_counter() - start


def _profile_jax(
    model: Any,
    paths: Sequence[Path],
    *,
    args: argparse.Namespace,
    output_path: Path,
    model_name: str,
) -> dict[str, Any]:
    profile_path = output_path.with_name(f"{output_path.stem}.{model_name}.profile.txt")
    profiler = cProfile.Profile()
    selected = list(paths[: args.profile_samples])
    started = time.perf_counter()
    profiler.enable()
    _, profile_stats = _run_engine(
        "jax",
        model,
        selected,
        batch_size=args.batch_size,
        batch_mode=args.batch_mode,
        cfg=model.config.feature,
        td=model.config.td,
        length_bucket=args.length_bucket,
        preprocess_workers=1,
    )
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    with profile_path.open("w") as handle:
        handle.write(f"profile_seconds={time.perf_counter() - started:.6f}\n")
        handle.write(f"sample_count={len(selected)}\n\n")
        stats.stream = handle
        stats.print_stats(40)
    return {
        "sample_count": len(selected),
        "wall_seconds": time.perf_counter() - started,
        "profile_file": profile_path.name,
        "stage_metrics": profile_stats,
    }


def _load_jax_model(name: str, device: str, precision: str):
    from nisqa_jax import load_model

    return load_model(WEIGHTS_DIR / MODEL_FILES[name], device=device, precision=precision)


def _load_environment(torch: Any, jax: Any, datasets_version: str | None) -> dict[str, Any]:
    import librosa
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": getattr(jax, "__version__", None),
        "jaxlib": getattr(__import__("jaxlib"), "__version__", None),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
        "scipy": scipy.__version__,
        "datasets": datasets_version,
        "nvidia": _nvidia_identity(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIGS, help="comma-separated configurations streamed in order")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--download-mode",
        choices=["download_shards", "remote_stream"],
        default="download_shards",
        help="download selected Parquet shards then stream locally, or use remote HF streaming",
    )
    parser.add_argument("--revision", default=None, help="Hub revision; resolved commit is always recorded")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--decode-threads", type=int, default=1)
    parser.add_argument(
        "--max-self-att-wins",
        type=int,
        default=1300,
        help="reject streamed audio exceeding this common self-attention window limit; <=0 disables",
    )
    parser.add_argument("--models", default="all", help="comma-separated names or all")
    parser.add_argument("--jax-device", default="gpu")
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--precision", choices=["float32", "bf16"], default="float32")
    parser.add_argument("--torch-source-root", type=Path, required=True)
    parser.add_argument("--torch-weights-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-mode", choices=["fixed", "cost_aware"], default="cost_aware")
    parser.add_argument("--length-bucket", type=int, default=None)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--correctness-samples", type=int, default=64)
    parser.add_argument("--profile-samples", type=int, default=64)
    parser.add_argument("--profile-model", default="all", help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.samples < 1 or args.batch_size < 1 or args.correctness_samples < 1 or args.profile_samples < 1:
        raise ValueError("samples, batch-size, correctness-samples, and profile-samples must be >= 1")
    if args.shuffle_buffer < 1 or args.decode_threads < 0 or args.preprocess_workers < 1:
        raise ValueError("shuffle-buffer/preprocess-workers must be >= 1 and decode-threads must be >= 0")
    if args.precision != "float32" and args.torch_device != "none":
        raise ValueError("PyTorch comparison requires --precision float32")
    model_names = _parse_models(args.models)
    args.profile_model = "all" if len(model_names) != 1 else model_names[0]
    if args.torch_weights_dir is None:
        args.torch_weights_dir = args.torch_source_root / "weights"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    owned_work_dir = args.work_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="nisqa-jax-hf-")) if owned_work_dir else args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _set_hf_cache(work_dir)
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "hf_real_audio_framework_comparison",
        "git": _git_identity(),
        "arguments": {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "download_mode": args.download_mode,
            "samples": args.samples,
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "decode_threads": args.decode_threads,
            "max_self_att_wins": args.max_self_att_wins,
            "models": model_names,
            "jax_device": args.jax_device,
            "torch_device": args.torch_device,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "batch_mode": args.batch_mode,
            "length_bucket": args.length_bucket,
            "preprocess_workers": args.preprocess_workers,
            "correctness_samples": args.correctness_samples,
            "profile_samples": args.profile_samples,
        },
        "environment": None,
        "dataset": None,
        "models": {},
        "scratch": {"temporary": True},
    }

    try:
        paths, result["dataset"] = _collect_streamed_audio(args, work_dir)
        import datasets
        import jax
        import torch

        _load_runtime_helpers()
        result["environment"] = _load_environment(torch, jax, getattr(datasets, "__version__", None))
        source_root = args.torch_source_root.resolve()

        for name in model_names:
            jax_checkpoint = WEIGHTS_DIR / MODEL_FILES[name]
            torch_checkpoint = args.torch_weights_dir / MODEL_TAR_FILES[name]
            metadata = json.loads(jax_checkpoint.with_suffix(".json").read_text())
            source_sha = _sha256(torch_checkpoint)
            if source_sha != metadata["source_sha256"]:
                raise ValueError(
                    f"{name}: PyTorch source checkpoint SHA-256 {source_sha} does not match "
                    f"converted artifact metadata {metadata['source_sha256']}"
                )
            model_result: dict[str, Any] = {
                "jax_checkpoint": jax_checkpoint.name,
                "jax_checkpoint_sha256": _sha256(jax_checkpoint),
                "torch_checkpoint": torch_checkpoint.name,
                "torch_checkpoint_sha256": source_sha,
                "source_sha256_match": True,
            }
            print(f"[{name}] loading JAX {args.jax_device} and PyTorch {args.torch_device}", flush=True)
            jax_model = _load_jax_model(name, args.jax_device, args.precision)
            torch_model, torch_device = _load_torch_model(torch, source_root, torch_checkpoint, args.torch_device)
            if getattr(torch_device, "type", None) == "cuda":
                torch.cuda.reset_peak_memory_stats(torch_device)

            jax_output, jax_metrics = _run_engine(
                "jax",
                jax_model,
                paths,
                cfg=jax_model.config.feature,
                td=jax_model.config.td,
                batch_size=args.batch_size,
                batch_mode=args.batch_mode,
                length_bucket=args.length_bucket,
                preprocess_workers=args.preprocess_workers,
            )
            torch_output, torch_metrics = _run_engine(
                "torch",
                torch_model,
                paths,
                cfg=jax_model.config.feature,
                td=jax_model.config.td,
                batch_size=args.batch_size,
                batch_mode=args.batch_mode,
                length_bucket=args.length_bucket,
                preprocess_workers=args.preprocess_workers,
                torch=torch,
                torch_device=torch_device,
            )
            model_result["jax"] = jax_metrics
            model_result["torch"] = torch_metrics
            model_result["gpu_output_comparison"] = _diff_metrics(jax_output, torch_output, 5e-5)
            model_result["gpu_output_comparison"]["interpretation"] = (
                "diagnostic only; PyTorch cuDNN LSTM can accumulate gates in a different order"
                if name == "nisqa_tts"
                else "diagnostic output comparison"
            )

            profile = _profile_jax(jax_model, paths, args=args, output_path=args.output, model_name=name)
            model_result["profile"] = profile
            result["models"][name] = model_result
            _write_json(args.output, result)
            del torch_model, jax_model
            if getattr(torch_device, "type", None) == "cuda":
                torch.cuda.empty_cache()
            jax.clear_caches()

        # Correctness is a separate CPU-vs-CPU check. This avoids treating cuDNN's
        # documented GPU LSTM accumulation order as a port failure. It compares
        # real audio-derived segment tensors, not synthetic inputs.
        correctness_paths = paths[: args.correctness_samples]
        for name in model_names:
            print(f"[{name}] CPU correctness on {len(correctness_paths)} real samples", flush=True)
            jax_model = _load_jax_model(name, "cpu", "float32")
            torch_checkpoint = args.torch_weights_dir / MODEL_TAR_FILES[name]
            torch_model, torch_device = _load_torch_model(torch, source_root, torch_checkpoint, "cpu")
            jax_output, jax_metrics = _run_engine(
                "jax",
                jax_model,
                correctness_paths,
                cfg=jax_model.config.feature,
                td=jax_model.config.td,
                batch_size=args.batch_size,
                batch_mode=args.batch_mode,
                length_bucket=1,
                preprocess_workers=1,
            )
            torch_output, torch_metrics = _run_engine(
                "torch",
                torch_model,
                correctness_paths,
                cfg=jax_model.config.feature,
                td=jax_model.config.td,
                batch_size=args.batch_size,
                batch_mode=args.batch_mode,
                length_bucket=1,
                preprocess_workers=1,
                torch=torch,
                torch_device=torch_device,
            )
            comparison = _diff_metrics(jax_output, torch_output, 5e-5)
            result["models"][name]["cpu_correctness"] = {
                "sample_count": len(correctness_paths),
                "comparison": comparison,
                "jax_metrics": jax_metrics,
                "torch_metrics": torch_metrics,
                "reference": "upstream PyTorch CPU model with exact source checkpoint",
            }
            _write_json(args.output, result)
            del torch_model, jax_model
            jax.clear_caches()

        result["completed"] = True
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_value))
        return 0
    finally:
        if owned_work_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
