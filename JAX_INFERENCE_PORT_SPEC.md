# JAX Inference Port Specification

## Summary

Port only the inference path for the three shipped checkpoints:

- `nisqa.tar`: `NISQA_DIM`, adaptive CNN + self-attention + five attention-pooling heads.
- `nisqa_mos_only.tar`: `NISQA`, adaptive CNN + self-attention + one attention-pooling head.
- `nisqa_tts.tar`: `NISQA`, standard CNN + bidirectional LSTM + last-step-bi pooling.

Keep Librosa-compatible CPU feature extraction in v1 for exact input parity, then run the model forward pass in JAX on a single NVIDIA GPU. Do not port training, finetuning, evaluation metrics, or the unshipped double-ended `NISQA_DE` model.

References for implementation choices: JAX recommends outermost `jit` for performance and `block_until_ready()` for benchmarks, plus explicit device placement to avoid transfer skew ([JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)); persistent compilation cache should be enabled before first compile ([JAX cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)); transfer guard can catch accidental host/device transfers ([JAX transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html)); recent GPU performance guidance includes matmul precision and XLA flags with version-dependent behavior ([JAX GPU tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)). The v1 implementation should stay pure JAX unless Flax removes concrete complexity; this keeps checkpoint parity and dependency management simpler.

## Public API And Packaging

Create a new JAX inference package alongside the PyTorch code, e.g. `nisqa_jax/`, with these public entrypoints:

- `load_model(checkpoint_path: str | Path, *, device: str | None = None, cache_dir: str | Path | None = None) -> NisqaJaxModel`
  - Loads either a standalone converted JAX `.npz` artifact with sidecar JSON metadata or, when optional PyTorch is installed, an original PyTorch `.tar` checkpoint.
  - The standalone `.npz` path is the normal inference path and does not require the original PyTorch source checkout.
- `predict_file(model, wav_path, *, channel=None) -> dict[str, float]`
  - Returns `{mos, noi, dis, col, loud}` for `nisqa.tar`, `{mos}` for `nisqa_mos_only.tar`, and `{naturalness}` or `{mos}` with documented alias for `nisqa_tts.tar`.
- `predict_batch(model, wav_paths, *, batch_size=..., channel=None) -> pandas.DataFrame`
  - Uses Librosa feature extraction on host, pads to model-specific static max segments, batches by compatible shape.
- CLI parity command: `python -m nisqa_jax.predict --mode predict_file|predict_dir|predict_csv --pretrained_model ...`
  - Match existing PyTorch arguments where practical: `--deg`, `--data_dir`, `--csv_file`, `--csv_deg`, `--output_dir`, `--bs`, `--ms_channel`.

Use typed immutable config objects for checkpoint args, model architecture, and feature extraction settings. Fail fast on unsupported architectures, unexpected state keys, missing weights, mismatched tensor shapes, samples shorter than `ms_seg_length`, or samples longer than `ms_max_segments`.

## Implementation Changes

Implement inference as pure JAX functions with explicit param pytrees and a thin model wrapper. Avoid a mandatory Flax dependency in v1; keep core computations functional and easy to compare against PyTorch.

- Audio/frontend:
  - Keep existing Librosa settings exactly: `n_fft=4096`, `hop_length=int(sr*0.01)`, `win_length=int(sr*0.02)`, `power=1.0`, `amplitude_to_db(ref=1.0, amin=1e-4, top_db=80)`.
  - Preserve checkpoint-specific `fmax`, `seg_hop_length`, `ms_max_segments`, `cnn_model`, and output dimensions.
  - Segment exactly like PyTorch: input `[mel, time]` to `[windows, 1, mel, seg_length]`, then pad to `ms_max_segments`.

- JAX model kernels:
  - CNN: implement `conv_general_dilated` in NHWC internally for GPU efficiency; convert PyTorch OIHW conv weights to HWIO once during conversion.
  - BatchNorm: inference-only affine normalization using PyTorch `running_mean`, `running_var`, `weight`, `bias`, and epsilon matching PyTorch default.
  - Adaptive CNN pooling: implement exact PyTorch-compatible adaptive max pool for fixed target sizes `[24,7]`, `[12,5]`, `[6,3]`; validate against PyTorch layer outputs.
  - Standard CNN pooling: match PyTorch max-pool padding/stride behavior for TTS.
  - Self-attention: split PyTorch `in_proj_weight`/bias into q/k/v, use mask from `n_wins`, apply softmax with `-inf` masked positions, preserve layer norm and FFN order.
  - Attention pooling: implement masked attention pooling and one or five heads depending on checkpoint.
  - TTS LSTM: implement a custom bidirectional LSTM with `jax.lax.scan`, PyTorch gate order, and PyTorch bias semantics (`bias_ih + bias_hh`), then last-step-bi pooling.

- Weight conversion:
  - Provide deterministic converter from PyTorch checkpoint to JAX `.npz`/msgpack cache, with strict state-key accounting.
  - Convert linear weights from PyTorch `[out,in]` to JAX `[in,out]`; conv weights `[out,in,h,w]` to `[h,w,in,out]`.
  - Drop `num_batches_tracked`; preserve all trainable weights, BN running stats, layer norm stats, attention projections, pool heads, and LSTM forward/reverse weights.
  - Store converted artifact metadata: source checkpoint hash, conversion version, architecture args, tensor shape manifest.
  - Publish stable converted artifacts under top-level `weights/` for standalone JAX inference.

## Performance Requirements

Optimize for warmed single-GPU inference after feature extraction.

- Wrap the full batched forward pass in one outer `jax.jit`, with static model type and static padded segment length to avoid recompilation.
- Pre-place params and input batches with `jax.device_put`; avoid per-layer or per-sample host/device transfers.
- Enable persistent compilation cache via `JAX_COMPILATION_CACHE_DIR` or `cache_dir` before first compilation.
- Crop each batch to its actual maximum valid segment count before device transfer, then cache/JIT by `(model_id, batch_size, cropped_segments)`. This matches PyTorch packed-sequence behavior and avoids running CNN/LSTM work over padded windows.
- Benchmark with warmup excluded and `.block_until_ready()` included.
- Use `jax.transfer_guard("log")` in performance tests to detect accidental implicit transfers.
- Use `jax.default_matmul_precision("tensorfloat32")` on NVIDIA GPUs for speed only after parity tests confirm output drift stays within tolerance; keep strict float32 mode for conformance tests.
- Target acceptance: JAX warmed model-forward latency is at least 2x faster than PyTorch for the two self-attention checkpoints at batch sizes 8 and 16 on the same NVIDIA GPU. For `nisqa_tts.tar`, which compares against PyTorch/cuDNN LSTM, require materially faster warmed forward latency, with a target of at least 1.5x at batch 8 and 2x at batch 16. End-to-end speedup is reported separately because Librosa remains CPU-bound in v1.

## Validation Tests

Add a test suite that compares JAX against the existing PyTorch implementation at multiple levels.

- Checkpoint conversion tests:
  - All three checkpoints load and convert with no missing/unexpected keys except ignored `num_batches_tracked`.
  - Converted tensor shapes match expected manifests.
  - Re-converting the same checkpoint produces byte-stable metadata and numerically identical tensors.

- Unit parity tests:
  - Segment extraction matches PyTorch `segment_specs` exactly for synthetic mel arrays.
  - CNN block outputs match PyTorch after each conv/BN/pool group within `rtol=1e-4`, `atol=1e-4`.
  - Self-attention layer outputs match PyTorch masked attention within `rtol=2e-4`, `atol=2e-4`.
  - Attention pooling and five-head DIM pooling match within `rtol=1e-4`, `atol=1e-4`.
  - TTS bidirectional LSTM and last-step-bi pooling match within `rtol=5e-4`, `atol=5e-4`.

- Golden inference tests:
  - Generate deterministic WAV fixtures: silence, sine sweep, clipped sine, short valid speech-like noise, and a stereo file with channel selection.
  - Run PyTorch and JAX for all three checkpoints on the same fixtures.
  - Acceptance: final predictions match PyTorch within `max_abs_error <= 1e-3` for strict float32 mode.
  - CSV/dir/file modes preserve row order and output columns.

- Failure-mode tests:
  - Too-short files raise the same class of validation error with file path included.
  - Files exceeding `ms_max_segments` raise a clear error unless future truncation is explicitly added.
  - Unsupported `NISQA_DE` checkpoint reports "unsupported in JAX inference v1."
  - Missing/corrupt checkpoints and unsupported checkpoint args fail before compilation.

- Performance tests:
  - Compare PyTorch and JAX model-forward only on precomputed padded inputs for batch sizes 1, 8, and 16.
  - Report compile time, warm latency, throughput samples/sec, host-to-device transfer time, and end-to-end time.
  - CI marks performance as informational unless an NVIDIA GPU runner is available; GPU benchmark script enforces the 2x warmed-forward target.
  - Use `python -m nisqa_jax.bench_compare --pretrained_model ... --batch_size 8 --seq_len 128 --transfer_guard log` for local CUDA forward comparisons.

## Assumptions

- "Three models" means the three shipped checkpoint files, not `NISQA_DE`.
- Librosa remains the source of truth for features in v1; a JAX audio frontend is a later optimization after model parity is locked.
- Local validation pins JAX to `0.4.30` to keep NumPy at `1.26.x`, which is compatible with the existing pandas/librosa stack. GPU deployments should install the matching CUDA JAX wheel for the target driver.
- Strict parity tests run in float32 precision; faster TF32/bfloat16 options are opt-in and must pass relaxed drift checks before being enabled by default.
- Training, finetuning, dataset evaluation metrics, plotting, and double-ended inference are out of scope for this port.
