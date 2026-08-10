# Benchmarking

Performance claims are accepted only when they are reproducible, versioned, and
separate from correctness evidence.

## Tools

JAX-only model-forward measurement:

```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.bench_compare \
  --pretrained_model "$W" \
  --device gpu --precision float32 \
  --batch_size 8 --seq_len 128 --no_torch
```

Hash-bound JAX/PyTorch comparison:

```bash
python -m nisqa_jax.bench_compare \
  --pretrained_model "$W" \
  --torch_checkpoint /path/to/NISQA/weights/nisqa_mos_only.tar \
  --torch_source_root /path/to/NISQA \
  --device gpu --precision float32 \
  --batch_size 8 --seq_len 128 --steps 100
```

End-to-end preprocessing and inference:

```bash
python -m nisqa_jax.bench \
  --pretrained_model "$W" \
  --device gpu --precision float32 \
  --batch_size 8 --data_dir /path/to/wavs
```

## Current real-data benchmark

`scripts/benchmark_hf_real.py` benchmarks all three shipped checkpoints on the
same real audio files using the same length-aware scheduler for JAX and the
modern PyTorch working tree. It records preprocessing, input transfer, first
compiled forward, warmed forward, output transfer, padding overhead, peak GPU
memory, and framework output differences. It also runs a CPU PyTorch parity
subset and writes a cProfile report per model.

The committed run uses:

- Dataset: [`PolyAI/minds14`](https://huggingface.co/datasets/PolyAI/minds14),
  `train` split.
- Configurations requested: `en-US`, `en-GB`, `de-DE`, `fr-FR`, and `es-ES`.
  The resolved revision yielded rows from the first four; `es-ES` contributed no
  rows to this split.
- Selection: deterministic seed, bounded shuffle, two thousand valid examples,
  variable duration, and a common self-attention limit of `n_wins <= 1300`.
- Acquisition: only selected language Parquet shards are downloaded. The local
  Parquet files are then consumed with `streaming=True`; audio WAVs and the
  task-owned HF cache are deleted after the run by default.
- Dataset role: this is a runtime and numerical-parity corpus, not a MOS ground
  truth benchmark. Minds14 is real telephone speech and is commonly stored at
  8 kHz. It should not be used to claim NISQA perceptual-quality accuracy.
- Correctness: full two-thousand-sample JAX/PyTorch output comparison is
  diagnostic on the CUDA path. The CPU reference subset is the correctness gate,
  because PyTorch CUDA/cuDNN LSTM accumulation can introduce backend-specific
  drift.

Install optional benchmark dependencies into the benchmark environment:

```bash
python -m pip install -r requirements-benchmark.txt
# Install the CUDA-compatible PyTorch wheel separately for the target machine.
python -m pip install -e . --no-deps
```

Run the benchmark from the repository root:

```bash
python scripts/benchmark_hf_real.py \
  --torch-source-root /path/to/NISQA \
  --models all \
  --samples 2000 \
  --correctness-samples 64 \
  --profile-samples 64 \
  --batch-size 4 \
  --preprocess-workers 4 \
  --decode-threads 0 \
  --output docs/benchmarks/results/hf-minds14-2k.json
```

`--decode-threads 0` is intentional for local shards: the `datasets` threaded
decode path can leave an async task pending at shutdown when bounded selection
stops early. Serial decode is therefore the benchmark default; model
preprocessing workers remain parallel and independent.

The exact resolved Hugging Face revision, shard hashes, sample-manifest hash,
length distribution, environment, checkpoint hashes, raw timings, correctness
metrics, and profile filenames are stored in
[`hf-minds14-2k.json`](results/hf-minds14-2k.json). The adjacent `*.profile.txt`
files contain the Python-level profile for each model. The preserved pre-change
run is [`hf-minds14-2k.baseline.json`](results/hf-minds14-2k.baseline.json), and
[`hf-minds14-2k-optimization.md`](results/hf-minds14-2k-optimization.md)
explains the frontend optimization and before/after limits. The canonical
cross-framework v3 result is documented in
[`hf-minds14-2k-cross-framework.md`](results/hf-minds14-2k-cross-framework.md)
and keeps one raw JSON plus one cProfile artifact per shipped checkpoint. No
user audio is committed.

The NISQA-specific `hewliyang/nisqa-vcc-mos` archive was considered but not used
for this bounded run. Its public layout is a multi-gigabyte tar archive, so
selecting two thousand members would require a substantially larger sequential
transfer rather than row-level HF streaming. It remains a candidate for a
separate MOS correlation study with an explicitly approved storage budget.

## Cross-repository benchmark contract

The framework comparison is a runtime and numerical-parity experiment, not a
quality leaderboard. A valid comparison must satisfy all of the following:

1. **One immutable manifest.** Select the same two thousand valid Minds14 rows
   once, record the resolved Hub revision, row identifiers, duration estimates,
   and manifest SHA-256, then feed that manifest to every model and framework.
2. **One checkpoint lineage.** Use the matching PyTorch `.tar` and converted
   JAX artifact, verify the converted metadata's source SHA-256, and record both
   repository commits plus dirty state. Do not compare a converted artifact with
   a different upstream checkpoint.
3. **Two explicit frontend lanes.** `native` measures each shipped product as
   users run it: PyTorch uses its own librosa/segment implementation and JAX
   uses its own frontend. `shared` feeds the same JAX-generated NumPy segments
   to both model cores and is reported separately. Neither lane may be silently
   substituted for the other.
4. **Same scheduling contract.** Use stable descending length order, the same
   batch size, the same padded length bucket, the same serial audio decode
   policy, and the same segment-budget mode. Final short batches are padded with
   masked dummy rows so both engines execute identical requested shapes.
5. **Separate timing stages.** Report frontend/preprocess, host-to-device input
   transfer, first-shape compilation or lookup, warmed forward, output transfer,
   and total wall time. Synchronize CUDA explicitly. A cold compile-inclusive
   total and a warmed steady-state total are different claims.
6. **Correctness gates.** Run CPU PyTorch-vs-JAX comparison on real audio-derived
   segments for all shipped checkpoints. Treat GPU output comparison as
   diagnostic for `nisqa_tts`, since cuDNN LSTM accumulation order can drift.
   Minds14 has no NISQA MOS ground truth in this experiment, so do not claim
   perceptual accuracy from it.

The canonical shipped-product command uses the modern PyTorch working tree as
`--torch-source-root`, `--frontend-mode native`, `--batch-mode fixed`, and
`--decode-threads 0`. On GPUs where JAX and PyTorch share one process, disable
XLA preallocation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/benchmark_hf_real.py ...
```

The result records these allocator settings. `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
is a memory-reclamation fallback, not a canonical performance setting: it can
materially increase JAX allocation overhead and must be reported as a separate
ablation. If a model still cannot fit both frameworks in one process, run that
model with a smaller common batch size and record the constraint explicitly.
`shared` frontend and `segment_budget` are controlled ablation modes, not
defaults. Every result JSON must retain these arguments so an apparent speedup
cannot be produced by changing preprocessing or scheduling between frameworks.

## Required evidence

Every report must record:

- repository commit;
- JAX, jaxlib, PyTorch, CUDA runtime, driver, and GPU;
- model, precision, batch sizes, sequence lengths, and input distribution;
- warmup and timed iteration counts;
- whether compilation and transfers are included;
- JAX allocator/preallocation settings when CUDA is used;
- explicit synchronization and aggregation method;
- raw machine-readable output.

Compilation, host-to-device transfer, warmed model-forward, and preprocessing
must be reported separately. A smoke test proves execution, not performance.
Do not combine measurements from different revisions or PyTorch baselines.

## Optimization policy

The current implementation already has stable sorting, cost-aware chunking,
bucket padding, bounded preprocessing workers, explicit synchronization, and
OOM retry behavior. An optimization is accepted only when a before/after
measurement on the same manifest shows a meaningful improvement and all CPU
parity, CUDA smoke, and regression gates remain green.

The real-data result records padding ratio and stage timings so optimization
work can distinguish frontend, transfer, compilation, and model-forward costs.
If no change beats the baseline without weakening a contract, the result should
say so explicitly instead of claiming a universal speedup.

## Historical evidence

Historical JAX 0.4.30 evidence is preserved under
[historical/](historical/README.md) for provenance only. It must not be mixed
with the current JAX 0.6.2 baseline or with a different PyTorch checkpoint.
