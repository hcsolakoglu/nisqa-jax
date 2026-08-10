# Minds14 benchmark optimization comparison

Both runs used the same deterministic selection:

- dataset: `PolyAI/minds14`
- split: `train`
- requested configurations: `en-US`, `en-GB`, `de-DE`, `fr-FR`, `es-ES`
- contributing configurations at the resolved revision: `en-US`, `en-GB`,
  `de-DE`, `fr-FR` (`es-ES` yielded no rows)
- requested valid examples: `2000`
- rejected examples over the common self-attention limit: `3`
- manifest SHA-256: `d9f74a260b487e6785b46df939557904697df52385008863c3be19282650f94c`
- duration distribution: minimum `1.621375 s`, median `7.338625 s`, mean
  `9.0693585625 s`, p95 `19.91 s`, maximum `45.41825 s`
- audio sample rate: `8 kHz` for all selected examples
- batch size: `4`
- batch mode: `cost_aware`
- profile subset: `64` examples per model

The baseline is [`hf-minds14-2k.baseline.json`](hf-minds14-2k.baseline.json).
The optimized result is [`hf-minds14-2k.json`](hf-minds14-2k.json). Raw
cProfile output is stored beside both result sets.

## Change

`nisqa_jax.features.load_melspec` now computes the same Librosa frontend as an
explicit STFT followed by a cached mel filter bank. The filter bank is keyed by
sample rate and frontend parameters and has a bounded LRU cache of 32 entries.
This removes repeated filter-bank construction across files without changing
STFT framing, windowing, mel normalization, dB conversion, or output dtype.

The frozen frontend and score regression suite remained green: five tests
passed. A direct equivalence probe on the same generated WAV produced zero
maximum and mean absolute difference before the frozen regression run.

## End-to-end measurements

Wall times are single-run measurements on the same machine. They are evidence,
not a universal framework ranking. Negative values mean the optimized run was
faster.

| Model | JAX baseline | JAX optimized | JAX change | PyTorch baseline | PyTorch optimized | PyTorch change |
|---|---:|---:|---:|---:|---:|---:|
| `nisqa_mos_only` | 131.509 s | 112.068 s | -14.8% | 97.778 s | 82.684 s | -15.4% |
| `nisqa` | 118.243 s | 128.289 s | +8.5% | 100.995 s | 124.916 s | +23.7% |
| `nisqa_tts` | 292.556 s | 279.980 s | -4.3% | 127.877 s | 126.816 s | -0.8% |

The `nisqa` end-to-end increase shows why these runs do not justify a general
speedup claim. GPU scheduling, CPU contention, JAX compilation, and the
single-run measurement introduce noise larger than the frontend change for
some configurations.

## Profile evidence

The isolated JAX cProfile subset measured these cumulative frontend times for
`load_melspec`:

| Model | Baseline `load_melspec` | Optimized `load_melspec` | Change |
|---|---:|---:|---:|
| `nisqa_mos_only` | 4.909 s | 1.604 s | -67.3% |
| `nisqa` | 2.488 s | 3.007 s | +20.9% |
| `nisqa_tts` | 4.208 s | 2.548 s | -39.5% |

The isolated profile is also subject to process and filesystem noise. The
consistent result is structural: optimized profiles no longer contain the
repeated `librosa.feature.melspectrogram` call or repeated mel-basis creation.
The existing full frozen frontend test is the correctness authority, not the
wall-time difference of one cProfile process.

## Correctness

CPU JAX versus upstream PyTorch comparison passed for every model over the
64-sample real-audio subset:

| Model | max absolute difference | strict tolerance | result |
|---|---:|---:|---|
| `nisqa_mos_only` | `1.6689301e-6` | `5e-5` | passed |
| `nisqa` | `2.8610229e-6` | `5e-5` | passed |
| `nisqa_tts` | `2.1457672e-6` | `5e-5` | passed |

All full CUDA comparisons had equal shapes and finite outputs, but exceeded the
strict CPU-oriented diagnostic threshold:

| Model | CUDA max absolute difference | strict threshold | interpretation |
|---|---:|---:|---|
| `nisqa_mos_only` | `8.6832047e-4` | `5e-5` | diagnostic drift |
| `nisqa` | `1.9052029e-3` | `5e-5` | diagnostic drift |
| `nisqa_tts` | `2.0222664e-3` | `5e-5` | diagnostic drift; cuDNN LSTM accumulation order |

These CUDA comparisons are deliberately not labelled as correctness passes.
The CPU comparison is the correctness gate because the upstream CUDA path uses
cuDNN LSTM accumulation that can differ from the JAX recurrence while remaining
finite and shape-compatible.

## Decision

Keep the cached mel-basis implementation. It preserves frozen frontend and
score outputs and reduces a repeated frontend allocation path. Do not claim a
universal end-to-end speedup from this two-run sample. Further speed work should
focus on JAX compilation and shape reuse, especially `nisqa_tts`, where the
full run spent approximately `197.5 s` of its baseline `292.6 s` in first
forward compilation calls.
