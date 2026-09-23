# PocketLM

**A from-scratch runtime for running large language models locally on limited hardware.**

> 🚧 **Active development.** PocketLM is a work in progress. The core runtime works and
> is benchmarked, but performance, model coverage, and the app are still changing.

PocketLM imports, validates, and runs open-weight models on ordinary Windows machines,
with no cloud and often no GPU. It picks the right execution backend for each model,
streams weights from disk when a model is too large for RAM, and checks every speed
claim against its own runtime counters.

To stress-test it, PocketLM runs **DeepSeek-V3 (671B parameters) on a CPU-only laptop**.
The weights take about 1.3 TB of disk, many times the machine's RAM.

## Progress so far

These are measured on the same consumer laptop, and every run passed PocketLM's
anti-cheat checks: every layer ran, on the stated model and quantization.

| What improved | Before | Now | Speedup |
|---|---:|---:|---:|
| **DeepSeek-V3 671B generation**, full 61 layers, CPU-only | ~120 s/token | **34.23 s/token** | **~3.5×** |
| **Qwen2.5-14B chat**, from the direct runtime to the managed GGUF backend | ~20 s/token | **0.35 s/token** | **~57×** |
| Repeated-prefix reuse, first visible token | 87.09 s | **0.94 s** | **92×** |
| Full `lm_head` cache, cached tail | 17.59 s | **1.61 s** | **10.9×** |
| Native C++ kernels vs. Python fallback, full 62-layer pass | 800.55 s | **307.89 s** | **2.6×** |

Each optimization became the default only after a full-model proof. Optimizations
that turned out slower were rejected and are listed in [BENCHMARKS.md](BENCHMARKS.md).
These are current numbers, not final ones.

![PocketLM desktop](docs/images/desktop.png)

## Highlights

- **Multiple model families.** Qwen, Qwen MoE, Mixtral, DeepSeek-V3, Gemma 3, and
  Kimi K2, plus a Kronos time-series forecasting adapter. It imports Hugging Face,
  safetensors, PyTorch, and GGUF sources.
- **Automatic backend selection.** A capability-based engine selector chooses between
  the native direct runtime, a managed `llama.cpp` / GGUF server, and the FP8
  weight-streaming runtime. It uses the model's architecture, quantization, free RAM,
  and installed backends to decide.
- **Native C++ kernels.** 14 hand-written translation units (about 6.6k lines) cover
  FP8 and FP16 linear layers, AVX-512 FP8 mixture-of-experts, attention, KV cache,
  packed GEMV, and Q4 dequantization. Each kernel has a kill switch and a Python
  fallback, so its contribution can be measured.
- **Out-of-core execution.** A persisted tensor catalog and execution plan stream
  weights through hot and warm residency windows. Cached weights are checked with
  checksums, and residency survives restarts.
- **Speculative decoding.** Multi-token-prediction drafts are verified in batched
  tree sweeps. The full model still decides every committed token.
- **Anti-cheat benchmarking.** A result is not recorded unless every expected layer
  actually ran, with the stated model, quantization, and strategy.
- **Local app.** A web UI and desktop status screen offer chat, model loading,
  benchmarks, backend comparison, and an installation health check. Everything stays
  on the machine.

## Current results by model

| Model | Backend | Current result |
|---|---|---|
| Qwen2.5-14B-Instruct (Q4_K_M) | managed GGUF server | **0.35 s/token** (2.8 tok/s) warm, about 9 GB RAM |
| Qwen2.5-14B-Instruct | native direct runtime | about 20 s/token, correct full-stack output |
| DeepSeek-V3 671B (FP8, 61 layers) | FP8 streaming + MTP speculative decoding | **34.2 s/token**, 10 of 10 draft tokens accepted in one verifier sweep |

The DeepSeek result is a systems benchmark, not yet a usable chat experience. It shows
that a model far larger than RAM runs correctly on a laptop, and that it keeps getting
faster through measured optimizations. For everyday chat, PocketLM uses the fastest
backend that a model supports. Full history and reproduction commands are in
**[BENCHMARKS.md](BENCHMARKS.md)**.

## Status and next steps

PocketLM is far from finished. Work in progress includes:

- **DeepSeek-V3 speed:** cutting per-token time further, targeting the attention and
  expert-loading costs that dominate each pass.
- **GPU path:** turning the single-layer GPU probe into a full end-to-end GPU run.
- **Model coverage:** moving Gemma 3, Kimi K2, and Kronos from fixture-checked to proven
  with real local weights, and bringing MoE models up to Qwen's level.
- **Direct runtime:** reducing tensor I/O and copying so the custom runtime closes the
  gap with the GGUF backend.
- **App and install:** validating installs on more Windows machines and polishing the
  UI.

## Architecture

```text
            ┌──────────────── Web UI / Desktop / CLI tools ────────────────┐
            │  chat · model loading · benchmarks · installation health     │
            └───────────────────────────────┬──────────────────────────────┘
                                            │
   Model import ──► Registry & profiles ──► Engine selector ──► Supervisor / proofs
 (HF, safetensors,   (immutable originals,   (capabilities, RAM,
  PyTorch, GGUF)      reversible artifacts)   installed backends)
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          ▼                                 ▼                                 ▼
  Native direct runtime            GGUF / llama.cpp server           FP8 streaming runtime
  dense & MoE, tensor residency,   managed lifecycle, fastest        out-of-core weights,
  prefix / response reuse          interactive path                  native FP8 kernels, MTP
```

### How a model larger than RAM runs

1. **Staged weight streaming.** The model is split into units that are scheduled
   across a *hot* window and a *warm* prefetch window. Units rotate from warm to hot
   and refill from an overflow head.
2. **Native FP8 kernels.** Computation runs on the original FP8 weights, with no
   down-conversion. Turning off the native kernels slows a full 62-layer pass from
   308 s to 801 s.
3. **Caching.** Reusing an exact repeated prefix cuts the first visible token from
   87 s to 0.9 s. A full `lm_head` cache and a 4 GB FP8 attention cache also help.
4. **Speculative decoding.** DeepSeek-V3's own MTP head proposes a depth-10 token
   tree. The full 61-layer model accepts all 10 tokens in a single verifier sweep.

### Anti-cheat verification

Speed claims for large models on small machines are easy to fake: skip layers, drop
to a smaller quantization, swap in a smaller model, or offload to a GPU. The runtime
therefore records its own work on every proof run:

- `layers_executed` must equal `expected_layers_executed`
- each proof records `model_id`, quantization, expert routing, and `strategy`
- the verifier contract enforces one weight sweep per pass
- any blocker invalidates the proof

An optimization becomes the default only when `anti_cheat_passed` is true and
`blockers` is empty.

## Model support

| Family | Status | Backend |
|---|---|---|
| Qwen 2.5 / 3 (dense) | Working: real local chat and benchmarks | direct, GGUF |
| DeepSeek-V3 | Working: full 61-layer runtime proof, still being optimized | FP8 streaming |
| Qwen MoE, Mixtral | Experimental: paged-expert runtime | direct |
| Gemma 3, Kimi K2 | Chat contract checked against fixtures | GGUF |
| Kronos | Forecasting adapter checked against fixtures (not a chat model) | CPU |

A family is marked *working* only after real weights have gone through import,
readiness checks, runtime, and benchmarks on local hardware.

## Getting started

Requires Python 3.12+ on Windows. **Model weights are never included**; you supply
them locally.

```powershell
.\install-pocketlm.ps1 -StartApp   # isolated venv + first-run health check
.\start-pocketlm.ps1               # relaunch later
```

The installer creates an isolated environment under `%LOCALAPPDATA%\PocketLM\venvs`
and writes a sanitized installation report to `state\supervisor\install-proof.json`.
With no models installed, the app reports a `partial` state. Once a runnable model is
available, it reports `ready`.

### Development

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pytest -q
python tools/build_native.py --force   # native kernels (requires local OpenBLAS)
```

The full suite covers 81 test modules, including numerical checks of the native
kernels against reference implementations. GitHub Actions installs the project and
runs the portable contract tests on a clean Windows runner for every push.

### Local API

While running, the app serves local-only endpoints on `127.0.0.1:8765`, for example:

- `GET /api/status`: runtime, backend, and model state
- `POST /api/chat`: chat with the selected backend
- `POST /api/benchmark/comparison`: compare backends on the same prompt
- `GET /api/supervisor/health`: sanitized installation health report

Health reports use a random installation ID and contain no usernames, hostnames,
paths, prompts, or model contents. Nothing is uploaded.

## Project layout

| Path | Contents |
|---|---|
| `src/pcketlm/core/runtime/` | engine selector, direct and GGUF backends, weight streaming, speculative decoding |
| `src/pcketlm/core/model_import/` | model intake, inspection, Q4 and FP8 packing |
| `src/pcketlm/core/{registry,profiles,benchmark,validation}/` | model registry, runtime profiles, benchmark runs, validation |
| `src/pcketlm/native/` | C++ kernels (FP8/FP16 linear, AVX-512 MoE, attention, KV cache) |
| `src/pcketlm/app/` | web UI, desktop status screen, CLI tools |
| `tools/` | benchmarking, diagnostics, weight packing, native build |
| `tests/` | unit, contract, and kernel correctness tests |
| `docs/` | design documents, GPU testing notes, privacy manifest |

## Documentation

- [BENCHMARKS.md](BENCHMARKS.md): measured results and optimization history
- [docs/design/](docs/design/): blueprint, roadmap, schemas, risk register
- [docs/gpu-cloud-testing.md](docs/gpu-cloud-testing.md): GPU validation on cloud notebooks
- [docs/privacy.md](docs/privacy.md): what is excluded from this repository

## License

[MIT](LICENSE)
