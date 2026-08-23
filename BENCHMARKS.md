# Benchmarks

Every number here comes from a JSON proof artifact committed under [`state/`](state/).
Nothing is estimated. Each artifact records the model, layer count, generated text,
timing, and anti-cheat counters for that run.

## Hardware

All CPU measurements are single-machine, consumer Windows hardware, **CPU-only** —
no GPU, no cloud offload, no API calls. DeepSeek-V3 weights are read in their
original FP8 format from local disk.

## Headline result

**DeepSeek-V3 (671B parameters, 61 transformer layers) generating on a consumer laptop CPU.**

| | |
|---|---|
| Model | `deepseek-v3`, FP8, 61 config layers |
| Prompt | `Write exactly ten short words about the sky.` |
| Output | `Blue, vast, endless, clouds, stars,` |
| Accepted tokens | 10 in **1** verifier sweep |
| Speed | **34.2292 s / visible token** |
| Layers executed | 122 / 122 expected |
| Anti-cheat | passed |

Proof: [`state/mtp-warm-prefill-mtphead-evict-before-verify-rank-onepass-top2048-depth10-full61-tenvisible.json`](state/mtp-warm-prefill-mtphead-evict-before-verify-rank-onepass-top2048-depth10-full61-tenvisible.json)

## The optimization path

Each row is a promoted default. Every promotion required an exact proof on the full
61-layer model with the same prompt and 10 visible tokens — no shortcuts between rows.

| Stage | s / visible token | Proof artifact |
|---|---:|---|
| First batched MTP verifier | 569.11 | `plm-20-mtp-batched-full61-tenvisible-k8.json` |
| Integration default | 273.68 | `plm-21-final-exact-default-full61-tenvisible-k8.json` |
| Tree branch verification (depth-2) | 164.92 | `mtp-tree-default-final-full61-tenvisible.json` |
| Adaptive depth-4 | 157.81 | `mtp-adaptive-default-depth4-full61-tenvisible.json` |
| Warm prompt-prefill cache | 98.18 | `mtp-warm-prefill-finish6-depth4-full61-tenvisible.json` |
| Rank-first top-3 continuation (depth-6) | 85.52 | `mtp-warm-prefill-rank-first-next3-depth6-full61-tenvisible.json` |
| Continuation pruned to rank-3 | 60.61 | `mtp-warm-prefill-rank-first-continuation-rank3-depth6-full61-tenvisible.json` |
| One-pass top-2048 (depth-10) | 37.49 | `mtp-warm-prefill-rank-onepass-top2048-depth10-full61-tenvisible.json` |
| **Shared-head warm-evict (current)** | **34.23** | `mtp-warm-prefill-mtphead-evict-before-verify-...json` |

**16.6× faster**, from 569.11 to 34.23 s/visible-token.

Measured separately, before the MTP work, the first *correct* full 62-layer exact
cached next token was `386.038s` (attention `186.970s`, FFN `167.654s`, tail `1.327s`),
improving to `275.327s` once the FP8 attention cache moved to a 4096 MB prefix policy.

## Component-level wins

| Optimization | Before | After | Gain |
|---|---:|---:|---:|
| Exact repeated-prefix reuse (8-layer first visible token) | 87.085 s | 0.943 s | 92× |
| Full `lm_head` cache (cached 8-layer tail) | 17.585 s | 1.611 s | 10.9× |
| Native kernel kill-switch (full 62-layer) | 800.553 s | 307.894 s | 61.5% faster |
| AVX-512 FP8 MoE (bounded FFN) | 11.8807 s | 11.2604 s | 5.2% |

## Anti-cheat verification

Speed claims for a 671B model on a laptop are easy to fake — by silently skipping
layers, dropping to Q4, substituting a smaller model, or offloading to a GPU. The
runtime therefore counts and asserts real work on every measured run:

- `layers_executed` must equal `expected_layers_executed` (122/122 on the headline run)
- warm-up executes a separate `61/61` layer proof, recorded independently
- the artifact records `model_id`, quantization, expert routing, and `strategy`
- the verifier contract enforces **one weight sweep per pass** (`one_weight_sweep_per_pass`)
- any blocker populates a `blockers` array and invalidates the proof

A run only counts as a promotion if `anti_cheat_passed` is `true` and `blockers` is empty.

## Rejected experiments

Optimizations that were built, measured, and **rejected** for being slower than the
standing default. Kept here because negative results are part of the record.

| Experiment | Measured | vs. default | Outcome |
|---|---:|---|---|
| Active-pair FP8 MoE kernel | 34.72 s | 33.77 s baseline | rejected |
| Routed-all 2 GB cache filter | 36.51 s | 34.23 s | rejected |
| Shared-only 3 GB cache filter | 36.17 s | 34.23 s | rejected |
| Shared-only 2 GB cache filter | 35.48 s | 34.23 s | rejected |
| Row-weighted chunk8 | 42.03 s | 34.23 s | rejected |
| Row-weighted chunk32 | 41.53 s | 34.23 s | rejected |
| Forced depth-20 branch | 61.06 s | accepted only 11/20 candidates | rejected |
| Native attention sequence gate | — | changed verifier token ids | rejected |

Depth-3 and depth-5 were also measured and found slower than depth-4 before depth-4
was promoted.

## GPU probe

A single resident-layer GPU experiment on a Kaggle T4 measured `0.02874 s/layer`
once weights were resident, projecting `1.753 s/token` across 61 layers. Cold
HTTP + dequant on the same path was `39.580s`, and direct paging from Kaggle's
mounted model source was `29.020s` for one layer plus tail — so the projection is
an upper bound on a resident-weights machine, not a delivered end-to-end result.
The CPU exact path remains the default.

## Test suite

`560 passed, 2 skipped` on the full suite as of the last recorded run, across 79 test
files. GitHub Actions runs installation and the portable contract suite on a clean
Windows runner for every push.

## Reproducing

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Benchmark tools live in [`tools/`](tools/) — for example
`tools/bench_fp8_mtp_generate.py` regenerates the MTP proofs and
`tools/bench_fp8_avx512_moe.py` regenerates the AVX-512 MoE proof. Model weights
are never committed and must be supplied locally.
