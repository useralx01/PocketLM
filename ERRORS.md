# Errors

Use this file for:

- mistakes we made
- false assumptions
- blocked experiments
- repeated failures
- lessons we do not want to relearn

Template:

## Error
- date:
- area:
- what went wrong:
- why it happened:
- fix:
- lesson:

## Error
- date: 2026-04-29
- area: Phase 2 / Qwen 32B first run
- what went wrong: the first full Qwen2.5-32B-Instruct direct runtime attempt exited natively with Windows exit code `-1073741819` (`0xC0000005`, access violation). The command wrote an empty stdout file and an empty stderr file, so there was no Python traceback to capture.
- why it happened: the model files and tensor catalog were present, but the native runtime process crashed before returning a `PromptDecodeLoopResult`. This is likely a native memory/access fault in the Torch/safetensors path under the current machine pressure, not a normal Python exception.
- fix: stop Phase 2 Step 7 here. The next session should add narrower crash-localization instrumentation or run a smaller isolated token-entry/layer diagnostic before retrying the full 64-layer 32B prompt.
- lesson: once a direct 32B run exits with `0xC0000005`, do not keep retrying the same full command. Capture the exit code, residency policy, RAM state, and stop.

## Error
- date: 2026-04-29
- area: Phase 2 / Recovery / 14B baseline
- what went wrong: the recovery diagnostic passed Qwen 14B slices through `layer-0-15`, but the required 14B `full` slice exited natively with code `-1073741819` (`0xC0000005`) before emitting an `after` checkpoint. stdout stopped at `before full-prompt-decode`; stderr was empty.
- why it happened: the full prompt path crashes independently of Qwen 32B on this run, so the recovery diagnostic cannot honestly localize the 32B crash yet. The smaller layer-stack slices are healthy, which points toward the full prompt/decode-tail path or the environment, not embedding or early layers.
- fix: stop before touching 32B. Next, split the diagnostic further between full prompt prefill stack and final norm + lm_head/decode tail on 14B, then repeat only after the 14B full baseline is stable.
- lesson: do not use 32B as the first crash-localization target when 14B full prompt decode is also producing the same native access violation.

## Error
- date: 2026-04-29
- area: Phase 2 / Recovery 2 / 32B full prompt decode
- what went wrong: Qwen 32B `full` still exits natively with `-1073741819` / `0xC0000005` at the `before full-prompt-decode` checkpoint, while all progressive 32B slices through `all-layers-norm-lm` pass.
- why it happened: the 32B weights, embedding lookup, all 64 layers, final norm, and lm_head can run in isolation. The remaining failure is specific to the full prompt decode loop/KV path, likely prompt-length prefill with KV state or scoped handle behavior rather than a layer-shape/catalog issue.
- fix: stop this session per the recovery rule because 32B now fails at a different slice than the recovered 14B path. Next session should split 32B `full` into prompt token-entry, prompt-length prefill stack with `return_kv_cache=True`, prefill decode tail, and first continuation step.
- lesson: the 32B problem is not model-file access, early layers, all layers, final norm, or lm_head alone; localize inside full prompt/KV decode before applying any fix.

## Error
- date: 2026-04-29
- area: Phase 2 / Recovery 3 / 32B full decode loop
- what went wrong: Qwen 32B `full --max-new-tokens 1` still exits natively with `-1073741819` / `0xC0000005` after the diagnostic emits `before full-prompt-decode`. The split diagnostic path passes `prompt-prefill`, `decode-step-1`, and `decode-step-2`.
- why it happened: KV creation and growth are not the immediate trigger: `prompt-prefill` created 64 KV layers for 31 tokens (`8,126,464` bytes), and `decode-step-2` grew them to 32 tokens (`8,388,608` bytes) while reloading/running all 64 layers. The failure correlates with the full decode-loop wrapper path, not isolated KV budget or weight reload.
- fix: do not patch yet. Next session should instrument inside `run_prompt_decode_loop` itself around the same sub-operations, especially the wrapper's lifecycle/timing/final-result path, to find the difference from the split diagnostic path.
- lesson: when isolated runtime pieces pass but the wrapper dies natively, the next evidence must come from inside the wrapper, not broader retries or lower-level tensor changes.

Follow-up:
- Setting `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=0` made `full --max-new-tokens 1` pass on Qwen 32B with generated text `Hello` in `40.317s`.
- Updated fix direction: keep scoped safetensor handle caching disabled for the 32B/full path and audit the scoped handle lifecycle before re-enabling it.

## Error
- date: 2026-04-30
- area: Phase MoE / Step 9 / Qwen3 all-layers-moe
- what went wrong: `qwen3-30b-a3b --slice=all-layers-moe` failed at layer 0 with `RuntimeError("shape '[1, 1, 32, 64]' is invalid for input of size 4096")`.
- why it happened: the dense Qwen2 path derived attention head dimension as `hidden_size / num_attention_heads`. Qwen3 MoE stores `head_dim=128` in config, so q_proj output is `32 * 128 = 4096` even though hidden size is 2048.
- fix: teach the layer bridge config to read `head_dim` from config.json and use it for Q/K/V reshaping, falling back to the dense derivation only when the field is absent.
- lesson: MoE support must respect explicit config dimensions instead of deriving every attention shape from hidden size.

## Error
- date: 2026-04-30
- area: Phase MoE / Step 9 / Qwen3 all-layers-moe
- what went wrong: after Q/K/V reshaping was fixed, `qwen3-30b-a3b --slice=all-layers-moe` failed at layer 0 with `RuntimeError("shape '[1, 1, 2048]' is invalid for input of size 4096")`.
- why it happened: the attention context before `o_proj` is `num_attention_heads * head_dim`, which is 4096 for Qwen3, not `hidden_size` 2048. The bridge was reshaping the pre-output-projection attention context directly to hidden size.
- fix: reshape attention context to `num_attention_heads * head_dim`, then let `o_proj` map it back to hidden size.
- lesson: the bridge must distinguish attention projection width from residual hidden width.

## Error
- date: 2026-04-30
- area: Phase MoE / Step 9 / Qwen3 full prompt decode
- what went wrong: `qwen3-30b-a3b --slice=full --max-new-tokens 1` succeeded with `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=0`, but the same command with the default scoped handle policy exited after the `before full-prompt-decode` checkpoint without an `after` record.
- why it happened: Qwen3's full decode path has the same scoped safetensor handle instability previously observed on Qwen 32B. The model can run when scoped handle reuse is disabled.
- fix: make `qwen3-30b-a3b` default to no scoped safetensor handle cache, while preserving the explicit env override for diagnostics.
- lesson: correctness defaults beat handle reuse for large paged models until the scoped handle lifecycle is redesigned.

## Error
- date: 2026-04-23
- area: model download / Hugging Face import
- what went wrong: the initial `Qwen/Qwen2.5-14B-Instruct` download stalled and never progressed beyond lock files or tiny metadata files
- why it happened: Hugging Face Xet/CAS failed with repeated `HTTP 416 Range Not Satisfiable` errors, which left the process alive but not meaningfully downloading the safetensor shards
- fix: stop the stalled download, clear the partial target, clear the Xet cache, disable Xet with `HF_HUB_DISABLE_XET=1`, and retry over plain HTTPS
- lesson: if Hugging Face downloads create lock files but shard sizes stay at zero, check the Xet log immediately before waiting longer

## Error
- date: 2026-04-23
- area: model download / plain HTTPS retry
- what went wrong: the retry with Xet disabled also failed to produce a usable model folder
- why it happened: the process stayed open, but the local folder remained stuck at metadata-only state and never progressed into a full source model
- fix: stop the detached download and switch to a different acquisition path
- lesson: if disabling Xet still leaves the folder frozen at tiny metadata size, stop early and use another model source instead of retrying blindly

## Error
- date: 2026-05-01
- area: Phase MoE Correctness / gibberish MoE output
- what went wrong: Qwen3 generated text like `exion particular,...` and Mixtral generated text like `XamarinpfnINCLUDINGINCLUDING...` during speed validation, despite green tests and good timing metrics.
- why it happened: the MoE long-generation path used only 12 transformer layers for max_new_tokens > 8, so output came from a truncated model. Mixtral also combined expert outputs with unnormalized top-k router weights when config.json omitted `norm_topk_prob`, unlike Transformers' Mixtral router.
- fix: use the full configured MoE layer count for prompt decoding and default missing MoE `norm_topk_prob` to true while honoring explicit config values.
- lesson: generated text is a correctness signal. Speed measurements on a truncated or mathematically mismatched model are not product evidence.
## Phase MoE Honest Speed / STOP-2 / Tiny Mixtral strict checkpoint drift

```text
slice=compare-with-reference
model=tiny_moe_mixtral
checkpoint=layer0_combined_hidden
strict_gate=max_abs_diff < 1e-4 and cosine >= 0.999

attempt_1:
- PCKETLM_RUNTIME_MATH_DTYPE=float32
- generated_token_ids exact 10/10
- cosine=0.9999986952
- max_abs_diff=0.00023440271615982056

attempt_2:
- Regenerated HF oracle with attn_implementation="eager"
- generated_token_ids exact 10/10
- cosine=0.9999985647
- max_abs_diff=0.00023440271615982056

attempt_3:
- PCKETLM_RUNTIME_MATH_DTYPE=float32
- PCKETLM_TORCH_THREADS=1
- OMP_NUM_THREADS=1
- MKL_NUM_THREADS=1
- generated_token_ids exact 10/10
- cosine=0.9999985647
- max_abs_diff=0.00023440271615982056

classification:
- Not a semantic MoE routing failure: generated ids match the HF oracle exactly.
- Still a STOP under the phase rules because the strict checkpoint max_abs gate remains red after three attempts.
```

## Phase Speculative / STOP-3 / GGUF speculator mismatch

```text
symptom:
- Non-speculative Qwen3-30B-A3B first greedy token for "The capital of France is" is token 151667, decoded as '<think>'.
- GGUF Qwen 14B proposes direct-answer text instead:
  - K=2: [785, 6722] -> 'The capital'
  - K=4: [785, 6722, 315, 9625] -> 'The capital of France'
  - K=8: [785, 6722, 315, 9625, 374, 12095, 13] -> 'The capital of France is Paris.'

real runs:
- Non-speculative baseline: 386.676s total, 19.3338s/token, layers_executed=960/960.
- Speculative K=4, max_new_tokens=1: 130.7245s total, 130.719s/token, accepted_token_count=0, corrected_token_count=1, layers_executed=48/48.
- Speculative K=4, max_new_tokens=20: timeout after 1200s, no result row.
- Speculative K=4, max_new_tokens=4: process exited 1 before an after/result row; last checkpoint free_ram=2824MB, working_set=204MB.

classification:
- STOP-3. The speculator/verifier distributions are mismatched; the verifier accepts zero candidates at the first position.
- Tuning K cannot fix a first-token mismatch. A useful next attempt needs a speculator with Qwen3-thinking behavior or a verifier prompt mode that disables thinking consistently for both non-speculative and speculative baselines.
```

## Phase Speculative Pair / acquisition fallbacks

```text
failed_exact_repos:
- bartowski/Qwen3-1.7B-Instruct-GGUF
- Qwen/Qwen3-1.7B-Instruct-GGUF
- bartowski/Qwen3-4B-Instruct-GGUF
- bartowski/Qwen3-0.6B-Instruct-GGUF
- Qwen/Qwen3-1.7B-Instruct
- Qwen/Qwen3-0.6B-Instruct
- Qwen/Qwen3-4B-Instruct

fallback_taken:
- Qwen/Qwen3-1.7B safetensors
- Qwen/Qwen3-0.6B safetensors
- bartowski/Qwen_Qwen3-1.7B-GGUF Q4_K_M

classification:
- Recoverable naming mismatch, not a model availability failure. Same-family base Qwen3 models exist and match reasoning behavior.
```

## Phase 32B Fix / old native crash no longer reproduces

```text
previous symptom:
- Qwen2.5-32B-Instruct --slice=full crashed with native Windows 0xC0000005 and no Python traceback.

current result on branch phase-32b-fix from commit 0fe28ef:
- all diagnostic slices through all-layers-norm-lm exit 0.
- --slice=full --max-new-tokens=1 exits 0 and generates "The" with layers=64/64.
- --slice=full --max-new-tokens=4 exits 0 and generates "The capital of France" with layers=256/256.
- repeat=3 full validation also exits 0 for all runs.

classification:
- Resolved by intervening runtime changes; no new crash root cause remained to bisect in this phase.
```

## Phase Q4 Streaming / target miss

```text
symptom:
- Qwen 32B Q4 streaming generated coherent text but ran slower than fp16.

numbers:
- fp16 4-token comparison: 51.4145s/token, generated "The capital of France".
- Q4 4-token run: 119.0094s/token, generated "The capital of France".
- Q4 + speculative bounded 4-token check: 118.8495s/token, accepted=3, corrected=1.

root cause:
- Q4 disk reads work, but Python dequantization during tensor load dominates the runtime.
- Repeated decode-step cache misses mean the same class of tensors is dequantized again across generated tokens.

fix direction:
- Keep the Q4 artifact format and correctness tests.
- Do not call this a speed path until dequant is native/fused or the runtime keeps a larger fp16 dequantized window resident across decode steps.
```
## Phase C++ Q4 Dequant / STOP-4

Native Q4 dequant built and matched Python byte-for-byte on focused tensors, but real Qwen 32B Q4 speed regressed to `311.1491s/token`. The measured bottleneck remained tensor loading: `prefill_stack_op_load_tensors=303.8371s` for one generated token, `q4_loads=769`, `q4_loaded_mb=14880.53`.

Root cause: the first native kernel is scalar C++ invoked per tensor through ctypes. It removes Python tensor math but does not use SIMD/threading, and it is slower than the existing PyTorch vectorized Q4 fallback in a one-tensor microprofile (`0.17-0.30s` native vs `0.10s` Python vectorized for `model.layers.0.self_attn.q_proj.weight`).

Next fix direction: replace scalar dequant with a SIMD/threaded fused unpack+dequant kernel, or move to a larger packed executor that amortizes calls and copies across tensor groups.

## Phase C++ Q4 Dequant SIMD / target miss

The SIMD/OpenMP kernel fixed the scalar-kernel problem but did not make the full Qwen 32B Q4 path fast enough. Native SIMD one-token runtime was `63.5385s/token`; Python fallback was `131.967s/token`; both produced `The` with `64/64` layers. The microbench target passed (`0.0037s` native vs `0.0729s` Python on `5120x5120`), so the remaining full-stack bottleneck is not raw dequant speed.

Root cause: the layer bridge reports `56.8791s` in tensor loading, while a standalone grouped Q4 pass over all `771` tensors reports `12.629s` wall time. The gap points at runtime load/residency orchestration overhead rather than the SIMD kernel itself.

Next fix direction: make Q4 tensor loading grouped and persistent at the bridge/residency level, or build a packed executor that avoids per-layer/per-call tensor loading overhead.

## Phase Native fp16 Engine / unresolved blockers
- Production native layer orchestrator is missing.
- Native attention is prefill-only and does not own KV cache or rollback.
- Native matmul needs cache blocking or a better tiling strategy before model-scale routing; current 1024x1024 native=0.060882s vs torch=0.009111s.
- Real 14B native-loader-only path is slower than runtime-pack/scoped-handle path: 37.051s for max_new_tokens=1.

## Phase Native fp16 Integration / production routing blocker
- Production routing through layer_bridge.py is not enabled because the deployed model sources are BF16 while the native orchestrator path is fp16. Direct routing would change precision and risks failing the identical greedy-token regression gate.
- Native GEMM still loses to torch at 1024x1024 after the NR=16/FMA pass, so using it in production would likely slow model-scale layers.
- C-owned KV/decode/orchestrator correctness is proven on isolated tiny tests, but not through the full native path in layer_bridge.py.

## Phase Native fp16 Integration / native DLL load churn
- Symptom: full-suite runs intermittently failed native availability checks with `[WinError 4551] An Application Control policy has blocked this file` after rebuild/checkout churn.
- Root cause: Windows Application Control blocked freshly replaced DLL files, not a kernel correctness failure.
- Fix: delete/rebuild affected DLLs and run `Unblock-File` on `src/pcketlm/native/*.dll`; native module checks then returned kv=true, moe=true, loader=true, q4=true, matmul=true, and the full suite passed (`306 passed`).

## Phase Native fp16 Integration / rejected native defaults
- Dense native prefill was correct but regressed real 14B prefill (`prefill_stack_op_native_layer=64.3411s` vs default torch prefill around `19.8s`), so it remains opt-in via `PCKETLM_ENABLE_NATIVE_DENSE_PREFILL=1`.
- Native lm_head top-k was correct but regressed the real 14B row (`continuation_decode_tail=3.626s`), so it remains opt-in via `PCKETLM_ENABLE_NATIVE_LM_HEAD_TOPK=1`.

## Phase Native fp16 BLAS / 14B native regression
- Symptom: native Qwen 14B generated `HelloWorld<|im_end|>` for `hello world`, while Python fallback generated `Hello! How`.
- Root cause: native RoPE frequency exponent was off by a factor of 2.
- Fix: corrected RoPE exponent in `fp16_kv_cache.cpp` and `fp16_attention.cpp`; corrected the test oracle in `test_native_fp16_kv_cache.py`.

## Phase Native fp16 BLAS / OpenBLAS in KV DLL
- Symptom: real Qwen 14B continuation exited before diagnostic completion when `fp16_kv_cache.dll` was linked against OpenBLAS.
- Root cause: OpenBLAS dependency in the C-owned KV/layer DLL was not stable at real model dimensions on this Windows runtime; small tests were insufficient to catch it.
- Fix: removed OpenBLAS from the KV/layer DLL and kept BLAS isolated to `fp16_matmul.dll`.
