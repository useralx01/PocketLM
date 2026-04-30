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
