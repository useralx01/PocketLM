# project_pcketlm.md - Pocket LLM project context

Use this as the first-read orchestration brief for Pocket LLM. The source of truth still lives in the tracker files in this folder, especially `STATUS.md`, `TODO.md`, `DECISIONS.md`, `LOG.md`, `DONE.md`, and `ERRORS.md`.

## Project Shape

Pocket LLM is a Windows-first, local-first LLM control center. It should run and manage models directly instead of depending on Ollama, LM Studio, or another local LLM app as the primary runtime. It should help weak-hardware users while still offering power-user controls for backend selection, model artifacts, profiles, benchmarking, and agent workflows.

The active first model family is Qwen. Future family priority is Qwen, then Kimi, then Kronos/Kronk-family targets if there is a real supported open model path, then Gemma, then broader dense model families.

## Non-Negotiables

- Keep original model files immutable.
- Derived runtime packs, optimized artifacts, profiles, and caches must be reversible or recreatable.
- Do not present rough partial-layer output as finished customer quality.
- Do not expose raw runtime complexity as the beginner default.
- Do not claim universal model support before import, readiness, runtime, chat, personalization, compare, and benchmark paths are honest for that family.
- Tracker docs must be updated after meaningful state changes.

## Current State Summary

- Direct CPU Qwen runtime exists and can answer real prompts, but it is slow on the current laptop because repeated large tensor movement dominates.
- Quick mode exists for honest one-token full-stack checks.
- Exact response reuse and session-prefix reuse exist; repeated or prefix-matching requests can be much faster.
- Standard and Boosted tensor residency/runtime-pack presets exist; Boosted is optional and gives only a small win so far.
- The custom safetensors path is valuable foundation work, but more cache/pack expansion has shown low returns on this machine.
- Warm Agent runner exists as an opt-in mode in Settings (`off`, `safe`, `experimental`) with Start/Stop controls, status reporting, and Qwen chat-format handling fixed so the prompt is formatted once.
- Backend capability reporting exists for Direct CPU, CUDA, DirectML, and GGUF-style paths.
- GGUF/llama.cpp backend support is now real: Pocket can use a standalone llama.cpp runtime, a merged Qwen2.5-14B-Instruct Q4_K_M GGUF artifact, and a persistent `llama-server`.
- Web GGUF mode can produce fast short answers after the server is loaded.
- Load Model has GGUF server lifecycle controls for loading/unloading the persistent server and showing RAM/artifact state.
- Load Model now shows the selected GGUF file, all discovered GGUF files, expected RAM, estimated cold-load time, and a single Load/Unload action.
- GGUF prompts use Qwen instruct chat formatting and stop markers.
- Fast GGUF benchmark flow exists and saves short-answer, logic, and agent-style checks.
- Backend recommendation now prefers the ready GGUF path while keeping Direct CPU as the dense custom-runtime foundation and fallback.
- Benchmarks now include an honest backend comparison table for GGUF, Direct Standard, and Direct Boosted, tagging fastest, best quality, lowest RAM, and recommended from measured rows.
- Benchmarks now also include a dedicated `Run comparison` action that measures Direct Standard, Direct Boosted, and GGUF on the same short prompt.
- GGUF status includes disk summaries for complete GGUF files and split shards.

## Latest Known Proof Points

- Persistent GGUF server load/unload works through Pocket APIs.
- Loaded GGUF server uses roughly 8.6 GB to 10 GB RAM.
- Latest Phase 3C GGUF proof: cold GGUF call took 52.26s because it included model load; warm server call with 4-token cap took 2.36s at about 2.30 tokens/sec with server working set around 8.64 GB.
- Latest Phase 3D GGUF Load Model productization: expected RAM is file size times 1.05, cold-load estimate uses 6.2s/GB from local proof, and focused GGUF/web tests passed with 60 tests.
- Latest Phase 3F backend comparison: Direct Standard measured `21.03s`, Direct Boosted measured `19.81s`, and GGUF Compare measured `26.05s`; the comparison table still shows those rows after later GGUF-only benchmarks.
- Latest Phase 3F GGUF agent checks returned complete numbered plan/follow-up actions, and the full suite passed with 208 tests.
- GGUF server was unloaded after the proof and state reported `running=false`.
- Latest Warm Agent proof reused 64 prompt tokens, batch-appended 14 tokens, returned `Ok<|im_end|>` in 47.06s with roughly 1069 MB process working set.
- Latest short GGUF checks returned `OK`, `YES`, and `SUN` in roughly sub-second to low-second app-reported times after the server was loaded.
- Longer GGUF agent-style check can return two complete numbered steps in about 20 seconds for a 48-token cap.
- Latest saved GGUF benchmark includes 1, 4, 8, and 32 token rows plus logic and agent checks.
- Latest full unit suite after Phase 3F: 208 passed in 14.77s; compile verification was clean.
- Current productization estimate: Phase 3 speed/reliability productization is complete for this branch. Direct dense runtime remains slow but honest; GGUF is the practical speed path.

## Current Best Next Work

Favor Phase 4 agent product foundation:

1. Define the first real agent workflow path: goal, plan, inspect, act, verify, summarize.
2. Add job state, cancel/stop, logs, and bounded tool permissions for local agent work.
3. Use GGUF as the practical default backend for short agent reasoning when loaded, with Direct CPU kept as the dense/runtime research fallback.
4. Keep model/artifact management improving in parallel, but do not keep expanding direct safetensors speed tweaks without a measured architecture change.

## Orchestration Checklist After Another Session

1. Confirm what files/code changed.
2. Check whether `STATUS.md`, `TODO.md`, `DONE.md`, `DECISIONS.md`, `ERRORS.md`, and `LOG.md` were updated appropriately.
3. Look for mismatch between claimed result and actual verification.
4. Run or request focused tests if the other session did not.
5. If runtime/web chat changed, prefer one real short smoke through the relevant backend when safe.
6. Identify the next smallest useful step, self-prompt it, and execute it.
7. Continue autonomously through logical slices until a true decision point, destructive action, hardware ceiling, paid/cloud/GPU/install choice, or repeated loop appears.
8. If a product direction choice is needed, present exactly 3 options and recommend one.

## Useful Commands

From `C:\Users\isale\Documents\pcketlm`:

```powershell
git status --short
python -m pytest
python -m compileall -q src tests
```

Use repo-specific Python launcher choices from current docs if `python` does not target the active environment.
