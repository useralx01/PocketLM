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
- GGUF prompts use Qwen instruct chat formatting and stop markers.
- Fast GGUF benchmark flow exists and saves short-answer, logic, and agent-style checks.
- Backend recommendation now prefers the ready GGUF path while keeping Direct CPU as the dense custom-runtime foundation and fallback.

## Latest Known Proof Points

- Persistent GGUF server load/unload works through Pocket APIs.
- Loaded GGUF server uses roughly 8.6 GB to 10 GB RAM.
- Latest Phase 3C GGUF proof: cold GGUF call took 52.26s because it included model load; warm server call with 4-token cap took 2.36s at about 2.30 tokens/sec with server working set around 8.64 GB.
- GGUF server was unloaded after the proof and state reported `running=false`.
- Latest Warm Agent proof reused 64 prompt tokens, batch-appended 14 tokens, returned `Ok<|im_end|>` in 47.06s with roughly 1069 MB process working set.
- Latest short GGUF checks returned `OK`, `YES`, and `SUN` in roughly sub-second to low-second app-reported times after the server was loaded.
- Longer GGUF agent-style check can return two complete numbered steps in about 20 seconds for a 48-token cap.
- Latest saved GGUF benchmark includes 1, 4, 8, and 32 token rows plus logic and agent checks.
- Latest full unit suite after Phase 3C: 197 passed in 21.20s; compile verification was clean.
- Current productization estimate: custom direct-runtime foundation is mature, but overall Phase 3 speed/reliability productization is roughly 78% because GGUF recommendation, load UX, and comparison flows still need more polish.

## Current Best Next Work

Favor work that makes the real GGUF path safer and more product-grade while preserving the direct dense runtime as Pocket's unique foundation:

1. Decide how GGUF mode should sit beside Direct CPU Standard/Boosted in the customer UI.
2. Add stronger agent-style GGUF checks for tool-planning wording, task decomposition, and follow-up consistency.
3. Add a clean GGUF artifact manager so split downloads, merged artifacts, file sizes, and disk usage are visible from Load Model.
4. Add a clear pre-load warning before loading the roughly 9 GB GGUF artifact, including expected cold-load time and RAM use.
5. Add a backend comparison runner for GGUF versus Direct CPU Standard/Boosted, gated by available RAM and safe server lifecycle behavior.

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
