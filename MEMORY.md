# MEMORY.md - Pocket LLM orchestration memory

This file holds the durable working rules for reviewing and orchestrating Pocket LLM sessions.

## Identity

- Project: `pcketlm` / Pocket LLM.
- Owner/operator: Licht / Issa Alexandre Tagro.
- Root: `C:\Users\isale\Documents\pcketlm`.
- Product: local-first LLM control center for Windows, weak-hardware support, direct model loading, model management, personalization, profiles, comparison, benchmarking, and agent-oriented workflows.

## Orchestration Rule

Codex should operate autonomously by default on Pocket LLM:

1. Read `project_pcketlm.md` first for the current operating brief.
2. Read `STATUS.md`, `TODO.md`, `DECISIONS.md`, and the tail of `LOG.md` / `DONE.md` before giving direction.
3. Review the finished session for correctness, regressions, missing tests, documentation drift, and whether it obeyed project decisions.
4. Update tracker files when meaningful project state changed.
5. Choose the next smallest useful task, self-prompt it, and execute it without waiting for step-by-step approval.
6. Return to Issa only for true product direction choices, paid/cloud/GPU/install decisions, destructive actions, hardware ceilings, or loops that cannot be broken safely.
7. If a crash or blocker appears, switch immediately into debugging mode: localize, classify, fix if safe, verify, log, and continue.

## Standards

- Perfection standard: nothing ships without review.
- Clear bugs are debugged directly; do not turn obvious bug fixing into a decision ritual.
- Decisions should use 3 options with one explicit recommendation, plus pros, cons, and failure risk.
- Keep originals immutable; optimized artifacts must be derived and recreatable.
- Beginner UX should be plain English; advanced/runtime detail belongs in deeper views.
- Avoid claiming broad model support before direct runtime paths are real.

## Current Strategic Direction

- Qwen is the active first family.
- GGUF/llama.cpp is now the recommended fast backend path for practical local speed.
- Direct safetensors CPU remains important as Pocket's custom runtime foundation, but repeated large tensor movement is the wall.
- GGUF is a power-user backend because the loaded server uses roughly 8.6 GB to 10 GB RAM on the current machine.
- Warm Agent mode exists as an opt-in direct-runtime path, but current proof shows it is mostly a correctness/session-reuse feature, not yet a major speed breakthrough.
- Load Model now exposes GGUF file list, selected artifact, expected RAM, cold-load estimate, state, and one Load/Unload action.
- Benchmarks now expose an honest backend comparison table for GGUF, Direct Standard, and Direct Boosted, with a dedicated `Run comparison` action that produces real rows for all three paths when RAM allows.
- Latest live comparison measured Direct Standard at `21.03s`, Direct Boosted at `19.81s`, and GGUF Compare at `26.05s` on the same one-token prompt; GGUF remains the recommended practical backend because loaded-server use is faster for real chat/agent work.
- GGUF benchmarks now include stronger agent-plan and follow-up checks, and GGUF status includes disk summaries for complete and split artifacts.
- Phase 3 speed/reliability productization is complete for the current branch. Next valuable work should move to Phase 4 agent product foundation and workflow reliability.
- User corrected direction: Phase 4 is not agents. Phase 4 is Qwen 14B speed only.
- Success target: normal Qwen 14B chat must move from about `19-22s/token` direct runtime to `2-4s/token` or better.
- First Phase 4A slice changed the product default to GGUF, labeled direct modes as Direct, added qwen14b_speed_target status, and blocked hidden cold GGUF loads behind Send.
- Current status: GGUF fast path is available but not loaded; expected RAM is about `9001 MB`, cold load estimate about `51.9s`, current free RAM after tests was about `4.5 GB`.
- Phase 4B loaded Qwen 14B GGUF after reclaiming memory. The server loaded in `23.24s` and reported ready.
- Live app-path warm speed proof: prompt `Give one concise sentence about why local AI speed matters.`, `mode=GGUF`, generated `19` tokens at `0.352s/token` / `2.843 tokens/sec`, wall `7.28s`, target met.
- GGUF chat responses now include `generation_speed`; runtime details show Token speed and Speed target.
- Current status: Qwen 14B speed goal is achieved for the warmed GGUF path. Remaining work is product hardening around load state, RAM pressure, persistence/recovery, and making this feel smooth after app restart.
- High effort is best for architecture, native crash debugging, large runtime changes, and phase planning. Medium is enough for docs, UI polish, focused tests, and small productization slices.

## Verification Habit

- For code changes, run focused tests first, then the full unit suite when scope justifies it.
- After runtime or web chat changes, verify with at least one real short model smoke when RAM and backend state allow it.
- Keep compile verification clean with `python -m compileall -q src tests` or the repo's current equivalent.
- After each meaningful phase, update `STATUS.md`, `TODO.md`, `DONE.md`, `DECISIONS.md`, `LOG.md`, and these memory files when long-term project state changed.
