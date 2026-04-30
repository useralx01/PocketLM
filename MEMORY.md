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
- Next valuable work should improve GGUF management, stronger GGUF agent checks, backend comparison, and safe UI defaults rather than more blind safetensors cache expansion.
- High effort is best for architecture, native crash debugging, large runtime changes, and phase planning. Medium is enough for docs, UI polish, focused tests, and small productization slices.

## Verification Habit

- For code changes, run focused tests first, then the full unit suite when scope justifies it.
- After runtime or web chat changes, verify with at least one real short model smoke when RAM and backend state allow it.
- Keep compile verification clean with `python -m compileall -q src tests` or the repo's current equivalent.
- After each meaningful phase, update `STATUS.md`, `TODO.md`, `DONE.md`, `DECISIONS.md`, `LOG.md`, and these memory files when long-term project state changed.
