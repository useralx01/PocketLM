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
