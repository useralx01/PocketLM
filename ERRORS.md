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
