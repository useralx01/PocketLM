# PocketLM

PocketLM is a Windows-first local AI runtime and control center. Model weights are never included in this repository.

## Test On Another Desktop

1. Open the private GitHub repository and download the latest prerelease ZIP.
2. Extract it to a local folder.
3. Open PowerShell in that folder and run:

```powershell
.\install-pocketlm.ps1 -StartApp
```

The installer creates an isolated environment under `%LOCALAPPDATA%\PocketLM\venvs`, installs PocketLM, and writes a sanitized proof to `state\supervisor\install-proof.json`. Keeping the environment there avoids Windows path-length failures even when the ZIP is extracted into a deep folder. Python 3.12 or newer is required. To launch it again later, run `start-pocketlm.ps1`.

PocketLM can install without model weights. The Settings screen distinguishes a healthy installation with no model (`partial`) from a machine with a runnable model (`ready`). Add models separately on each desktop or connect the storage that already contains them.

## Installation Supervisor

Every copy exposes local-only health endpoints while the app is running:

- `GET http://127.0.0.1:8765/api/supervisor/health` returns the current sanitized report.
- `POST http://127.0.0.1:8765/api/supervisor/check` runs the check and saves a proof artifact.

The report uses a random installation ID and does not contain usernames, hostnames, IP addresses, local paths, prompts, or model contents. It is never uploaded automatically. GitHub Actions runs the same installer and contract checks on a clean Windows machine for every push and pull request.

## Development

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pytest -q
```

## Project Notes

This is the Mission Control tracking category for the `pcketlm` project.

Use this folder as the single place to track:

- what we are building
- what is in progress
- what is blocked
- what is done
- what broke
- what decisions we made

Working communication rule:

- when a decision point comes up, present 3 clear options
- recommend one option explicitly
- explain why it is recommended
- include pros, cons, and likely failure risk for all options
- keep these tracker files updated whenever the project changes

Bug handling rule:

- if something is clearly bugged, debug it immediately
- do not turn obvious bug-fix work into a 3-option decision
- treat debugging as the default action until the broken behavior is understood

Main files:

- [BLUEPRINT.md](C:/Users/isale/Documents/pcketlm/BLUEPRINT.md)
- [DESKTOP_STATUS_SCREEN.md](C:/Users/isale/Documents/pcketlm/DESKTOP_STATUS_SCREEN.md)
- [RISK_REGISTER.md](C:/Users/isale/Documents/pcketlm/RISK_REGISTER.md)
- [STATUS.md](C:/Users/isale/Documents/pcketlm/STATUS.md)
- [TODO.md](C:/Users/isale/Documents/pcketlm/TODO.md)
- [DONE.md](C:/Users/isale/Documents/pcketlm/DONE.md)
- [ERRORS.md](C:/Users/isale/Documents/pcketlm/ERRORS.md)
- [DECISIONS.md](C:/Users/isale/Documents/pcketlm/DECISIONS.md)
- [LOG.md](C:/Users/isale/Documents/pcketlm/LOG.md)

Current implementation snapshot:

- desktop test UI exists and launches from the desktop shortcut
- official `Qwen2.5-14B-Instruct` source is downloaded locally
- plain full CPU load is still blocked by RAM on this machine
- staged streaming now includes planning, materialization, verification, rotation, residency tracking, control decisions, and both live `Advance Stream` and `Safe Advance` actions in the desktop app
- a persisted tensor catalog now exists so the runtime can reason about real tensors and layers instead of only raw streamed byte segments
- a persisted tensor execution plan now exists so the runtime can reason about grouped execution units instead of only individual tensors
- the runtime can now load one real tensor or one small grouped execution unit from the original shards on demand
- the runtime can now also verify loaded tensors and larger grouped execution units against persisted metadata before we attempt real execution math
- the runtime can now execute a first real CPU-only layer-0 bridge for Qwen using synthetic single-token input and on-demand tensor loading
- the runtime can now also chain that bridge across multiple real layers, with a live two-layer Qwen pass already working
- the runtime can now enter that bridge from a real token id through a low-memory embedding-row lookup, and a live token-id-to-layer-stack path is working too
- the runtime can now continue from that hidden state through final norm and a streamed `lm_head` pass, so a live token-id-to-logits path is working too
- the runtime can now also run a first repeated greedy decode loop on top of that path, and it now carries a small recent-token history summary even though it is still not true KV/cache-aware decoding
- the runtime now also supports a pluggable next-token policy layer, a reusable comparative decode benchmark with regression-style summaries, a first real K/V-carrying loop with RoPE applied on the live key path through an explicit stop-aware decode-state object, and a first real text-prompt entry path with instruct-style prompt wrapping, local generation defaults, and basic prompt/session controls
- the runtime now also has a real GGUF/llama.cpp backend path for Queen/Qwen, using a local merged Qwen2.5-14B-Instruct Q4_K_M GGUF artifact and a standalone llama.cpp runtime
- the web app now has a `GGUF` mode that can talk to a persistent local `llama-server`, giving fast short responses after the model is already loaded
- the Load Model screen can now load and unload the GGUF server so the user can free the RAM used by the power-user backend
- the desktop test UI now also includes a real prompt test panel with prompt input, system prompt override, raw-prompt mode, max-new-tokens control, custom stop-token parsing, and generated output/details wired to the live runtime
- prompt generation now defaults to a more conservative runtime profile by keeping prompt runs greedy unless sampling is explicitly requested and by choosing a model-aware automatic layer budget when no prompt layer count is forced
- the automatic prompt layer budget is now deeper for short prompt sessions, so the real Qwen prompt path can use 8 carried layers by default instead of staying artificially shallow
- prompt prefill now uses a real multi-token causal path in the layer bridge instead of stepping token-by-token, so prompt sessions hand a more faithful cached state into generation and run faster
- the automatic short-prompt fidelity budget is now raised again, and the live real-Qwen prompt path can use 12 carried layers by default on this machine
- the automatic short-prompt fidelity budget is now raised again to a more serious default, and the live real-Qwen prompt path can use 24 carried layers by default for short prompt sessions
- the automatic short-prompt fidelity budget now uses the full carried stack by default for short prompt sessions, and the current live real-Qwen baseline can produce `Hello! How can` on the short `hello world` test
- the runtime now also supports stronger generation-session controls including `top_p`, `min_new_tokens`, and prompt stop strings, and the desktop app now includes a simple chat-style test surface with persistent local turn history
