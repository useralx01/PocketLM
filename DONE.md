# Done

- Created a dedicated `pcketlm` tracking category in Mission Control
- Set up plain-English tracker files for status, todo, done, errors, decisions, and log notes
- Defined the official `pcketlm` V1 blueprint
- Defined the first engineering roadmap
- Defined the first draft schemas and initial code layout
- Created the first Python package skeleton for the import/runtime core
- Implemented the first validation and registry foundation for model import
- Implemented real source inspection for Qwen config and shard metadata
- Hardened import inspection against partial or invalid JSON metadata files
- Added the first model source status command for local inspection
- Added the first local import command for model folders
- Added early download-state estimation for official source folders
- Added the first registry-backed model catalog/status layer
- Added registry record removal and stale-entry cleanup support
- Debugged the registry/catalog inconsistency and verified the stale entry cleanup succeeded
- Added a product-facing acquisition/progress layer with plain-English summaries and progress bars
- Added the first runtime source-readiness layer so pcketlm can describe what the loader can actually use
- Installed `pytest` and verified the current pcketlm foundation with a passing local test run
- Installed the first lightweight runtime libraries and added a runtime bootstrap/preflight prototype
- Installed `torch` and verified that runtime preflight is now blocked only by incomplete model shards
- Added a real risk register and failure-avoidance rules so likely project failures are tracked explicitly
- Added the first desktop status-screen spec grounded in live acquisition and runtime state
- Created a `pcketlm` desktop shortcut as a placeholder app entry point for later retargeting
- Built the first real desktop test UI and repointed the desktop shortcut to launch it
- Connected the desktop UI to registry-backed model selection with a safe detected-source fallback
- Registered the full official Qwen source and ran the first guarded real-load attempt
- Added reusable blocker category, severity, and recommended-action state across runtime and desktop layers
- Added a reduced-memory strategy planner and verified that staged disk streaming is the best next path on this machine
- Added the first staged disk-streaming planner with adaptive window sizing and cache-layout planning
- Added the first staged-streaming manifest/bootstrap layer and verified it writes real cache state to disk
- Added the first staged-streaming manifest reader and verified the runtime can read back ready streaming state
- Added the first staged-streaming weight-unit mapper and verified it writes a real shard-to-unit map to disk
- Refactored staged-streaming units from full-shard mapping into segment-sized units that fit the real streaming windows
- Added the first hot/warm window scheduler and verified a live schedule can now place units into both windows
- Fixed repo-root import ergonomics so `pytest` and `py -m pcketlm...` work without manual `PYTHONPATH`
- Moved the active `pcketlm` project out of `.openclaw` into `C:\Users\isale\Documents\pcketlm`
- Repointed the desktop shortcut to the new `Documents` project root and removed the old `.openclaw` copy
- Fixed registry path relocation after the project move so desktop status uses the new `Documents` model source instead of the deleted old root
- Fixed streaming manifest path relocation after the project move so staged streaming uses the new `Documents` cache and model paths
- Added the first staged-streaming segment materializer and verified it writes real hot and warm cache files from shard byte ranges
- Added lightweight cache verification with checksums and on-demand cache-index verification for staged streaming
- Added the first staged-streaming rotation flow and verified it promotes warm to hot, refills warm, and re-materializes the live cache
- Added persistent staged-streaming residency state with rotation step, current hot/warm units, consumed units, overflow head, and refill count
- Added cache hit/miss accounting to staged-streaming verification and locked in a regression test so verification no longer resets rotation/refill state
- Repaired the live staged-streaming residency state so the current cache position, refill count, and cache-hit telemetry are internally consistent again
- Added a staged-streaming control layer that chooses the next runtime action from live residency telemetry and can execute that action directly
- Surfaced staged-streaming control state and telemetry inside the desktop app, including a live `Advance Stream` action
- Fixed the desktop app layout so the larger status view scrolls correctly and the main action buttons stay visible in a fixed bottom bar
- Added a verify-first `Safe Advance` runtime path and surfaced it in the desktop app as a separate safer streaming action
- Added a tensor-aware safetensors catalog layer that persists real tensor names, dtypes, shapes, offsets, layer IDs, and shard placement for runtime use
- Added a tensor-aware execution-plan layer that groups tensors into ordered runtime units such as embeddings, per-layer norm/attention/MLP, and decode head
- Added the first tensor loader slice that can load one real tensor or one grouped execution unit from the original shards on demand
- Added a stronger tensor verification pass that validates loaded tensors and grouped execution units against catalog and execution-plan metadata
- Loaded and verified the first larger real execution unit from the live model: `layer-00-attention`
- Added the first minimal CPU-only layer-forward bridge that performs real layer-0 Qwen math from synthetic single-token input
- Reworked the layer bridge to load required tensors one at a time so the first real execution slice works without holding full attention and MLP units in memory at once
- Ran the first live real execution slice for `qwen2.5-14b-instruct`, producing a `1 x 1 x 5120` output tensor with no blockers
- Extended the layer bridge into a stack runner that can chain multiple real layers while reusing the same low-memory one-tensor-at-a-time load path
- Ran the first live two-layer real execution slice for `qwen2.5-14b-instruct`, successfully chaining layers `0` and `1` with no blockers
- Proved the stacked bridge can go deeper by running a live four-layer real execution slice for `qwen2.5-14b-instruct`, successfully chaining layers `0` through `3`
- Added a low-memory token-entry bridge that reads only the needed embedding row for a token id instead of loading the full embedding table
- Ran the first live token-entry execution path for `qwen2.5-14b-instruct`, starting from token id `42` and successfully chaining layers `0` and `1`
- Added a low-memory decode tail that applies final norm and streams `lm_head` rows in chunks to build real logits
- Ran the first live token-to-logits execution path for `qwen2.5-14b-instruct`, starting from token id `42` and producing a real `1 x 1 x 152064` logits tensor
- Added the first repeated greedy decode loop on top of the token-entry + layer-stack + decode-tail path
- Ran the first live repeated decode loop for `qwen2.5-14b-instruct`, producing the two-step chain `42 -> 123571 -> 102664`
- Strengthened the repeated loop with an explicit recent-token history summary state so it no longer relies only on the newest token
- Ran the first live history-summary repeated decode loop for `qwen2.5-14b-instruct`, producing the two-step chain `42 -> 123571 -> 117864` with `history_window=3`
- Added a token-selection policy layer with greedy and deterministic top-k sampling on top of the decode path
- Ran the first live sampled loop for `qwen2.5-14b-instruct`, producing the two-step chain `42 -> 123571 -> 123571`
- Added the first real K/V-carrying decode step and repeated loop so projected keys and values persist across steps per layer
- Ran the first live K/V-aware loop for `qwen2.5-14b-instruct`, producing the two-step chain `42 -> 123571 -> 10862`
- Added a small comparative decode benchmark for the live decode paths so history-summary and K/V-aware behavior can be checked side-by-side on the same seed
- Added RoPE-aware key-path handling to the first real K/V-carrying decode loop
- Ran the first live RoPE-aware K/V loop for `qwen2.5-14b-instruct`, producing the two-step chain `42 -> 123571 -> 46322`
- Expanded the decode benchmark with reusable regression-style summaries like step count, final token, and unique-token count
- Reworked the RoPE-aware K/V path to carry an explicit decode-state object with next token, next position, generated chain, and per-layer cache lengths
- Made the decode-state path stop-aware by loading EOS metadata from model config / generation config and exposing finished state plus stop reason in the K/V loop and benchmark
- Added the first real prompt/tokenizer entry path on top of the stop-aware K/V decode session
- Ran the first live prompt-entry generations for `qwen2.5-14b-instruct`, starting from real text prompts instead of raw token ids
- Improved the first prompt/session path with instruct-style prompt wrapping, local generation defaults from `generation_config.json`, and faithful forced-token prompt prefill
- Added the first real prompt/session control layer with explicit repetition penalty, custom system prompt support, and raw-prompt mode
- Added max-new-tokens and explicit stop-token control to the prompt runtime path and CLI
- Added a real desktop prompt test panel with prompt input, system prompt override, raw-prompt mode, max-new-tokens input, stop-token parsing, generated output, and session-detail reporting
- Improved prompt runtime defaults so plain prompt runs now stay greedy unless sampling is explicitly requested, and prompt generation can automatically use a model-aware deeper layer budget when no explicit layer count is forced
- Deepened the automatic prompt layer budget for short sessions so the real Qwen prompt path can now use 8 carried layers by default instead of staying at 4
- Added real multi-token causal support to the layer bridge and switched prompt prefill from token-by-token stepping to a true multi-token prefill slice
- Raised the automatic short-prompt layer budget again so the real Qwen prompt path now uses 12 carried layers by default on this machine
- Raised the automatic short-prompt layer budget again so the real Qwen prompt path now uses 24 carried layers by default on this machine while keeping a generic prompt-length-sensitive heuristic
- Raised the automatic short-prompt layer budget again so the real Qwen prompt path now uses the full carried stack by default on this machine, producing the first believable short-prompt baseline
- Added the first product-level model-family metadata layer with owner-priority ordering for Qwen, Kimi, Kronos/Kronk, and Gemma
- Connected the desktop status surface to model-family support status so Qwen is marked active and future families can be marked planned or unverified without hardcoding UI paths
- Normalized imported model family aliases so registry records can map values like `qwen2`, `kronk`, and `gemma3` to stable family keys
- Added the first desktop `Model Home` action summary so imported models are presented as flexible surfaces for chat, personalization, comparison, inspection, and benchmarking instead of one forced workflow
- Verified the current project after the universality/model-home slice with `77` passing tests
- Turned the desktop `Model Home` summary into clickable actions for Chat, Personalize, Compare, Inspect, and Benchmark
- Wired Chat to the real local chat surface and Inspect to the existing details view
- Added the first benchmark readiness backend so the Benchmark action can report whether lightweight checks are ready before the full benchmark engine exists
- Added the first safe personalization profile templates: Balanced Local, Agent Coder, and Low Memory
- Softened the desktop chat wording from test-panel language into product-facing chat language
- Verified the project after the model-home/profile/benchmark slice with `82` passing tests
- Fixed the desktop chat freeze issue by running local prompt generation in a background thread, disabling Send while work is active, and showing a clear working state
- Verified the real Qwen prompt path after the desktop threading fix with `hello world`, `max_new_tokens=2`, yielding `Hello!` with no blockers
- Changed the desktop chat default to `2` max new tokens and added a visible alpha-runtime speed hint so first tests do not look stuck
- Verified the real Qwen prompt path again after the desktop default change; `hello world` with `max_new_tokens=2` still yielded `Hello!` with no blockers in about `67` seconds
- Added a desktop runtime mode selector with Fast `8` layers, Balanced `32` layers, and Quality full-stack modes
- Changed the desktop chat default mode to Balanced `32` layers for faster alpha testing while keeping Quality available for the best current output
- Added mtime-aware caching for persisted tensor catalog and tensor execution-plan reads
- Profiled the prompt path and found the main remaining cost is tensor conversion and CPU linear math, especially repeated `.float()` conversion of loaded bf16 weights and `torch._C._nn.linear`
- Verified Balanced mode with a real Qwen smoke check: `hello world`, `max_new_tokens=2`, `32` layers, ready with no blockers in about `47` seconds, but output quality drifted compared with full-stack Quality mode
- Added PyTorch inference mode around the heavy layer stack, decode tail, K/V decode step, and prompt decode loop paths
- Added mtime-aware caching for small layer-bridge config and generation-config JSON reads
- Improved the Balanced-mode real Qwen smoke check from about `47` seconds to about `44` seconds for `hello world` with `max_new_tokens=2`
- Added persisted lightweight benchmark runs under each model's `benchmarks` directory, including a latest benchmark snapshot
- Wired the desktop Benchmark action to save a lightweight benchmark result instead of only showing readiness
- Added a memory-capped tensor residency layer for converted CPU tensors, including a default `256 MB` total cap, `32 MB` per-tensor cap, LRU eviction, and hit/miss/store/eviction/skip counters
- Routed layer-bridge tensor conversion through the residency layer so repeated per-layer tensor requests can reuse safe converted tensors instead of always converting immediately in the bridge
- Added tensor residency tests for cache reuse, oversized-tensor skipping, and cache reset behavior
- Verified the project after tensor residency with `88` passing tests
- Re-ran the real Balanced-mode Qwen smoke check after tensor residency; `hello world` with `max_new_tokens=2` and `32` layers stayed ready with no blockers
- Added tensor residency stats to persisted lightweight benchmark runs and the desktop Benchmark action
- Verified the project again after benchmark-stat wiring with `88` passing tests
- Re-ran the real Balanced-mode Qwen smoke check after benchmark-stat wiring; `hello world` with `max_new_tokens=2` and `32` layers stayed ready with no blockers in about `48` seconds
- Measured the first broad tensor cache policy on a real in-process Balanced Qwen run and found it had `0` cache hits with `406` evictions
- Reworked tensor cache admission to keep the front layer window warm by default instead of caching every tensor that fits
- Measured the front-layer tensor cache policy on a real in-process Balanced Qwen run and confirmed it produced `43` hits, `0` evictions, and about `252 MB` resident tensors
- Tested a larger per-tensor cache variant and kept the safer front-layer default because the larger variant did not improve enough to justify the extra memory risk
- Verified the project after front-layer cache tuning with `89` passing tests
- Re-ran the real Balanced-mode Qwen CLI smoke check after front-layer cache tuning; `hello world` with `max_new_tokens=2` and `32` layers stayed ready with no blockers
- Measured real Qwen projection math and found `bfloat16` CPU linear was much faster than `float32` for the same loaded Qwen projection on this machine
- Added a runtime math dtype switch with `bfloat16` and `float32` modes
- Fixed same-dtype tensor residency so bfloat16 tensors loaded from safetensors are cloned into owned CPU memory instead of referencing a closed shard handle
- Promoted `bfloat16` to the default runtime math dtype after the real Balanced Qwen smoke check produced the same generated tokens with a much faster runtime
- Raised the default lm_head decode-tail chunk size from `4096` to `8192` rows after live checks showed it was the best current chunk size among `4096`, `8192`, and `16384`
- Added runtime setting visibility to lightweight benchmark snapshots and the desktop Benchmark action
- Verified the project after bfloat16/chunk-size optimization with `91` passing tests
- Re-ran the real Balanced-mode Qwen CLI smoke check after bfloat16/chunk-size optimization; `hello world` with `max_new_tokens=2` and `32` layers stayed ready with no blockers in about `38.5` seconds
- Rechecked Quality full-stack after the bfloat16 speed change; `hello world` with `max_new_tokens=2` generated `Hello!` with no blockers in about `60` seconds
- Changed desktop chat to default to Quality full-stack mode because it gives the best current answer behavior
- Reworded desktop mode hints so Fast is framed as smoke testing, Balanced as a speed preview, and Quality as the best current output mode
- Verified the project after the desktop default/mode-hint change with `91` passing tests
- Re-ran the real default Quality-mode Qwen smoke check; `hello world` with `max_new_tokens=2` generated `Hello!` with no blockers in about `60` seconds
- Added persisted measured benchmark runs with timed Fast, Balanced, and Quality prompt checks
- Wired the desktop Benchmark action to run measured benchmarks in a background thread instead of freezing the app
- Fixed benchmark readiness handling so plain CPU RAM limitations become warnings when the direct measured runtime path completes successfully
- Verified the project after measured benchmark wiring with `93` passing tests
- Ran and saved a real measured Qwen benchmark: Fast about `12.3` seconds with rough output, Balanced about `39.7` seconds with odd output, and Quality about `57.1` seconds with `Hello!`
- Added a new local web UI shell based on the downloaded Pocket LLM dark sidebar design
- Added a Python local web server with API routes for live status, chat, and measured benchmark runs
- Repointed `launch_pcketlm.pyw` from the old Tkinter UI to the new local web UI
- Wired the new web Chat screen to the real Qwen prompt runtime
- Wired the new web Benchmarks screen to the real measured benchmark backend
- Wired the new web Load Model, Personalize, Agents, and Settings screens to current project state or structured placeholders
- Verified the new web helper layer and existing project with `95` passing tests
- Ran a local web-server smoke check confirming the new HTML loads, Qwen is active, one model is visible, and all three built-in profile templates are visible
- Ran a real Qwen web chat API smoke check: Quality mode generated `Hello!` from `hello world` with no blockers
- Added bounded recent conversation history support to the web chat API so follow-up messages are no longer isolated single prompts
- Wired the web chat UI to send recent user/assistant turns, show a live timer while generation runs, and default to `4` max new tokens
- Verified the project after conversation-history web chat wiring with `98` passing tests
- Ran a real Qwen web chat API smoke check with history included; the route returned ready with no blockers and confirmed `2` previous turns were used
- Changed web chat history formatting from a transcript inside one user message to real Qwen `system/user/assistant` chat turns when Qwen chat tokens are available
- Added a Pocket LLM web-system prompt that keeps default answers brief, direct, and English unless the user asks otherwise
- Changed the prompt layer-budget heuristic so short Quality chat conversations stay on the full 48-layer stack instead of dropping layers once the prompt passes 32 tokens
- Verified the project after the web chat quality fix with `101` passing tests
- Ran a real Qwen web memory smoke check: after `My name is Sam`, the follow-up `What is my name?` generated `Your name is Sam`, ready with no blockers
- Added background web chat jobs with `/api/chat/start`, `/api/chat/status`, and `/api/chat/cancel` while keeping the original blocking `/api/chat` route for compatibility
- Added web chat Cancel and Retry controls plus polling-based elapsed status so long local generation no longer depends on one blocked browser request
- Verified the project after background chat jobs with `103` passing tests
- Ran a real Qwen background-job smoke check through start/status polling; `hello world` in Quality mode generated `Hello`, ready with no blockers
- Added cooperative runtime cancellation hooks to the prompt decode loop, layer stack, K/V decode step, and streamed decode tail
- Wired web chat jobs to pass their cancel state into the runtime so Cancel can stop between expensive runtime phases instead of only discarding the final response
- Verified the project after runtime cancellation with `104` passing tests
- Ran a real Qwen cancel smoke check; a Quality job reached `canceled` in about `4` seconds after starting, and the next normal completion smoke still generated `Hello` with no blockers
- Added automatic PyTorch CPU thread configuration for runtime math, with `PCKETLM_TORCH_THREADS` as an override
- Tuned the current 16-core machine to use `14` runtime threads by default, preserving the full-stack Quality output while slightly improving real 2-token timing
- Added runtime thread count to benchmark runtime settings
- Added a fixed web app port and single-instance lock port so repeated launches reuse one local Pocket LLM server instead of creating duplicate servers on random ports
- Verified the project after runtime thread and launcher singleton work with `107` passing tests
- Ran a live web-job Qwen smoke check through the fixed app port; Quality generated `Hello`, ready with no blockers, in about `25.6` seconds for `1` token
- Added detailed prompt runtime phase timings, including prefill stack, decode tail, token selection, continuation steps, stack average layer time, and slowest layer time
- Added a streamed top-k decode-tail path so greedy and top-k sampling can avoid materializing the full logits tensor when top-k candidates are enough for selection
- Added timing and streamed top-k coverage to the runtime tests, and isolated the web single-instance lock test from the live app port
- Verified the full project after the timing and decode-tail optimization work with `108` passing tests
- Re-ran the clean real Qwen Quality check: `hello world`, `max_new_tokens=2`, generated `Hello!`, ready with no blockers, in about `49.6` seconds
- Verified the live web background-job route on `http://127.0.0.1:8765`; Quality generated `Hello`, ready with no blockers
- Added all-layer small-tensor residency so tiny weights and biases can stay cached across the full 48-layer path without caching large projection matrices
- Added cached RoPE trig tables for repeated position/head-dimension pairs and avoided an extra float conversion inside RoPE application
- Kept token-entry hidden states in the active runtime math dtype instead of building them as float32 and converting again later
- Added a normal-chat fast path that skips extra per-layer summary reductions while preserving detailed layer summaries for tests and debug callers
- Verified the project after the full-stack speed pass with `109` passing tests
- Re-ran the clean real Qwen Quality check: `hello world`, `max_new_tokens=2`, generated `Hello!`, ready with no blockers, in about `42.7` seconds
- Restarted the web app and verified the live background-job route; Quality generated `Hello`, ready with no blockers, in about `21.0` seconds for `1` token
- Added a runtime details panel to web chat so completed runs show total time, stack timing, decode-tail timing, torch threads, math dtype, lm-head chunk size, and tensor cache stats
- Added runtime settings to the Settings screen using the same live backend payload
- Changed web status copy to report the direct Pocket runtime as working when the direct runtime path is usable, while still preserving the older full-RAM preflight status internally
- Verified the project after runtime visibility/status cleanup with `109` passing tests
- Restarted the web app and verified `/api/status` reports `Working` for the direct runtime with `bfloat16`, `14` torch threads, and `8192` lm-head chunk rows
- Ran a live web background-job smoke check after the cleanup; Quality generated `Hello`, ready with no blockers, in about `22.9` seconds for `1` token and returned runtime timing/settings data
- Polished web chat cancellation so canceled runs remove the temporary assistant placeholder and leave Retry available instead of writing a fake assistant reply into the transcript
- Upgraded the Benchmarks screen from simple rows to richer cards showing generated text, layer count, ready/blocked state, tensor-cache hits, resident cache size, and latest runtime settings
- Reworked the Load Model screen to show the active runtime, local model folder, and support plan for Qwen, Kimi, Kronos/Kronk, and Gemma
- Verified the project after chat/benchmark/load product cleanup with `109` passing tests
- Restarted the web app and verified the new HTML/CSS markers are served from `http://127.0.0.1:8765`
- Ran a live cancel smoke check after the UI cleanup; the job reached `canceled`, kept no result, and had no error
- Ran a live normal web background-job smoke check after cancellation; Quality generated `Hello`, ready with no blockers, in about `23.6` seconds for `1` token
- Added persisted per-model profile records with JSON storage under each model's `profiles` directory
- Added default profile seeding from the three free templates: Balanced Local, Agent Coder, and Low Memory
- Added a profile comparison summary that uses saved profiles and the latest measured benchmark when available
- Updated `/api/status` to return saved profiles and the profile comparison payload
- Added a Compare screen to the web UI and changed Personalize to show saved profile records instead of only template cards
- Verified the profile and web changes with `111` passing tests
- Restarted the web app and verified Qwen has the three saved profile JSON records plus `profile_compare.profile_count: 3`
- Ran a live normal web background-job smoke check after saved profile/Compare wiring; Quality generated `Hello`, ready with no blockers, in about `25.1` seconds for `1` token
- Added operation-level runtime timing for the full-stack layer path, including tensor loading, attention pieces, MLP, and decode phases
- Switched attention to PyTorch scaled-dot-product attention by default with a manual fallback and cached causal masks for repeated shapes
- Tested persistent safetensors handle caching on the real Qwen run, found it could crash silently, and made it opt-in instead of default
- Added safe batched tensor loading by shard for grouped tensor misses
- Wired the layer bridge to batch-load norms, Q/K/V tensors, and MLP projection weights while keeping the output projection separate after measurement showed batching it was slower
- Added a real GGUF backend path for Queen/Qwen using the official standalone llama.cpp Windows CPU runtime.
- Downloaded and merged the official `Qwen2.5-14B-Instruct` Q4_K_M GGUF split files into a single local GGUF artifact under the model artifacts folder.
- Added Pocket runtime detection for `llama-cli.exe`, `llama-server.exe`, main-process `llama_cpp`, and the Python 3.12 sidecar environment.
- Added web `GGUF` mode so chat can route through the GGUF backend instead of the older safetensors direct runtime.
- Added persistent `llama-server` support so the GGUF model can stay loaded between prompts instead of paying the full load cost each time.
- Fixed the web app so GGUF mode bypasses the old direct-runtime RAM guard after the server is already loaded.
- Verified the live GGUF web path through `http://127.0.0.1:8765`: the app returned `READY`, `ready: true`, `strategy: llama-cpp-gguf-server`, and `stop_reason: gguf-complete`.
- Verified the project after the GGUF backend phase with `158` passing tests.
- Added GGUF server lifecycle APIs so Pocket can load, unload, and report the persistent llama.cpp server.
- Added GGUF server controls to the Load Model screen, including ready/unloaded state, PID/RAM when available, artifact path, and Load/Unload buttons.
- Optimized the GGUF chat response metadata path so chat does not run the slower full process/RAM probe after every answer.
- Verified the new lifecycle path live: unload succeeded, cold reload through Pocket took about `45.75s`, and the reloaded server answered a fresh GGUF prompt in about `1.07s` wall time.
- Verified the project after the GGUF control phase with `161` passing tests.
- Fixed GGUF prompt formatting so Qwen-Instruct receives proper chat-formatted `system/user/assistant` prompts instead of raw completion text.
- Verified the UI bug case after the fix: `Reply with OK only.` returned `OK`.
- Verified the project after the GGUF prompt-format fix with `162` passing tests.
- Added GGUF benchmark cases for 1-token, 4-token, 8-token, 16-token, basic logic, and short agent-style prompts.
- Added a dedicated `Run GGUF` benchmark action in the web app so fast GGUF results can be saved without running the older slow Direct CPU benchmark.
- Persisted GGUF benchmark results in the existing measured benchmark format so the Benchmarks screen and history can display them.
- Ran the live GGUF benchmark and saved results; the benchmark was ready, with the fastest case at `0.54s` and the logic check returning `YES`.
- Verified the project after the GGUF benchmark phase with `164` passing tests.
- Added stop-string support to the GGUF backend for server, Python package, sidecar, and CLI paths so Qwen chat markers do not leak into visible answers.
- Raised GGUF web chat to a `64` token cap while keeping direct CPU modes capped at `16`, so loaded GGUF can produce short complete answers without changing the weak-hardware direct path.
- Updated the GGUF benchmark to include a `32` token sentence check and a `48` token agent-style checklist check.
- Live GGUF longer-answer smoke returned two complete numbered local-model checking steps through `llama-cpp-gguf-server` in about `21.57s`.
- Live saved GGUF benchmark now returns complete `GGUF 32` and `GGUF Agent` answers instead of cutting off at the token limit.
- Verified the project after the GGUF longer-answer phase with `167` passing tests.
- Added regression tests for batched tensor loading and batched tensor residency reuse
- Re-ran the full unit suite after the heavy loading pass: `113` tests passed
- Re-ran the clean real Qwen Quality CLI check; `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `40.5` seconds
- Restarted the web app and re-ran the real background-job smoke check; Quality generated `Hello`, ready with no blockers, in about `20.7` seconds for `1` token
- Raised the default tensor residency front-layer cache window from `6` to `12` while keeping the same `256 MB` total cache cap
- Added regression coverage for the default `12`-front-layer residency policy
- Re-ran focused runtime/tensor tests after the cache-window pass: `39` tests passed
- Re-ran the full unit suite after the cache-window pass: `114` tests passed
- Re-ran the clean real Qwen Quality CLI check; `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `36.1` seconds, with about `253 MB` resident cache and no evictions
- Re-ran a longer real Qwen Quality CLI check; `hello world` with `max_new_tokens=4` generated `Hello! How can`, ready with no blockers, in about `85.0` seconds
- Restarted the web app again and re-ran the real background-job smoke check; Quality generated `Hello`, ready with no blockers, in about `26.9` seconds for `1` token while free RAM was very low
- Added a low-memory guard to the tensor residency policy: below `2 GB` free RAM, the runtime automatically falls back to a safer `128 MB` cache and `6` front layers unless explicit environment overrides are set
- Cached the memory snapshot used by the guard so repeated tensor-policy checks do not slow generation
- Exposed the effective tensor residency policy in web runtime settings, including guard state, free memory, max cache size, and front-layer count
- Added regression coverage for the low-memory fallback, explicit overrides, and the default front-layer policy
- Re-ran focused runtime/web/tensor tests after the guard pass: `47` tests passed
- Re-ran the full unit suite after the guard pass: `116` tests passed
- Re-ran the standalone real Qwen Quality CLI check after the guard pass; `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `41.5` seconds
- Restarted the web app and re-ran the real background-job smoke check after the guard pass; Quality generated `Hello`, ready with no blockers, in about `21.9` seconds for `1` token
- Added measured benchmark history summaries built from saved benchmark JSON runs, with best, average, and worst timing per mode
- Added benchmark history to the web status payload and Benchmarks screen
- Added saved-profile lookup and light profile behavior for chat requests, so a selected profile can provide runtime mode and default token settings
- Added a profile picker to the web Chat composer
- Added cache-policy and cache-cap rows to runtime details
- Re-ran focused benchmark/profile/web tests after the medium groundwork pass: `20` tests passed
- Re-ran the full unit suite after benchmark-history/profile behavior wiring: `118` tests passed
- Re-ran two short realistic CLI prompt checks; both were ready with no blockers, while their short outputs showed quality needs a deeper decode pass
- Restarted the web app and verified `/api/status` exposes direct runtime `Working`, `2` benchmark history runs, `3` saved profiles, and memory guard state
- Ran a web smoke through the `low-memory` profile; it completed ready with no blockers and used the profile route, but output quality was rough/mixed-language
- Changed the Low Memory built-in profile to default to the Quality runtime path instead of the rough Balanced preview path, while keeping short output defaults and high memory priority
- Added default-profile refresh logic so existing saved built-in profile JSON records migrate when their template runtime defaults change
- Added Qwen chat stop markers to web chat runtime calls and trimmed generated stop markers from visible assistant text in the runtime result
- Re-ran focused runtime/profile/web tests after the high-phase quality pass: `44` tests passed
- Re-ran the full unit suite after the high-phase quality pass: `120` tests passed
- Restarted the web app on `http://127.0.0.1:8765`; `/api/status` showed direct runtime `Working` and Low Memory runtime mode `Quality`
- Ran a live Low Memory web smoke after restart. With `max_new_tokens=1`, it generated `Hello`, ready with no blockers, in about `26.6` seconds. An earlier 3-token smoke generated `Hello! How`, ready with no blockers, in about `57.6` seconds
- Added timing summaries to measured benchmark cases, including stack time, tensor-load time, decode-tail time, total time, and current bottleneck
- Added timing-summary aggregation to benchmark history so fresh measured runs can compare average stack, tensor-load, and decode-tail costs by mode
- Updated the Benchmarks screen to show timing chips for fresh benchmark cases and history labels when the saved data includes timing summaries
- Re-ran focused benchmark and web tests after the medium timing prep: `16` tests passed
- Re-ran the full unit suite after the medium timing prep: `120` tests passed
- Ran `python -m compileall src tests -q` cleanly after the medium timing prep
- Restarted the web app on `http://127.0.0.1:8765`; the latest live Low Memory web smoke generated `Hello`, ready with no blockers, in about `19.0` seconds for `1` token, with timing total about `19.0s` and prefill stack about `17.7s`
- Added a runtime identity system prompt for web chat so Qwen sees Pocket LLM context, current loaded model, runtime mode, and active profile on every request
- Added `runtime_context` to web chat responses so the app can confirm the model/profile/mode used by a run
- Cached web chat-format detection per model id and added an mtime-aware JSON metadata cache for tokenizer/generation config reads
- Added regression tests for runtime identity context and chat payload propagation
- Re-ran focused web/runtime tests after the identity pass: `40` tests passed
- Re-ran the full unit suite after the identity pass: `121` tests passed
- Ran `python -m compileall src tests -q` cleanly after the identity pass
- Restarted the web app on `http://127.0.0.1:8765` and ran a model-identity smoke. It generated `Qwen2.5-14`, ready with no blockers, in about `184.8` seconds for `8` tokens, while response context reported `Qwen2.5-14B-Instruct`
- Ran a basic logic smoke. It generated `YES` for `If 2 plus 3 equals 5, answer YES only.`, ready with no blockers, in about `27.0` seconds for `1` token
- Added a local runtime-context answer path for clear model-identity questions, so Pocket LLM can answer known app/runtime facts without spending a Qwen generation
- Added mtime-aware caching for the built layer bridge config object used during stack execution
- Added regression coverage proving model-identity questions short-circuit without calling `run_prompt_decode_loop`
- Re-ran focused web/runtime tests after the speed shortcut: `42` tests passed
- Re-ran the full unit suite after the speed shortcut: `123` tests passed
- Ran `python -m compileall src tests -q` cleanly after the speed shortcut
- Restarted the web app on `http://127.0.0.1:8765`; `Which model do you run on?` returned `Qwen2.5-14B-Instruct (qwen2.5-14b-instruct)` in about `0.015` seconds with no blockers
- Re-ran the real Qwen logic path after the shortcut; it still generated `YES`, ready with no blockers, in about `29.0` seconds for `1` token
- Raised the tensor residency low-memory guard threshold from `2 GB` to `3 GB`, with an advanced override through `PCKETLM_TENSOR_CACHE_LOW_MEMORY_GUARD_MB`
- Added a hard web-chat memory guard that blocks model generation below `2 GB` free RAM and returns a clear blocker instead of starting a likely-crashing Qwen run
- Added regression coverage for the `3 GB` residency guard threshold, threshold override, and web memory-guard response
- Re-ran focused web/tensor/runtime tests after the hard memory guard: `56` tests passed
- Re-ran the full unit suite after the hard memory guard: `127` tests passed
- Ran `python -m compileall src tests -q` cleanly after the hard memory guard
- Restarted the web app on `http://127.0.0.1:8765`; with about `0.74 GB` free RAM, identity answered instantly and real Qwen generation returned a clean `memory-guard` blocker instead of crashing
- Added an mtime-aware tensor-name index shared by the tensor catalog, tensor loader, tensor residency path, and layer bridge so hot lookups do not repeatedly scan the full catalog.
- Added regression coverage proving the tensor-name index matches the persisted catalog.
- Re-tested persistent safetensors handle caching as a default/batched-load speed lever; the live Qwen request dropped the connection and killed the server, so the handle cache remains opt-in.
- Shortened the web runtime identity prompt while preserving app/model/mode/profile facts and the local model-identity instruction.
- Re-ran focused web/runtime/tensor tests after the speed pass: `36` tests passed.
- Re-ran the full unit suite after the speed pass: `127` tests passed.
- Restarted the web app on `http://127.0.0.1:8765` and re-ran the real Low Memory/Quality logic smoke. It returned `YES`, ready with no blockers, with prompt tokens reduced from `135` to `121`; timings were about `23.1s` cold-ish and `22.1s` warm for `1` token.
- Added an experimental request-scoped safetensors handle cache and proved it reuses one handle across load calls in unit coverage.
- Tested the request-scoped handle cache in the real web Qwen path; it still crashed the server, so the prompt loop now keeps that path behind `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=1` and leaves it off by default.
- Raised the web chat hard memory guard from `2 GB` to `3 GB` after a real run crashed the server around `2.8 GB` free RAM.
- Tightened the injected runtime identity prompt again. The same Low Memory/Quality logic smoke now uses `76` prompt tokens, down from `121` after the previous pass and `135` before this speed work.
- Re-ran focused web/runtime/tensor tests after the tighter prompt and guard pass: `50` tests passed.
- Re-ran the full unit suite after the tighter prompt and guard pass: `128` tests passed.
- Restarted the web app on `http://127.0.0.1:8765` and re-ran the same real Low Memory/Quality logic smoke twice. It returned `YES`, ready with no blockers, in about `22.8s` cold-ish and `21.3s` warm for `1` token.
- Added cumulative tensor-load diagnostics to the tensor loader and web runtime settings. The app can now report tensor requests, loaded bytes, shard opens, and handle-reuse counters.
- Added conversation-state metadata to chat responses. It currently reports bounded recent prompt replay and explicitly marks true KV reuse as not ready yet.
- Added the first planning-only optimized artifact manifest layer under `pcketlm.core.optimize`.
- Built the first real Qwen Low Memory artifact plan at `models/qwen2.5-14b-instruct/artifacts/runtime-pack-plan.low-memory.artifact.json`; it is ready and covers `579` source tensors, `8` shards, and `48` layers.
- Ran a live Qwen pressure smoke with diagnostics. It still answered `YES`, but under memory pressure it took about `97.7s`; diagnostics showed `577` tensors loaded, about `25.2 GB` moved, and `198` shard opens.
- Raised the hard web generation guard to `4 GB` free RAM. At about `3.13 GB`, the same prompt now returns a clean `memory-guard` response in under a second instead of starting a pressure run.
- Re-ran focused tests after the diagnostics/artifact pass: `24` tests passed.
- Re-ran the full unit suite after the diagnostics/artifact/guard pass: `129` tests passed.
- Extended the optimized artifact manifest into a real small-tensor safetensors runtime pack.
- Added artifact-aware tensor loading. The loader now prefers packed tensors when they exist and falls back to original model shards when they do not.
- Added runtime-pack diagnostics for artifact pack opens and artifact tensor hits.
- Added tests proving the pack builder writes a small safetensors pack and the runtime loader uses it.
- Built the real Qwen Low Memory small runtime pack at `models/qwen2.5-14b-instruct/artifacts/runtime-pack-plan.low-memory.small.safetensors`; it contains `97` tensors and is about `0.95 MB`.
- Re-ran focused runtime/artifact/web tests after the real pack path: `26` tests passed.
- Re-ran the full unit suite after artifact-aware loading: `131` tests passed.
- Restarted the app and verified status reports the artifact ready with `97` packed tensors.
- Ran two real artifact-backed Low Memory/Quality smokes. Both returned `YES` with no blockers; the first took about `22.3s`, the warm repeat about `20.9s`. Diagnostics showed `97` artifact tensor hits on the first run and shard opens reduced to `148`.
- Added Tier 2 artifact selection for front-layer Q/K/V projection tensors.
- Added test coverage proving the front-attention pack includes only the configured front layer count.
- Rebuilt the real Qwen Low Memory runtime pack with small tensors plus front `2` layers of Q/K/V projection tensors.
- The Tier 2 Qwen pack now contains `109` tensors and is about `140.97 MB`.
- Re-ran focused artifact/runtime/web tests after Tier 2: `27` tests passed.
- Re-ran the full unit suite after Tier 2: `132` tests passed.
- Restarted the app and verified `/api/status` sees the Tier 2 artifact as ready with `109` packed tensors.
- Verified the live generation guard after Tier 2: with about `2.38 GB` free RAM, the Qwen logic prompt returned `memory-guard` instead of starting a risky run.
- Added runtime-pack auto-selection. Pocket LLM now chooses the largest ready pack that fits a conservative free-RAM budget instead of blindly using the newest pack.
- Added web `speed_status` diagnostics with pack selection, resident-cache use, loaded MB, shard opens, and artifact hits.
- Built a real Qwen `speed-core` pack with front `4` layers of Q/K/V/O attention projections plus small tensors. It contains `125` tensors and is about `481.0 MB`.
- Live validation showed the `481.0 MB` pack is too aggressive for the current laptop state: it pushed free RAM down to about `1.77 GB` during generation and slowed the request to about `54.9s`, so the selector now avoids it below larger free-RAM budgets.
- Re-tested adaptive larger resident caching and rolled it back to the safer `256 MB` / `32 MB` default after live smokes showed cache churn and no useful hit rate improvement.
- Kept the proven safe path active: the app currently selects the `140.97 MB` Low Memory pack, answers the real `OK` smoke with no blockers, and stays inside the RAM guard.
- Re-ran focused runtime/artifact/web tests after the speed-core selector and rollback work: `68` tests passed.
- Re-ran the full unit suite after the speed-core phase: `134` tests passed, and `python -m compileall -q src tests` completed cleanly.
- Added the first safe session/reuse layer: exact chat-result reuse for identical model/profile/mode/prompt/settings requests.
- Cached chat responses now return with `response_reuse.hit: true`, strategy suffix `+response-cache`, and `elapsed_seconds: 0.0` without running Queen again.
- Added response-cache diagnostics to runtime settings and speed status, including entries, hits, misses, stores, evictions, and TTL.
- Added a runtime engine-decision diagnostic. Status now reports the selected engine/backend; on the current machine it selects `direct-cpu` / `torch-cpu` because no supported GPU backend is available to the Python runtime.
- Added tests for exact response reuse, response-cache status, and engine selection.
- Restarted the app and ran live proof: first identical `OK` prompt used Queen; the next direct `/api/chat` call returned from cache in about `0.454s` wall time with model elapsed `0.0s`.
- Re-ran the full unit suite after the reuse/engine phase: `138` tests passed, and `python -m compileall -q src tests` completed cleanly.
- Extended exact-result reuse into the background job path used by the web app. Cached/local/guarded jobs can now return from `/api/chat/start` as already `completed` instead of waiting for the polling loop.
- Updated the web chat UI to finish immediately when `/api/chat/start` returns a completed job, and to show `Ready cached` for reused answers.
- Live proof: after one real Queen `OK` run filled the cache in about `37.83s`, the repeated `/api/chat/start` call returned a completed cached job in about `0.007s` with `job_fast_path: response-cache` and model elapsed `0.0s`.
- Re-ran the full unit suite after the instant cached-job phase: `139` tests passed, and `python -m compileall -q src tests` completed cleanly.
- Added a full-stack `Quick` chat mode that caps generation to `1` token instead of reducing layers, giving agents a faster short-check path without using the rough partial-layer modes.
- Made request-scoped safetensors handle reuse automatic only for one-token runs; multi-token Quality remains on the stable grouped-load path after live tests showed forced scoped reuse is unsafe across continuation decode.
- Reused the selected runtime-pack handle inside scoped one-token requests, so the live Quick path opens the Low Memory pack once instead of repeatedly opening it during a single answer.
- Updated the web UI mode picker and benchmark labels to include `Quick`.
- Added benchmark coverage for the new Quick row and regression coverage proving Quick mode caps tokens without reducing layer count.
- Restarted the app and live-tested the real web API: Quick returned `OK`, ready with no blockers, in about `19.4s` wall time with all `48` layers active; Quality still returned `OK` in about `36.1s` for `2` tokens.
- Re-ran the full unit suite after the Quick-speed phase: `140` tests passed, and `python -m compileall -q src tests` completed cleanly.
- Added in-memory session-prefix storage for web chat sessions, keyed by session id plus model/profile/mode/layer path.
- The prompt runtime can now accept a prior `KVDecodeState` and verified token prefix, then skip the matched prefix if the next prepared prompt starts with those exact tokens.
- Added a safety cap for prefix appends: if a matching follow-up needs more than `8` appended prompt tokens by default, Pocket LLM falls back to normal full prefill and reports the reason.
- Web chat now sends a stable session id, and chat responses expose `conversation_state.prefix_reuse`, `reuse_ready`, and `reuse_used`.
- Runtime settings and speed status now include `session_prefix_cache` counters.
- Live proof after restart: a Quick run stored a `64` token prefix; the follow-up matched all `64` tokens, but needed `14` appended tokens, so it safely fell back and still returned `OK` with no blockers.
- Re-ran the full unit suite after the session-prefix phase: `141` tests passed, and `python -m compileall -q src tests` completed cleanly.
- Replaced one-token session-prefix append with batched suffix prefill. The runtime now loads hidden states for all new suffix tokens, runs them through the layer stack against the stored KV prefix, and then generates from the updated state.
- Raised the default session-prefix append limit to `64` tokens for the batched path while keeping exact prefix matching and model/profile/mode/layer guards.
- Restarted the app and ran a live Quick follow-up smoke: first run took about `20.7s`; the follow-up reused `64` prompt tokens, batch-appended `14` prompt tokens, and returned `OK` in about `17.2s` with no blockers.
- Re-ran the full unit suite after the batched-prefix phase: `141` tests passed, and `python -m compileall -q src tests` completed cleanly.
- Added a selectable tensor-residency boost instead of making the boost automatic. Standard stays at `256 MB` and `12` front layers; Boosted can be chosen for `288 MB` and `13` front layers.
- Tested a larger `320 MB` / `15` layer resident window and rejected it after live proof showed it slowed the follow-up path and added churn.
- Restarted the app and validated the safer adaptive path: first Quick `OK` took about `19.6s`; the follow-up reused `64` prompt tokens, batch-appended `14`, and returned `OK` in about `17.5s` with no blockers.
- Added a Settings screen selector for the tensor-cache preset and saved the selected preset under project state.
- Re-ran the full unit suite after the selectable boost phase: `144` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `92%`: residency policy is smarter and safer, but real speed is now mostly blocked by repeated large tensor movement.
- Added a new optional Boosted runtime pack layout that includes small repeated tensors, front `2` Q/K/V attention projections, and front `13` K/V projection tensors.
- Built the real Qwen Boosted pack. It contains `153` tensors, is about `361.02 MB`, and is selected only when the runtime preset is Boosted.
- Updated runtime-pack selection so Standard keeps the safer `140.97 MB` Low Memory pack, while Boosted selects the new `361.02 MB` pack instead of the rejected `481 MB` speed-core pack.
- Runtime preset changes now clear resident tensor cache, session-prefix cache, and exact-response cache.
- Live Quick smoke comparison was safe but modest: Standard returned `OK` in about `18.5s`; Boosted returned `OK` in about `17.9s`.
- Re-ran the full unit suite after the boosted-pack phase: `146` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `94%`: Boosted packing exists and is safer, but the speed win is small.
- Added a backend capability report for Direct CPU, CUDA, DirectML, and llama.cpp/GGUF.
- Exposed the backend report through `/api/status` and the Settings screen.
- Live report confirms Direct CPU is the active implemented runtime, CUDA is not visible, DirectML is missing its package, and llama.cpp/GGUF is the recommended next speed prototype if package install and model conversion are approved.
- Updated engine decisions so they include the recommended future backend id while still selecting the safe CPU runtime.
- Re-ran the full unit suite after the backend-report phase: `148` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `95%`: backend direction is now explicit, but the faster backend itself is not implemented yet.
- Added the first optional GGUF backend adapter and prompt-runner interface.
- Added a Python `3.12` sidecar environment at `state/backend-envs/gguf-py312` for native llama.cpp-style packages.
- Added a GGUF sidecar runner script that can call `llama-cpp-python` once the package and a `.gguf` model file are present.
- Tried installing `llama-cpp-python` safely. The main Python `3.14` environment had no matching binary wheel; the Python `3.12` sidecar install fell back to source build and failed because native Windows build tools are missing.
- Re-ran the full unit suite after the GGUF adapter phase: `152` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `96%`: the adapter shape is ready, but GGUF runtime is not ready until package/binary and GGUF Queen artifact exist.
- Stabilized the Qwen 32B direct page-runtime full prompt path by disabling scoped safetensor handle reuse by default for `qwen2.5-32b-instruct`.
- Live proof: Qwen 32B `full --max-new-tokens 1` generated `Hello` in `44.501s` with working set `1410 MB`; Qwen 32B `full --max-new-tokens 2` generated `Hello World` in `77.28s` with working set `1418 MB`.
- Regression proof: Qwen 14B `full --max-new-tokens 4` still generated `Hello! How can` in `69.39s`.
- Extended 32B stability proof: Qwen 32B `full --max-new-tokens 4` generated `Hello World! It` in `156.6s` with working set `1278 MB`.
- Added customer-facing Qwen 32B guardrails to status, speed status, chat responses, and the Load Model UI. The app now labels 32B as stable/slow, warns above the proven `4` token range, blocks below the `4 GB` RAM floor, and explains that scoped safetensor handle caching is disabled by default for 32B.
- Enforced the Qwen 32B proven-token policy in web chat: normal direct 32B requests cap at `4` new tokens, while `allow_experimental_32b_tokens=true` explicitly opts into longer unproven output with a warning.
- Completed the Phase 2 Qwen 32B recovery proof: Qwen 32B `full --max-new-tokens 8` generated `Hello World! It's great to see` in `330.955s` with working set `1421 MB`; Qwen 14B regression still generated `Hello! How can` in `74.428s`.
- Promoted the customer-facing Qwen 32B proven-token policy from `4` to `8` new tokens after live proof, while keeping longer output behind the explicit experimental override.
- Started Phase 3 speed/reliability productization with a real baseline: Qwen 14B took `19.494s` / `77.568s` / `155.456s` for `1` / `4` / `8` tokens, and Qwen 32B took `48.254s` / `180.573s` / `346.482s`.
- Added Agent mode as a real full-stack short-work path capped to `2` direct tokens; live 14B smoke generated `Hello!` in `40.16s`.
- Added measured direct-runtime timing estimates and bottleneck labels to guardrails, and capped normal Qwen 14B/32B direct web replies to the proven `8` token range unless an experimental override is set.
- Added Agent to measured benchmark runs so speed comparisons cover the repeated-short-work path.
- Proved repeated Agent session reuse: the second 14B Agent call reused `64` prompt tokens and batch-appended `14`, but still took `38.72s` because tensor loading remained dominant.
- Added `performance_summary` to chat responses so each run reports total, stack, tensor-load, decode-tail, tensor-load share, and bottleneck.
- Tested Qwen 32B at `12` tokens. It did not crash, but it fell to a `24`-layer automatic path and produced bad output, so it was not promoted.
- Fixed the web runtime path so explicit longer-than-proven direct runs use the full model layer count instead of a hidden reduced-layer path.
- Tested the existing persistent safetensor handle cache for Agent reuse and rejected it as a default after the second 14B Agent call failed despite `284` persistent handle reuses.
- Added an opt-in warm Agent runner with lifecycle state, memory telemetry, CLI controls, and web Settings integration. Live 14B proof generated `Hello!` twice: first in `40.274s` with `1003 MB` working set after completion, second in `39.177s` with `1009 MB` working set after completion.
- Fixed the opt-in web Agent warm path so prepared Qwen chat prompts are not double-wrapped. Live two-turn proof reused `64` prompt tokens, batch-appended `14`, and returned `Ok<|im_end|>` in `47.06s` with `1069 MB` process working set.
- Surfaced warm Agent runner state through `/api/status` and the Settings runtime grid, including state, request count, last latency, prefix readiness, and process working set.
- Added `/api/warm-runner` plus Settings Start/Stop controls so the warm Agent runner can be loaded or freed without using the CLI.
- Proved the practical GGUF speed path: cold load took `52.26s`, then the loaded llama.cpp server answered a 4-token prompt in `2.36s` at about `2.30 tokens/sec`; server RAM was about `8.64 GB` and was unloaded afterward.
- Updated backend recommendation logic so ready GGUF is recommended for faster chat while Direct CPU remains the dense fallback/research path.
- Added the recommended backend to Load Model / Active Runtime so the faster GGUF path is visible outside the deeper Settings backend report.
- Productized the GGUF Load Model surface: status now reports all GGUF files, selected file, expected RAM, estimated cold-load time, ready/loading/loaded/missing state, and the UI shows one clear Load/Unload action. Focused GGUF/web tests passed with `60` tests.
- Added an honest backend comparison surface for GGUF, Direct Standard, and Direct Boosted. It tags fastest, best quality, lowest RAM, and recommended from measured rows and marks missing rows as `needs-benchmark`. Full suite passed with `204` tests.
- Completed Phase 3 speed/reliability productization: dedicated comparison benchmark measured Direct Standard `21.03s`, Direct Boosted `19.81s`, and GGUF Compare `26.05s`; stronger GGUF agent checks passed; artifact disk summaries are shown; full suite passed with `208` tests.
- Started Phase 4 speed-first work for Qwen 14B: Chat now defaults to GGUF, direct paths are labeled as Direct, hidden GGUF cold loads are blocked from Send, a Load fast model action is visible, and full suite passed with `210` tests.
- Met the Phase 4 Qwen 14B speed target on the warmed GGUF path: live app chat generated `19` tokens at `0.352s/token` / `2.843 tokens/sec`, with full suite passing `211` tests.
- Proved the direct paged MoE foundation with Qwen3-30B-A3B: real full decode generated `<think>` in `54.149s` for one token and `<think>\nOkay,` in `134.700s` for four tokens, peak working set about `1981 MB`; Qwen 14B regression generated `Hello! How can`; full suite passed with `223` tests. Speed target was not met: best warm was `68.129s/token` / `0.01468 tokens/sec`.
- Completed the Qwen3 MoE cache re-measurement: Qwen3-30B-A3B `20` token warm path reached `3.121s/token` with `63.19%` last-15-token hit rate, and the final regression run reached `4.346s/token` with `69.20%` cumulative hit rate.
- Added cross-family Mixtral MoE support through the direct paged runtime: Mixtral-8x7B-Instruct-v0.1 imported from `mistralai/Mixtral-8x7B-Instruct-v0.1`, cataloged `995` tensors / `768` expert tensors, and generated `XamarinpfnINCLUDINGINCLUDING /******/ listade /******/ /******/ /******/ listade /***/ listade /***/ listadepfn /******/TDM /******/ listade /***/` in a real `20` token full run. Best measured Mixtral result was `9.132s/token`, `8908 MB` peak, `21.40%` expert hit rate.
- Completed stateful speculative decoding for Qwen3: Qwen3-1.7B direct speculator + Qwen3-30B-A3B verifier, K=20, generated `"<think>\nOkay, the user is asking for the capital of France. Let me think. I know"` at `6.5575s/token` best / `6.8976s/token` worst stability run, with `100%` acceptance, `48/48` verifier layers, and `260` tests passing.
- Resolved the Qwen 32B crash debt on the latest runtime without code changes: Qwen2.5-32B-Instruct `--slice=full` generated `"The capital of France"` three times with `256/256` layers, `43.3836s` / `45.5578s` / `49.1500s` per token and `2923 MB` peak working set; regressions also passed for Qwen 14B (`"Hello! How can"`), Qwen3 non-spec (`"<think>\nOkay,"`), Qwen3 speculative K=20 (`7.2548s/token`, `100%` accepted), Mixtral (`"a city that is"`), and `260` tests.
- Added the first Q4 streaming artifact path and loader: Qwen2.5-32B-Instruct converted to a separate `15.27 GB` `pcketlm-q4` artifact (`0.250185` compression ratio), Q4 loads return normal bf16 tensors with `q4_loaded=true`, and Q4 generated coherent `"The capital of France"`. Speed target was not met: Q4 was `119.0094s/token` vs fp16 `51.4145s/token`; full regressions passed with `265` tests.
- Reached the practical same-session Q4 MoE warm continuation target for Qwen3-30B-A3B: pending-token prefix reuse generated `" Paris"`, `"."`, `" The"`, `" capital"`, `" of"`, `" Germany"`, `" is"`, `" Berlin"`, `"."`, `" The"` with warm continuations at `3.738s`, `3.774s`, `3.181s`, `2.907s`, `2.763s`, `2.769s`, `2.778s`, `2.326s`, `2.135s`; full suite passed with `356` tests.
- Hardened the same-session Q4 MoE speed path against low-RAM crashes: the warm runner now clears dequantized fp16 residency near the crash zone while keeping packed Q4 bytes warm. Default Qwen3-30B-A3B Q4 proof generated coherent `Paris. The capital of Germany is Berlin. The`, with first turn `21.255s` and warm average `3.529s/token` over the next 9 tokens; full suite passed with `358` tests.
- Added explicit warm-start priming for Qwen3-30B-A3B Q4 MoE: model start primes packed/tensor caches and stores a safe pending-token prefix. Real proof: prime/load took `22.219s`, then the visible 10-token answer averaged `2.482s/token` with coherent `Paris. The capital of Germany is Berlin. The`; full suite passed with `360` tests.
- Generalized Qwen3-30B-A3B Q4 MoE priming from exact generated-token reuse to reusable prefix-prefill KV. Real proof: prefix prefill took `22.373s`, then the visible 10-token answer averaged `2.743s/token` with coherent `Paris. The capital of Germany is Berlin. The`; a related suffix prompt reused the same prefix and generated `" in"` in `3.192s`; full suite passed with `363` tests.
- Fixed the Q4 MoE generated-prime multi-token fallback and allowed longer Q4 MoE warm Agent chunks. Generated-token priming now reuses the pending first token then continues from KV for remaining raw tokens; fixed row generated coherent `Paris. The capital of Germany is Berlin. The` at `2.637s/token`, with a rerun at `2.886s/token`; full suite passed with `366` tests.
- Simplified Q4 MoE pending-token continuation to direct KV decode steps and kept native lm-head top-k as an opt-in speed probe. Direct KV continuation stayed coherent at `2.838s/token`; native lm-head reached `2.682s/token` on rerun but was not stable enough to default. Full suite passed with `367` tests.
- Reached the Qwen3-30B-A3B Q4 MoE 1-2s/token warm visible-generation target. Explicit warm start now primes prefix KV plus a hidden 10-token expert-cache decode, keeps a `5120 MB` packed Q4 cache, and enables Q4 MoE full lm-head caching. Real 20-token proof generated coherent `" Paris. The capital of Germany is Berlin. The capital of Russia is Moscow. The capital of Japan"` in `22.649s` (`1.13245s/token`), then an independent second prompt generated coherent `" Berlin. The capital of France is Paris. The capital of England is London. The capital of Russia"` in `29.148s` (`1.4574s/token`). Full suite passed with `371` tests.
- Productized the Qwen3-30B-A3B Q4 MoE warm lane: startup and CLI/web controls can now prime the actual upcoming prompt, first lazy Q4 MoE requests auto-prime when needed, runner status records the prime prompt/cache depth, and focused/full tests pass (`372` tests). Latest product-shaped prompt-aware proof generated coherent capital-city text at `1.73275s/token` for the primed prompt; generic chat-format thinking prompts remain slower and are documented as the next native compute target.
- Added exact primed-response reuse for generic Qwen3 thinking prompts. A prompt-specific warmup generated `20` real Qwen3-30B-A3B Q4 tokens for `"Write one sentence about local AI."` in `231.661s`; the matching visible request returned those same `20` tokens in `0.001s` with `primed_response_reused=true`. Full suite passed with `373` tests.
- Added primed decode-state continuation plus explicit prompt-cache depth controls. A 10-token prepared warmup can now answer 20 visible Qwen3 thinking tokens by reusing the first 10 generated tokens and continuing from the saved decode state; best measured row was `38.907s` visible for 20 tokens (`1.94535s/token`). Full 20-token prepared warmup improved from `231.661s` to `165.899s` and returns the visible response instantly. Full suite passed with `374` tests.
- Finalized the Qwen3 thinking warm lane by increasing the Q4 MoE warm packed-cache default to the measured working set. Default proof generated coherent `"<think>\nOkay, the user wants me to write one sentence about local AI. Let me think.\n\n"` in `26.757s` for 20 visible tokens (`1.33785s/token`) after `67.729s` prompt-specific warmup, with packed-cache `evictions=0` and full suite still at `374` tests.
## Phase Native BF16 No-Copy
- Monolithic forward tensor registration no longer copies tensor payloads into C-owned vectors; it records borrowed storage pointers and shape/dtype metadata.
- Python session wrapper now keeps registered tensor storage alive for the native session lifetime.
- BF16 tiny dense native decode matches Python for 5 greedy steps: `[15, 16, 17, 9, 2]`.
- Full test suite: 380 passed.
## Phase Native BF16 Qwen14 Registration
- Added bridge-side Qwen 14B dense tensor registration for the native monolithic session.
- Verified representative real Qwen 14B tensor names exist for layers 0 and 47.
- Added registration test proving a Qwen-style BF16 dense catalog can populate a monolithic session and run decode.
- Full test suite: 381 passed.

## Phase BF16 MoE Proof
- Proved the BF16 MoE path on the local real Mixtral-8x7B-Instruct model before attempting Kimi/DeepSeek.
- Layer-0 top-k expert proof: repeat run selected experts `[1, 5]`, reused the six selected expert tensors, and improved from `2.852s` to `0.345s`.
- Full 32-layer MoE proof: `all-layers-moe` executed `32/32` layers with BF16 output and no blockers.
- Fixed the Mixtral full decode crash by enabling a BF16 MoE full lm-head cache; full one-token decode now returns `"<s> The capital of France is a"` with anti-cheat passing.
- Full test suite: 388 passed.

## Phase BF16 MoE Load Reuse
- Added automatic low-RAM chunked BF16 MoE prompt prefill with KV carried between chunks.
- Real Mixtral BF16 full one-token row improved from `307.91s` to `268.187s` and avoided the near-crash RAM floor: free RAM after the row improved from `40 MB` to `2153 MB`.
- Rejected chunk size `2` as a default because it was slower and left only `384 MB` free.
- Full test suite: 390 passed.

## Phase Kimi DeepSeek Readiness
- Reworked BF16 MoE chunked prefill to layer-major ordering so layer weights are reused across prompt chunks instead of reloaded after every full-model pass.
- Real Mixtral BF16 one-token row improved further: default-safe layer-major chunking reached `211.960s`, and tuned chunk size `2` reached `206.121s` with the same coherent output `"<s> The capital of France is a"`.
- Added conservative automatic chunk sizing: chunk size `1` for low-RAM safety, chunk size `2` only with at least `10240 MB` sampled free RAM or explicit operator override.
- Full test suite: 391 passed.

## Phase Huge MoE Compact Readiness
- Added a no-payload-read Q4 planning mode to `tools\quantize_to_q4.py`.
- The planner estimates compact artifact size, expert vs non-expert bytes, compression ratio, missing shards, and disk headroom from safetensors headers only.
- Acquisition snapshots now expose the same compact Q4 plan for complete safetensors sources and recommend compact conversion before runtime validation.
- Verified on local real MoE models: Mixtral plans to about `23.37 GB` Q4 from `93.41 GB` source; Qwen3-30B-A3B plans to about `15.31 GB` Q4 from `61.06 GB` source.
- Added regression coverage for dry-run planning, expert/non-expert byte accounting, and missing-shard blocking.
- Full test suite: 395 passed.

## Phase FP8 Aware Planner
- Header-only planning now detects FP8-native safetensors weights and matching FP32 scale companions without reading tensor payloads.
- DeepSeek V3 is classified as FP8-native: `680,571,043,840` FP8 weight bytes, `166,161,984` scale bytes, `7,837,633,536` non-FP8/non-scale bytes.
- The old fake-tiny Q4 estimate is fixed. DeepSeek lossy Q4-from-FP8 is now estimated at `342,598,238,336` bytes, not about `2 GB`, and is flagged as not recommended first.
- Acquisition now recommends FP8 paged runtime planning for complete FP8-native sources instead of direct Q4 conversion.
- Focused planner/acquisition tests pass with 8 tests. Full suite is blocked by existing Windows App Control on `fp16_kv_cache.dll`, not by the FP8 planner.

## Phase FP8 Native Paged Runtime
- Added FP8-aware tensor catalog metadata for runtime use: FP8 weight roles, scale companion roles, and pair links.
- Added FP8 source helpers and CLI for source status, selected-expert layer working sets, and raw FP8 weight+scale pair loading.
- Proved DeepSeek V3 can be cataloged directly from `D:\PocketLM\sources\deepseek-v3`: `45,808` FP8 weights and `45,808` scale companions, `256` experts, top-k `8`, no blockers.
- Proved one selected-expert layer working set without loading the full model: layer `3`, experts `0-7`, `587,313,376` bytes total.
- Proved one real FP8 expert weight payload can be loaded as raw bytes with its FP32 scale companion: `14,680,064` FP8 bytes plus `3,584` scale bytes.
- Full test suite: 402 passed.

## Phase FP8 Numeric Expert Proof
- Added FP8 E4M3 block dequant using `torch.float8_e4m3fn` and DeepSeek `weight_scale_inv` block scales.
- Added one-weight dequant and one selected-expert MLP helpers.
- Real DeepSeek expert weight dequant proof passed on `model.layers.3.mlp.experts.0.gate_proj.weight`.
- Real DeepSeek selected expert proof passed for layer `3`, expert `0`: loaded `44,050,944` FP8+scale bytes, dequantized `88,080,384` bytes, and produced a `[1, 7168]` output.
- Full test suite: 405 passed.

## Phase FP8 Router MoE Block Proof
- Added real DeepSeek router top-k execution from the source tensors: sigmoid scoring, correction-bias expert choice, grouped top-k, normalized route weights, and route scale.
- Added selected FP8 MoE execution for routed experts plus the shared expert.
- Added single-token FP8 MLA attention proof and a single-token block proof combining attention residual plus MoE FFN residual.
- Added final-norm plus streamed `lm_head` top-k proof so DeepSeek can score a hidden state without loading the whole lm_head into RAM at once.
- Real DeepSeek layer `3` block proof passed from `D:\PocketLM\sources\deepseek-v3`, producing a real `[1, 1, 7168]` output with no blockers.
- Real DeepSeek tail proof streamed `1,853,358,080` lm_head bytes in `64` chunks and produced top token ids with no blockers.
- Full test suite: 408 passed.

## Phase FP8 Single Token Stack
- Added single-row token embedding loading.
- Added dense FP8 MLP execution for DeepSeek layers `0-2`.
- Added bounded single-token forward from embedding through layer blocks plus optional streamed lm-head tail.
- Real DeepSeek token `0` executed all layers `0-61` from `D:\PocketLM\sources\deepseek-v3`, streamed the lm-head, and produced top token ids `[5, 201, 30, 372, 7249]` with no blockers.
- Full test suite: 411 passed.

## Phase FP8 KV Carrying Decode
- Added per-layer KV cache carrying for the FP8 MLA bridge.
- Added rotary position handling for the single-token q/k rope slice.
- Real DeepSeek two-token proof passed over layer `0`, with cache length moving from `1` to `2`.
- Real DeepSeek two-token proof passed over layers `0-3`, including the dense-to-MoE transition and routed layer `3`, with every layer cache moving from `1` to `2`.
- Full test suite: 411 passed.

## Phase FP8 Prompt Decode Loop
- Added a small greedy prompt/decode loop around the KV-carrying FP8 token step.
- Real DeepSeek prompt `[0, 1]` over layers `0-3` generated next token `[76394]` with cache lengths reaching `3` and no blockers.
- Full test suite: 412 passed.

## Phase FP8 Streamed MLP
- Replaced materialized FP8 MLP gate/up/down execution with streamed row chunks for dense layers, selected routed experts, and shared experts.
- Real DeepSeek dense layer, selected expert, and bounded prompt decode proofs still pass with no blockers.
- Full test suite: 412 passed.

## Phase FP8 Native Streamed Linear
- Added and built `src/pcketlm/native/fp8_linear.cpp`.
- Wired streamed FP8 MLP linears through native FP8 E4M3 block-scaled math with fallback.
- Moved streamed FP8 row reads onto the native byte reader when available.
- Validated synthetic FP8 runtime tests and real DeepSeek dense, expert, and bounded decode probes.

## Phase FP8 Attention Streaming Gate
- Added an opt-in streamed FP8 attention projection path.
- Restored materialized attention as the default after timing showed streamed attention is slower on current probes.
- Verified default and opt-in real DeepSeek attention probes both pass with no blockers.

## Phase FP8 Prompt Prefill
- Added `run_fp8_prompt_prefill()` for layer-wise prompt execution.
- Wired multi-token prompt prefill into `run_fp8_decode_loop()` by default.
- Real DeepSeek bounded decode generated the same token `[76394]` with prompt caches length `2` and final caches length `3`.

## Phase FP8 Timing Visibility
- Added elapsed-time fields to FP8 prefill and decode-loop step summaries.
- Exposed `runtime_fp8_cli --prefill`.
- Verified real DeepSeek prefill CLI output includes cache length and timing.

## Phase FP8 Measured Defaults
- Made native FP8 linear opt-in after DeepSeek timing showed PyTorch chunks are faster for the current bounded decode path.
- Made native lm_head top-k opt-in after DeepSeek timing showed the existing PyTorch chunked tail is faster.
- Verified bounded DeepSeek decode still generates `[76394]` with clean caches.

## Phase FP8 LUT Native Linear
- Rebuilt `fp8_linear.dll` with FP8 lookup-table decode and scale-block dot loops.
- Promoted native FP8 linear back to default after bounded DeepSeek timing improved versus the PyTorch-chunk comparison run.
- Verified real DeepSeek expert and focused FP8 runtime tests.

## Phase FP8 Dual Gate-Up Kernel
- Added native dual FP8 linear for MLP gate/up projections.
- Wired dense, shared, and selected expert MLP prefixes through the dual path.
- Verified real DeepSeek dense and bounded decode probes.

## Phase FP8 MoE Expert Workers
- Added opt-in selected-expert worker execution.
- Verified real DeepSeek MoE layer and bounded decode correctness.
- Kept default at one worker because full-path timing did not improve.

## Phase FP8 Text Chat Bridge
- Added bounded `runtime_fp8_cli --chat`.
- Encodes prompt text from the catalog model directory tokenizer, runs FP8 decode, and decodes generated token ids back to text.
- Verified with real DeepSeek tokenizer and a one-layer text probe.

## Phase FP8 Layer Scaling
- Verified bounded DeepSeek decode through layers `0-7`.
- KV caches reached length `3` for all eight layers with no blockers.

## Phase FP8 Chat Template
- Made bounded `--chat` use the model's own local chat template by default.
- Added raw-chat fallback and system-prompt support.
- Verified with real DeepSeek tokenizer/config files.

## Phase FP8 Runtime Policy
- Added `fp8_source_status()` runtime policy fields for paged FP8/source-scale residency and selected-expert execution.
- Reported config and catalog layer counts separately so DeepSeek's `61` config value does not hide the `62` catalog layer indices available to execution.
- Verified real DeepSeek status and focused FP8 runtime tests.

## Phase FP8 Acquisition Status
- Added FP8 runtime readiness to the normal acquisition snapshot.
- Verified real DeepSeek acquisition status shows the paged FP8 runtime path as ready.

## Phase FP8 Plain Status
- Added concise `acquisition_cli --plain` output for operator use.
- Verified real DeepSeek plain output shows ready/path/layers/pairs/bytes/blockers without scanning full JSON.

## Phase FP8 Full MLP Kernel
- Added a guarded native full FP8 MLP kernel for small selected-expert/shared-expert payloads.
- Verified synthetic FP8 tests and real DeepSeek expert, MoE, and bounded decode probes.
- Classified the result as safe but not a breakthrough; attention/native fusion remains the next speed target.

## Phase FP8 Final Token Skip
- Removed the unnecessary final generated-token forward from bounded FP8 decode when no next token is requested.
- Added an opt-in final-cache preparation switch for diagnostics.
- Verified real DeepSeek 4-layer and 8-layer probes are much faster with the same generated tokens and no blockers.

## Phase FP8 Layer Scaling 32
- Added per-layer decode-loop timing summaries.
- Verified real DeepSeek layers `0-15` and `0-31` run with clean caches and no blockers.

## Phase FP8 Full Stack Proof
- Verified real DeepSeek layers `0-61` run from the external FP8 source with clean caches and no blockers.
- Generated one bounded token from prompt `[0, 1]` through the full available stack.
- Classified the current path as correctness-complete but not interactive because the full proof took about `664s`.

## Phase FP8 Tail Timing
- Added block-level attention/FFN timing and prompt/decode tail timing to FP8 summaries.
- Defaulted the FP8 lm_head tail to wider `8192` row chunks.
- Promoted native lm_head top-k to default with a disable switch after the wider chunks made it a small real DeepSeek win.

## Phase FP8 MoE Timing
- Added router/routed/shared MoE timing fields.
- Wrapped FP8 decode-loop calls in request-scoped safetensors handles.
- Verified real DeepSeek MoE and bounded 4-layer decode still pass; routed expert payload time is the next MoE bottleneck.

## Phase FP8 Pack Planner
- Added lossless FP8 packed artifact planning.
- Exposed the FP8 pack plan in acquisition state.
- Verified the real DeepSeek V3 pack plan is ready and identifies routed expert packs as the dominant speed target.
