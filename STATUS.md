# Status

Project: `pcketlm`

Current root:
- `C:\Users\isale\Documents\pcketlm`

Stage:
- foundation implementation underway

Current goal:
- turn the first real text-prompt entry path into an interactive desktop test flow while improving stop/session control and keeping staged streaming as the memory-control foundation

Product direction:
- `pcketlm` is a local-first LLM control center for weak hardware, direct model loading, personalization, agent-oriented profiles, comparison, and model management.
- The product should not depend on Ollama, LM Studio, or another local LLM app as the runtime layer.
- The user experience should serve both beginners and power users: plain-English default flows with advanced/runtime detail available when needed.
- The first model-family priority is Qwen, followed by Kimi, Kronos/Kronk-family targets if a real supported open model path exists, Gemma, and then broader dense model families later.

Current summary:
- `pcketlm` now has a defined V1 direction.
- `pcketlm` now has a first product-level model-family metadata layer so Qwen is the active implementation behind a more universal model-management surface.
- The desktop status surface now shows model-family support status and a clickable first `Model Home` for chat, personalization, comparison, inspection, and benchmark readiness.
- The first safe personalization templates now exist for Balanced Local, Agent Coder, and Low Memory targets, but they do not yet create optimized artifacts.
- The first benchmark readiness backend now exists, but it does not yet run or persist full benchmark results.
- Desktop chat now runs prompt generation in the background so the app shows a working state instead of freezing during slow local generation.
- Desktop chat now defaults to `2` max new tokens and shows an alpha-runtime speed hint because full-stack Qwen generation is currently slow.
- Desktop chat now has Fast, Balanced, and Quality runtime modes. Balanced uses `32` layers by default for faster testing; Quality uses the full current stack for better output.
- V1 is a Windows-first desktop product for power users.
- V1 starts with dense Qwen-family text models.
- The system preserves the original model and creates reversible optimized runtime artifacts.
- The project now includes working import, validation, registry, catalog, and acquisition foundations.
- The project now also includes a runtime source-readiness layer for the first loader pass.
- The project now includes a runtime bootstrap/preflight layer with dependency checks and session creation.
- The project now includes an explicit risk register to keep likely failure paths visible before they hit implementation.
- The project now includes the first desktop status-screen spec grounded in live acquisition and runtime state.
- The desktop test UI now reads model choices from the registry first and falls back to the detected local source when the registry is empty.
- The first real load command now exists and has been run against the full official Qwen source.
- The runtime and desktop layers now expose blocker category, severity, and recommended action as reusable product state.
- The reduced-memory strategy planner now recommends staged disk streaming as the best next runtime path on this machine.
- The first staged disk-streaming planner now produces a concrete working-window and cache-layout plan from real hardware state.
- The first staged-streaming manifest and cache bootstrap now works and writes real state under `state/streaming`.
- The runtime can now read the streaming manifest back and validate the cache layout as runtime-usable state.
- The runtime can now map the actual safetensor shard files into persisted streamable weight units.
- The staged-streaming layer now uses a shared segment budget so streamed units fit both the hot window and the warm prefetch window.
- Repo-root `pytest` and `py -m pcketlm...` commands now work without manual `PYTHONPATH` setup.
- The first live window schedule now places one unit in the hot window and one unit in the warm prefetch window with no blockers.
- The active project now lives under `Documents` instead of `.openclaw`, and the desktop shortcut points to the new root.
- The registry now auto-relocates stale in-project model paths after project moves, so the desktop app no longer reads deleted source folders after a root change.
- The streaming manifest now also auto-relocates stale in-project paths after project moves, so staged streaming survives a project root change.
- The staged-streaming runtime can now materialize the scheduled hot and warm segments into real cache files on disk.
- The staged-streaming runtime now records lightweight write-time checksums and can verify the live cache index and cache files on demand.
- The staged-streaming runtime can now rotate the warm prefetched segment into hot, refill warm with the next overflow segment, and re-materialize the cache.
- The staged-streaming runtime now persists residency and rotation position so it can remember current hot units, warm units, consumed units, overflow head, and refill count across operations.
- The staged-streaming runtime now also tracks cache hits and cache misses during cache verification without resetting rotation history or refill history.
- The staged-streaming runtime now has a control layer that chooses the next action from live telemetry and can execute that action directly.
- The desktop app now shows streaming control state, residency telemetry, and a live `Advance Stream` action.
- The desktop app now uses a scrollable content area with a fixed bottom action bar so new runtime sections do not push the controls out of view.
- The runtime and desktop app now also support a verify-first `Safe Advance` path that repairs cache when needed and only rotates when verification passes.
- The runtime now has a tensor-aware catalog layer that reads safetensors headers and maps real tensor names, dtypes, shapes, offsets, layer IDs, and shard placement.
- The runtime now has a tensor-aware execution-plan layer that groups real tensors into execution units like embeddings, per-layer norm/attention/MLP, and decode head.
- The runtime now has a first tensor loader slice that can load one real tensor or one small execution unit from the original shards on demand.
- The runtime now has a stronger tensor verification pass that checks loaded tensors and grouped execution units against catalog and execution-plan metadata.
- The runtime now has a first minimal CPU-only layer-forward bridge that performs real layer-0 math from on-demand loaded Qwen tensors instead of only loading or verifying them.
- The runtime now uses an mtime-aware tensor-name index for hot tensor lookup paths, avoiding repeated catalog scans during full-stack generation.
- Web chat now uses a compact runtime identity prompt, reducing the latest Qwen logic smoke from `135` to `76` prompt tokens while preserving model/profile context.
- Latest real Low Memory/Quality logic smoke: `If 2 plus 3 equals 5, answer YES only.` returned `YES`, ready with no blockers, in about `22.8s` cold-ish and `21.3s` warm.
- Web chat now blocks real model generation below `4 GB` free RAM after live checks showed the runtime can start above `3 GB` and still fall into severe memory pressure during tensor loading.
- Runtime settings now expose cumulative tensor-load diagnostics, including tensor requests, tensor bytes loaded, and shard opens.
- The first planning-only optimized artifact manifest exists for Qwen Low Memory at `models/qwen2.5-14b-instruct/artifacts/runtime-pack-plan.low-memory.artifact.json`; it maps `579` source tensors, `8` shards, and `48` layers.
- Chat responses now include lightweight conversation-state metadata so the app can later evolve from bounded prompt replay toward true KV/session reuse.
- Current Qwen heavy-engineering foundation estimate: about `73%`. Artifact groundwork and diagnostics are better; the next true speed leap needs a derived weight-pack/runtime artifact, not more safetensors handle reuse.
- The first real Qwen Low Memory runtime pack now exists at `models/qwen2.5-14b-instruct/artifacts/runtime-pack-plan.low-memory.small.safetensors`.
- The artifact-aware loader now prefers packed tensors when present and falls back to original safetensors for everything else.
- Latest artifact-backed Low Memory/Quality logic smoke returned `YES`, ready with no blockers, using `97` packed tensors and reducing shard opens from the earlier pressure-run pattern of `198` to `148`; cold-ish wall time was about `22.3s`, warm repeat about `20.9s`.
- Current Qwen heavy-engineering foundation estimate: about `78%`. The first real artifact path works; the next leap must pack/re-layout the large projection tensors instead of only the small repeated tensors.
- Tier 2 artifact packing now includes front-layer Q/K/V projection tensors for the first `2` layers, producing a `109` tensor pack of about `140.97 MB`.
- The latest app restart saw the Tier 2 artifact as ready, but free RAM was only about `2.31 GB`; the same test prompt correctly returned a `memory-guard` response instead of starting generation.
- Current Qwen heavy-engineering foundation estimate: about `81%`. Tier 2 pack creation and loader support are complete; a fresh speed comparison needs the machine above the `4 GB` generation guard.
- GGUF/llama.cpp backend support is now real for Queen/Qwen: Pocket downloaded the official standalone llama.cpp Windows CPU runtime, downloaded the official Qwen2.5-14B-Instruct Q4_K_M GGUF artifact, merged the split GGUF files, and added a `GGUF` mode in the web app.
- The app can now use a persistent local `llama-server` path instead of reloading the GGUF model for every prompt. After the model is loaded once, the latest direct llama-server check answered a one-token prompt in about `0.52s`, the Python adapter answered in about `0.39s`, and the live web API answered `READY` through `llama-cpp-gguf-server` with `0.91s` reported generation time.
- The persistent GGUF server currently uses about `8.6 GB` to `10 GB` RAM on this machine. It is a power-user backend path, not the weak-hardware default.
- Web chat GGUF mode now bypasses the older direct-runtime free-RAM guard, because that guard was designed for safetensors-based direct generation and incorrectly blocked GGUF after the llama server had already loaded.
- Pocket now has GGUF server lifecycle controls: `/api/gguf/server` can load or unload the persistent llama.cpp server, and the Load Model screen shows GGUF server state, RAM use, local artifact path, and Load/Unload controls.
- Latest lifecycle proof: Pocket unloaded the GGUF server successfully, then loaded it again through the new API in about `45.75s`; the loaded server reported ready with about `9.81 GB` resident RAM.
- Latest fast GGUF web chat proof after metadata cleanup: `Reply with SUN only.` returned `SUN` through `llama-cpp-gguf-server` with about `1.07s` wall time and `0.61s` app-reported generation time.
- GGUF prompts now use the proper Qwen instruct chat format. This fixes the UI case where `Reply with OK only.` produced unrelated continuation text instead of following the instruction.
- Latest GGUF prompt-format proof: `Reply with OK only.` returned `OK`; a second formatted GGUF check returned `YES` in about `1.29s` wall time.
- Benchmarks now include a fast GGUF-focused path. The web app has a `Run GGUF` benchmark button that saves instruction, short-answer, logic, and agent-style GGUF checks without triggering the older slow Direct CPU benchmark.
- GGUF chat and benchmarks now send Qwen stop markers to llama.cpp and allow GGUF mode up to `64` max tokens, while the older direct CPU modes keep the safer `16` token cap.
- Latest GGUF longer-answer smoke: an agent-style two-step local-model check returned two complete numbered steps through `llama-cpp-gguf-server` in about `21.57s` with `48` max tokens.
- Latest saved GGUF benchmark: `GGUF 1` `0.55s` -> `OK`; `GGUF 4` `1.56s` -> `Local AI enhances efficiency`; `GGUF 8` `4.0s` -> `Pocket LLM provides concise answers and assistance`; `GGUF 32` `10.37s` -> complete one-sentence Pocket LLM explanation; `GGUF Logic` `0.93s` -> `YES`; `GGUF Agent` `12.63s` -> two complete numbered checking steps.
- Current Qwen heavy-runtime foundation estimate: about `95%`. The major backend paths, server controls, GGUF benchmark loop, stop handling, and longer-answer tuning now exist; remaining heavy work is safe default selection, cleaner GGUF model/artifact management, and broader agent benchmarks.

What is true right now:
- the project tracking area exists
- the V1 blueprint exists
- the first engineering roadmap exists
- the first Python package skeleton exists
- model import and validation code exists
- source inspection and download-state logic exist
- registry persistence and catalog logic exist
- a product-facing acquisition/progress command exists
- a runtime source-readiness command exists
- a runtime bootstrap command exists
- a guarded runtime load-attempt command exists
- a reduced-memory strategy planner exists
- a staged disk-streaming planner exists
- a staged-streaming manifest/bootstrap layer exists
- a staged-streaming manifest reader exists
- a staged-streaming weight-unit mapper exists
- the required Python runtime libraries for the first correctness slice are installed
- the official Qwen source folder is fully downloaded, validated, and registered
- the first real CPU load is currently blocked by available system RAM, not by missing files or dependencies
- the recommended next runtime path is staged disk streaming, not plain CPU loading
- the current machine plan suggests about a 0.86 GB hot window, 0.43 GB prefetch window, and about 33 streamed chunks
- the streaming state now persists under `state/streaming/qwen2.5-14b-instruct`
- the current manifest reader sees the streaming state as ready with no cache-layout blockers
- the current unit map sees all 8 shard files as streamable units, now split into segment-sized units that fit both streaming windows
- the current live schedule uses about 388.5 MB in the hot window and about 388.5 MB in the warm window
- the current live cache now contains one materialized hot segment and one materialized warm segment under `state/streaming/qwen2.5-14b-instruct`
- the current live cache verification passes with no blockers after rotation
- the current live runtime control step is `rotate-forward`
- the current live residency state shows rotation step `9`, hot `unit-0001-seg-010`, warm `unit-0002-seg-001`, consumed `unit-0001-seg-009`, overflow head `unit-0002-seg-002`, refill count `9`, cache hits `4`, and cache misses `0`
- the current live tensor catalog contains `579` tensors across `8` shards and `48` transformer layers, with persisted tensor metadata at `state/streaming/qwen2.5-14b-instruct/tensor-catalog.json`
- the current live tensor execution plan contains `147` execution units with ordered phases of `prefill`, `layer-entry`, `layer-attention`, `layer-mlp`, and `decode-head`
- the current live tensor loader can successfully load the `layer-00-layer_norm` execution unit from the original Qwen shards, yielding two real `torch.bfloat16` tensors totaling `20480` bytes
- the current live verification pass successfully validates the full `layer-00-attention` execution unit, and the loader can load that larger unit as `7` real `torch.bfloat16` tensors totaling `125843456` bytes
- the current live layer bridge can execute a real synthetic single-token layer-0 slice for `qwen2.5-14b-instruct`, producing a `1 x 1 x 5120` `torch.float32` output tensor with no blockers
- the current live layer bridge now loads the required norms, attention projections, and MLP projections one tensor at a time so the first execution slice fits a weak-memory machine better than a full-unit load
- the current live layer bridge can now chain a real synthetic single-token pass through layers `0` and `1` for `qwen2.5-14b-instruct`, still producing a `1 x 1 x 5120` `torch.float32` output tensor with no blockers
- the current live layer bridge can now chain a real synthetic single-token pass through layers `0` to `3` for `qwen2.5-14b-instruct`, still producing a `1 x 1 x 5120` `torch.float32` output tensor with no blockers
- the runtime now has a token-entry bridge that can read the needed embedding row for a real token id without loading the full embedding table into RAM and then hand that hidden state into the layer stack
- the current live token-entry bridge can now start from token id `42` and execute layers `0` and `1` for `qwen2.5-14b-instruct` with no blockers
- the runtime now has a low-memory decode tail that applies final norm and streams `lm_head` row chunks to build real logits without loading the whole output matrix into RAM
- the current live token decode path can now start from token id `42`, execute layers `0` and `1`, and produce a real `1 x 1 x 152064` logits tensor with no blockers
- the runtime now has a first repeated greedy decode loop built on top of token entry, the layer stack, and the decode tail
- the current live repeated loop can now start from token id `42` and produce a two-step greedy chain with explicit history carry; with `history_window=3` it currently yields `42 -> 123571 -> 117864`
- the current repeated loop now carries a small recent-token history summary, but it still does not yet carry true causal attention or KV-cache state between steps
- the decode path now supports multiple token-selection policies, including greedy and deterministic top-k sampling
- the current live sampled loop can now start from token id `42` with `history_window=3` and produce the sampled two-step chain `42 -> 123571 -> 123571`
- the runtime now has a small comparative decode benchmark so the current history-summary and K/V-aware paths can be checked side-by-side on the same seed
- the current live decode benchmark for token id `42` compares `history-greedy`, `history-top-k-sample`, and `kv-greedy`, and now also reports reusable summary fields like `steps_completed`, `final_token_id`, and `unique_token_count`
- the current live decode benchmark now also reports stop reasons, so the current Qwen runs explicitly end on `step-limit` instead of being treated like open-ended demo chains
- the current live decode benchmark for token id `42` yields:
  - `history-greedy`: `42 -> 123571 -> 117864`
  - `history-top-k-sample`: `42 -> 123571 -> 123571`
  - `kv-greedy`: `42 -> 123571 -> 46322`
- the runtime now has a first real K/V-carrying loop that keeps projected K/V tensors across steps per layer and applies RoPE on the live key path
- the current live K/V-aware loop can now start from token id `42` and produce the two-step greedy chain `42 -> 123571 -> 46322`
- the current K/V-aware path now also carries an explicit decode-state object with next token, next position, generated token chain, and per-layer cache sequence lengths
- the current K/V-aware path now also carries stop-aware decode-session state derived from the model config and `generation_config.json`, including EOS token ids, finished state, and stop reason
- the current K/V-aware loop is now RoPE-aware on the live key path, but it is still not yet a full production decode implementation
- the runtime now has a first real prompt-entry path that loads the local tokenizer from the original model files, tokenizes text prompts locally, prefills the K/V session with prompt tokens, and then generates from that prompt state
- the prompt-entry path now also wraps prompts in a simple Qwen-style instruct/session format when tokenizer metadata supports it, uses generation defaults from `generation_config.json`, and advances prompt prefill with forced tokens instead of sampling during prefill
- the prompt-entry path now also exposes basic prompt/session controls: custom system prompt, raw-prompt mode, and repetition-penalty-aware token selection on the generation side
- the desktop test UI now also exposes the live prompt path with prompt input, system prompt override, raw-prompt mode, max new tokens, custom stop-token ids, generated output, and session-detail reporting
- the prompt runtime now also uses a more conservative default generation profile by staying greedy unless sampling is explicitly requested and by choosing an automatic model-aware layer budget when prompt runs do not force a layer count
- the automatic prompt layer budget is now deeper for short sessions, so the live real-Qwen prompt path can currently carry 8 layers by default instead of 4 for short prompt tests
- the layer bridge now supports multi-token causal execution, and the prompt runtime now uses that real multi-token prefill path instead of walking prompt tokens one-by-one
- the automatic short-prompt fidelity budget now uses the full carried stack by default for short prompt tests on this machine, so the live real-Qwen prompt path currently carries 48 layers on the short `hello world` check
- the runtime now also supports stronger generation-session controls including `top_p`, `min_new_tokens`, and stop strings, and the desktop app now has a chat-style local test surface with clear-history support
- the current live prompt-entry path can now take text like `hello world` and `write a short poem`, build real prompt token ids, and generate from the prompt instead of starting from raw token ids
- the current live prompt-entry results are still rough and not chat-quality yet:
  - wrapped `hello world` -> generated tokens `[144763, 94489]` -> text `Ởgaard`
  - wrapped `write a short poem` -> generated tokens `[143941, 109231]` -> text `อำนวยความ坐标`
- the current live prompt controls also work:
  - wrapped `hello world` with system prompt `Be brief.` and repetition penalty `1.2` -> ` alguataka`
  - raw `hello world` with repetition penalty `1.2` -> `eligeuish`
- the current live prompt path now also supports explicit max-new-token limits and custom stop-token lists, and a direct runtime check with `hello world` plus `max_new_tokens=4` produced `setTypeものicielty` with `stop_reason: step-limit`
- the current live prompt path now also defaults to `greedy-prompt-kv-cache-rope` for plain prompt tests and, with no explicit prompt layer count forced, currently uses 8 carried layers on the real Qwen model for short prompt sessions; a live `hello world` run with `max_new_tokens=4` produced `findFirstOrCreateyü背上` with cache lengths for layers `0` through `7`
- the current live prompt path now also uses a real multi-token causal prefill slice before generation; on the real Qwen model, the current default `hello world` + `max_new_tokens=4` test now runs with 48 carried layers and produced `Hello! How can`
- the current live prompt path now also supports stronger continuation/session controls; on the real Qwen model, `hello world` with `max_new_tokens=6`, `min_new_tokens=2`, and the current full-stack default produced `Hello! How can I assist`
- the current codebase now has product-level model-family metadata for Qwen, Kimi, Kronos/Kronk, and Gemma, with Qwen marked as the active runtime family and the others marked as planned
- the current desktop status model now exposes a flexible model-home action list so the app direction is not locked to a single forced workflow
- the desktop Model Home now has clickable actions for Chat, Personalize, Compare, Inspect, and Benchmark
- the Benchmark action now uses a lightweight backend readiness summary instead of only saying the feature is planned
- the Personalize action now shows safe built-in profile templates for Balanced Local, Agent Coder, and Low Memory
- the current full test suite passes with `82` tests after the first clickable model-home/profile/benchmark-readiness slice
- the latest real Qwen chat smoke check generated `Hello!` from `hello world` with `max_new_tokens=2`, `ready: true`, and no blockers
- the latest measured smoke-check runtime is about `67` seconds for `2` generated tokens on the current full-stack Qwen path
- the latest measured Balanced-mode smoke-check runtime is about `47` seconds for `2` generated tokens at `32` layers, but quality can drift compared with full-stack Quality mode
- profiling shows the next major speed bottleneck is inside tensor conversion and CPU linear math, especially repeated `.float()` conversion of loaded bf16 tensors and `torch._C._nn.linear`
- after inference-mode and metadata caching, the latest Balanced-mode smoke-check runtime is about `44` to `45` seconds for `2` generated tokens at `32` layers
- the desktop Benchmark action now persists a lightweight benchmark result under the selected model's `benchmarks` directory and updates a `latest.lightweight-benchmark.json` snapshot
- the runtime now has a memory-capped tensor residency layer for converted CPU tensors, with an environment-tunable total cache cap, per-tensor cap, LRU eviction, and hit/miss/store/eviction counters
- the layer bridge now uses this residency layer for required per-layer tensors instead of directly converting every loaded tensor with `.float()` in the bridge itself
- the default tensor residency policy is intentionally conservative: it keeps a front-layer warm window under a `256 MB` total cap and `32 MB` per-tensor cap, so weak machines do not accidentally become full-weight cache machines
- lightweight benchmark snapshots now include runtime settings and tensor residency stats, and the desktop Benchmark action shows math dtype, lm_head chunk rows, cache hits, misses, resident tensors, resident MB, evictions, and skips
- the initial broad cache policy had `0` hits and `406` evictions on the real Balanced Qwen smoke check, so it was replaced with front-layer admission
- the current front-layer policy produced `43` hits, `0` evictions, and about `252 MB` resident tensors on the measured in-process Balanced Qwen check
- the runtime now defaults to `bfloat16` math for CPU layer/decode linears, with `PCKETLM_RUNTIME_MATH_DTYPE=float32` available as a compatibility escape hatch
- the default lm_head decode-tail chunk size is now `8192` rows after live checks showed `8192` was better than `4096` and safer/faster than `16384` on this machine
- the tensor residency loader now clones same-dtype safetensors tensors before keeping or returning them, so bfloat16 mode owns safe CPU memory after shard handles close
- the latest unit suite passes with `91` tests after adding bfloat16 math and same-dtype residency coverage
- the latest real Balanced-mode Qwen smoke check reports `ready: true` with no blockers for `hello world`, `max_new_tokens=2`, and `32` layers; it completed in about `38.5` seconds from the CLI and generated `こんにちは世界的`
- desktop chat now defaults to Quality full-stack mode because it gives the best current answer behavior; Balanced and Fast remain available as speed/testing modes
- the latest real default Quality-mode Qwen smoke check reports `ready: true` with no blockers for `hello world`, `max_new_tokens=2`, and the full stack; it completed in about `60` seconds from the CLI and generated `Hello!`
- the desktop Benchmark action now runs a real measured benchmark in the background across Fast, Balanced, and Quality modes and persists a `latest.measured-benchmark.json` snapshot
- the latest real measured Qwen benchmark is ready with no benchmark blockers:
  - Fast: about `12.3` seconds, output `findFirstOrCreate`
  - Balanced: about `39.7` seconds, output `こんにちは世界的`
  - Quality: about `57.1` seconds, output `Hello!`
- measured benchmarks keep plain CPU RAM limitations as warnings when the direct runtime path still completes successfully
- the launcher now opens a new local web UI inspired by the downloaded Pocket LLM HTML template instead of the old Tkinter desktop screen
- the new web UI has sidebar navigation for Chat, Load Model, Personalize, Benchmarks, Agents, and Settings, with Chat selected by default
- the new web UI calls real local Python API routes for status, chat, and measured benchmarks
- the latest web-server smoke check loaded the new HTML, resolved the active Qwen model, and found the three built-in profile templates
- the latest real Qwen web chat API smoke check generated `Hello!` from `hello world` in Quality mode with no blockers
- web chat now sends a bounded recent conversation history to the local runtime instead of treating every message as isolated
- web chat now shows a live running timer during slow local generation and defaults to `4` max new tokens for more useful short replies
- the latest real Qwen web chat API smoke check with conversation history returned `ready: true`, no blockers, and confirmed `2` previous turns were included; the tiny 2-token output was still mixed-language, so generation quality remains the next runtime concern
- the web chat quality path now formats recent turns as real Qwen chat messages and keeps short Quality conversations on the full 48-layer stack
- the latest real Qwen memory follow-up smoke check answered `Your name is Sam` from previous turns, with `prompt_token_count: 61`, all `48` cache layers active, no blockers, and about `99` seconds for `4` generated tokens
- web chat now runs through background chat jobs with start/status/cancel endpoints instead of holding the browser inside one blocking request
- the web UI now has Cancel and Retry controls, keeps showing elapsed status through polling, and preserves the proven Quality request path
- the latest real Qwen background-job smoke check completed through `/api/chat/start` plus `/api/chat/status`, returned `ready: true`, no blockers, and generated `Hello` from `hello world` in about `25` seconds for `1` token
- runtime generation now supports cooperative cancellation through the prompt loop, layer stack, K/V decode step, and streamed `lm_head` decode-tail chunks
- the latest real Qwen cancel smoke check started a Quality job, requested cancel, and reached `canceled` in about `4` seconds with no final result kept
- the latest post-cancel completion smoke check still completed normally through the background job route, generating `Hello` with no blockers in about `25` seconds for `1` token
- runtime math now auto-configures PyTorch CPU threads, using a safe high-thread default with `PCKETLM_TORCH_THREADS` as an override; on the current 16-core machine this selects `14` threads
- the latest real Quality timing check for `hello world`, `max_new_tokens=2`, stayed correct with `Hello!` and no blockers in about `57` seconds
- the web launcher now uses a fixed local app port plus a single-instance lock port so repeated launches reuse the existing Pocket LLM server instead of piling up duplicate local runtimes
- the latest double-launch check left one actual server owning ports `8765` and `8764`, and the live web-job smoke generated `Hello` with no blockers in about `25.6` seconds for `1` token
- the prompt runtime now reports detailed phase timings, including prefill stack, decode tail, first selection, continuation stack, average layer time, and slowest layer
- the decode tail can now stream top-k candidates without materializing the full logits tensor when the active selection policy can choose from top-k directly
- the latest clean real Quality timing check for `hello world`, `max_new_tokens=2`, generated `Hello!`, ready with no blockers, in about `49.6` seconds
- the latest clean timing breakdown shows the remaining bottleneck is the full 48-layer stack: about `24.3` seconds for prefill stack and about `23.0` seconds for the continuation stack, while each decode tail is about `1.0` second
- the latest web API smoke check through `http://127.0.0.1:8765` generated `Hello`, ready with no blockers, from the background job route
- the full-stack speed pass now keeps tiny tensors cacheable across all layers, caches repeated RoPE trigonometry, keeps token-entry tensors in the active math dtype, and skips extra layer-summary reductions on the normal chat path
- the latest full test suite passes with `109` tests after the full-stack speed pass
- the latest clean real Quality timing check for `hello world`, `max_new_tokens=2`, generated `Hello!`, ready with no blockers, in about `42.7` seconds
- the latest clean timing breakdown shows prefill stack at about `20.6` seconds, continuation stack at about `20.0` seconds, and each decode tail below `1` second
- the latest web background-job smoke check through `http://127.0.0.1:8765` generated `Hello`, ready with no blockers, in about `21.0` seconds for `1` token
- the web UI now shows runtime timing and cache details after chat runs, including total time, stack timing, decode-tail timing, torch threads, math dtype, lm-head chunk size, cache hits, misses, and resident cache size
- the Settings screen now shows current runtime settings, and the sidebar/settings runtime copy now says the direct Pocket runtime is working even when the old full-RAM preflight still warns
- the latest full test suite passes with `109` tests after the runtime-visibility and status cleanup work
- the latest web background-job smoke check after the UI/status cleanup generated `Hello`, ready with no blockers, in about `22.9` seconds for `1` token, and returned timing/runtime settings through the API
- web chat cancel behavior is now cleaner: canceled jobs remove the temporary assistant placeholder instead of leaving a fake reply in the conversation, and Retry stays available
- the Benchmarks screen now renders richer result cards with generated text, layer count, ready/blocked state, tensor-cache hits, resident cache size, and latest runtime settings
- the Load Model screen now shows the active direct runtime status, local model folder, and a simple support plan for Qwen, Kimi, Kronos/Kronk, and Gemma instead of placeholder import/download cards
- the latest full test suite passes with `109` tests after the chat/benchmark/load product cleanup
- the latest cancel smoke check reached final status `canceled` with no result kept, and the latest normal web background-job smoke generated `Hello`, ready with no blockers, in about `23.6` seconds for `1` token
- the three free personalization templates now seed real per-model profile records under `models/<model_id>/profiles`
- Qwen now has saved `balanced-local`, `agent-coder`, and `low-memory` profile JSON records
- the web API now returns saved profile records plus a real default-vs-profile comparison summary
- the web UI now has a Compare section that compares the default Quality baseline with saved profile targets and latest benchmark data when available
- the latest full test suite passes with `111` tests after saved profile and Compare wiring
- the latest web status smoke check returned all three saved profile ids and `profile_compare.profile_count: 3`; the latest normal web background-job smoke generated `Hello`, ready with no blockers, in about `25.1` seconds for `1` token

Next milestone:
- continue heavy runtime engineering before more UI/profile polish: reduce repeated tensor loading, add safer memory behavior, and keep every runtime change verified with tests plus a real Qwen smoke check

Latest heavy-engineering update:
- the prompt runtime now reports operation-level stack timing, including tensor loading, norms, Q/K/V projection, RoPE, attention, output projection, MLP, and decode-tail phases
- the attention path now uses PyTorch scaled-dot-product attention by default with a manual fallback, plus cached causal masks for repeated decode shapes
- a persistent safetensors file-handle cache was tested and rejected as unsafe on the real Qwen run; it is now opt-in only instead of default
- the runtime now has safe batched tensor loading by shard for grouped misses, and the layer bridge uses it for norms, Q/K/V tensors, and the three MLP projection weights
- the latest clean real Qwen Quality CLI timing check generated `Hello!`, ready with no blockers, in about `40.5` seconds for `2` generated tokens
- the latest full test suite passes with `113` tests after the heavy runtime loading pass
- the web app was restarted on `http://127.0.0.1:8765`; the latest real web background-job smoke generated `Hello`, ready with no blockers, in about `20.7` seconds for `1` token
- heavy runtime foundation is now roughly in the `40` to `45` percent range for the first Qwen path: direct loading, full-stack prompt execution, KV carry, cancellation, timing visibility, conservative residency, and first safe speed passes exist, but long conversations, stronger decode quality, multi-model runtime generalization, and release-grade optimization artifacts are not done yet
- the default tensor residency front-layer window is now `12` layers while keeping the same `256 MB` cache cap, which improves reuse without allowing broad full-weight caching
- the latest clean real Qwen Quality CLI timing check after the residency-window pass generated `Hello!`, ready with no blockers, in about `36.1` seconds for `2` generated tokens; cache residency stayed under the cap at about `253 MB`, with no evictions
- a longer real Qwen Quality CLI check with `max_new_tokens=4` generated `Hello! How can`, ready with no blockers, in about `85.0` seconds
- the latest full test suite passes with `114` tests after the residency-window pass
- the web app was restarted again on `http://127.0.0.1:8765`; the latest real web background-job smoke generated `Hello`, ready with no blockers, in about `26.9` seconds for `1` token while system free RAM was very low
- the tensor residency policy now has a low-memory guard: when free RAM drops below `2 GB`, the runtime automatically falls back from the faster `12`-front-layer, `256 MB` cache setting to a safer `6`-front-layer, `128 MB` setting unless an advanced environment override is explicit
- the low-memory guard result is now visible in web runtime settings through `tensor_residency_policy`, including `memory_guard_active`, free memory, front-layer count, and max cache size
- the memory guard snapshot is cached briefly so the guard does not add measurable overhead during repeated tensor loads
- the latest full test suite passes with `116` tests after the low-memory guard pass
- the latest standalone real Qwen Quality CLI check after the guard pass generated `Hello!`, ready with no blockers, in about `41.5` seconds for `2` tokens; the latest web background-job smoke generated `Hello`, ready with no blockers, in about `21.9` seconds for `1` token
- benchmark history now summarizes recent measured benchmark JSON files with best, average, and worst timings per mode
- web status now returns `benchmark_history`; the latest app status check found `2` measured benchmark runs and all `3` saved profiles
- chat now accepts a saved `profile_id`; saved profile defaults can set runtime mode and default max-new-token count
- the web Chat screen now includes a profile picker, and runtime details now show cache policy and cache cap
- the latest full test suite passes with `118` tests after benchmark-history and profile-behavior wiring
- two short realistic CLI prompt checks were ready with no blockers, but the tiny `2`-token outputs confirm quality cannot be judged properly without the next decode-quality phase
- the latest low-memory profile web smoke completed with `profile_id: low-memory`, `32` carried layers, `3` generated tokens, and no blockers, but the generated text was mixed-language/rough; this confirms profile routing works and Balanced-quality behavior needs the high-effort decode phase
- current Qwen heavy-engineering foundation estimate: about `48%` complete. The runtime is working and safer, but quality/longer-conversation behavior and optimized artifacts remain major unfinished parts.
- high-phase quality pass completed: Low Memory now defaults to the safer Quality runtime path instead of the rough 32-layer Balanced preview path, while keeping short responses and high memory priority
- built-in saved profiles now refresh old default profile JSON records when their template runtime defaults change, so existing local Qwen profile files migrate from Low Memory/Balanced to Low Memory/Quality automatically
- web chat now passes Qwen chat stop markers to the runtime, and the runtime trims generated stop markers from visible assistant text
- the latest full test suite passes with `120` tests after the high-phase profile/runtime quality pass
- the web app was restarted on `http://127.0.0.1:8765`; `/api/status` reports direct runtime `Working` and Low Memory runtime mode `Quality`
- the latest real Low Memory web chat smoke generated `Hello`, ready with no blockers, in about `26.6` seconds for `1` token; an earlier 3-token Low Memory smoke generated `Hello! How`, ready with no blockers, in about `57.6` seconds
- current Qwen heavy-engineering foundation estimate: about `52%` complete. The customer-facing weak-hardware profile no longer uses the known-rough Balanced output path, but true speed optimization, longer conversation quality, and optimized artifacts still remain.
- medium prep phase completed for the next high-speed phase: measured benchmark cases now persist raw runtime timings plus compact timing summaries for stack time, tensor-load time, decode-tail time, and the current bottleneck
- benchmark history now aggregates average stack, tensor-load, and decode-tail timings when fresh benchmark runs include timing summaries
- the Benchmarks screen now displays timing chips for fresh runs/history, so the next speed pass can be judged from the app surface instead of terminal-only notes
- the latest full test suite passes with `120` tests after the benchmark timing-summary pass, and `python -m compileall src tests -q` completed cleanly
- the web app was restarted on `http://127.0.0.1:8765`; the latest real Low Memory web smoke generated `Hello`, ready with no blockers, in about `19.0` seconds for `1` token, with reported timing total about `19.0s` and prefill stack about `17.7s`
- current Qwen heavy-engineering foundation estimate: about `56%` complete. The next high phase should target full-stack speed, especially repeated layer execution/tensor loading, while preserving the Quality path.
- high-phase context/fidelity pass completed: web chat now injects a runtime identity header with Pocket LLM context, current loaded model, runtime mode, active profile, and instructions for model-identity questions
- web chat responses now return a `runtime_context` object so the app can display/debug which model/profile/mode was actually used
- tokenizer/generation metadata JSON reads now use a small mtime-aware cache, and web chat token-format support is cached per model id
- the latest full test suite passes with `121` tests after the runtime identity pass, and `python -m compileall src tests -q` completed cleanly
- the web app was restarted on `http://127.0.0.1:8765`; model-identity smoke generated `Qwen2.5-14`, ready with no blockers, in about `184.8` seconds for `8` tokens, with response context showing `Qwen2.5-14B-Instruct`
- the latest basic-logic smoke generated `YES` for `If 2 plus 3 equals 5, answer YES only.`, ready with no blockers, in about `27.0` seconds for `1` token
- current Qwen heavy-engineering foundation estimate: about `61%` complete. Runtime identity and basic logic/context are improved, but speed is still the main blocker; 8-token identity output took over three minutes.
- high-speed phase completed for deterministic runtime facts: Pocket LLM now answers clear model-identity questions locally from runtime context instead of spending a full Qwen generation on known app metadata
- the layer bridge config is now cached as an mtime-aware object, avoiding repeated config object construction during full-stack layer passes
- the latest full test suite passes with `123` tests after the local runtime-context answer path, and `python -m compileall src tests -q` completed cleanly
- the web app was restarted on `http://127.0.0.1:8765`; `Which model do you run on?` now returns `Qwen2.5-14B-Instruct (qwen2.5-14b-instruct)` in about `0.015` seconds with strategy `local-runtime-context-answer`
- the real Qwen logic path still works after the shortcut: `If 2 plus 3 equals 5, answer YES only.` generated `YES`, ready with no blockers, in about `29.0` seconds for `1` token
- current Qwen heavy-engineering foundation estimate: about `64%` complete. App-known runtime facts are now fast and reliable, but general Quality generation still needs true full-stack speed work.
- low-memory stability phase completed after a real crash: a normal Qwen timing request killed the web server when free RAM was around `1.2 GB`, so Pocket LLM now blocks model generation cleanly below `2 GB` free RAM instead of starting a run likely to crash
- the tensor residency low-memory guard now becomes conservative below `3 GB` by default, with `PCKETLM_TENSOR_CACHE_LOW_MEMORY_GUARD_MB` as an advanced override
- the latest full test suite passes with `127` tests after the hard memory guard, and `python -m compileall src tests -q` completed cleanly
- the web app was restarted on `http://127.0.0.1:8765`; with only about `0.74 GB` free RAM, model identity still answered instantly, while a real Qwen logic prompt returned a clean `memory-guard` blocker instead of crashing the server
- current Qwen heavy-engineering foundation estimate: about `65%` complete. Stability under low RAM improved, but true general-generation speed still needs more free memory and deeper full-stack optimization.

Owner:
- Licht + Codex

Latest speed-core update:
- runtime-pack auto-selection is now in place, so Pocket LLM selects a pack by current free-RAM budget instead of blindly taking the newest/largest artifact
- a larger real Qwen `speed-core` pack was built with `125` tensors and about `481.0 MB`, covering front `4` layers of Q/K/V/O attention projections plus small tensors
- live proof showed that pack is not safe as the default on this laptop yet: it pushed free RAM down to about `1.77 GB` during generation and slowed a short run to about `54.9s`
- the selector now keeps the safer `140.97 MB` Low Memory pack active under the current RAM range; the app reports this through the new `speed_status` payload
- final safe real web smoke returned `OK`, ready with no blockers, in about `37.4s` for `2` tokens; the speed wall is still tensor loading, not UI or routing
- the latest full unit suite passes with `134` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `83%` complete. The product is safer and better instrumented, and it now avoids a bad large-pack path automatically, but true customer-grade speed still needs a deeper execution strategy than safetensors disk streaming.

Latest reuse/engine update:
- exact repeated chat requests now reuse the prior result instead of running Queen again; this is intentionally narrow and only applies when model, profile, mode, prompt, messages, and generation settings match
- live proof: after one real Queen run, the same `OK` request returned through `/api/chat` in about `0.454s` wall time with `response_reuse.hit: true` and model elapsed `0.0s`
- status now reports response-cache counters and a runtime engine decision; the current machine selects `direct-cpu` / `torch-cpu` because no supported GPU backend is available yet
- the latest full unit suite passes with `138` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `85%` complete. Repeated checks/retries are now fast and engine choice is explicit, but brand-new general prompts still run through the slow CPU Quality path.

Latest instant-job update:
- exact repeated requests now short-circuit the web app's background job path too; `/api/chat/start` can return an already completed cached job
- live proof: after a real Queen `OK` run took about `37.83s`, the repeated `/api/chat/start` request returned completed in about `0.007s` with `job_fast_path: response-cache`
- the web UI now handles this immediately instead of waiting for the polling interval, and labels reused replies as cached
- the latest full unit suite passes with `139` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `86%` complete. Retried/repeated prompts are now effectively instant through the product path, but new prompts still need true prefix/KV reuse or backend acceleration.

Latest Quick-speed update:
- added a new `Quick` chat mode that keeps the full 48-layer Quality stack but caps generation to `1` token, so it is a short-answer path rather than a rough partial-layer path
- request-scoped safetensors handle reuse now runs automatically only for one-token Quick generations; multi-token Quality remains on the stable grouped-load path because forcing handle reuse across continuation decode was unsafe in live tests
- runtime-pack loading now reuses the selected artifact handle inside a one-token scoped request, reducing a Quick run to `8` original shard opens and `1` artifact-pack open in the live smoke
- live web proof after restart: `Quick` returned `OK`, ready with no blockers, in about `19.4s` wall time / `18.7s` model elapsed, with `48` layers active and `109` artifact tensor hits
- live Quality stability proof after restart: the normal two-token Quality `OK` path still returned ready with no blockers in about `36.1s`
- the latest full unit suite passes with `140` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `88%` complete. Pocket LLM now has an honest under-20s short-answer path for agent checks, while longer new prompts still need true session-prefix reuse or a different backend to get materially faster.

Latest session-prefix update:
- added an in-memory `session-prefix` cache for web chat sessions; the browser now sends a stable session id, and successful model runs can store a reusable KV prefix for that session
- the runtime can now accept a prior `KVDecodeState` plus verified token prefix and skip that matched prefix when the next prepared prompt starts with the same tokens
- prefix reuse is guarded by exact token-prefix matching, same model/profile/mode/layer path, a 20-minute TTL, and a small session-cache cap
- prefix reuse also has a safety limit: by default it only appends up to `8` new prompt tokens with the one-token KV path, because appending too many tokens one-by-one can be slower than full prefill on this CPU runtime
- live web proof after restart: first Quick run stored a `64` token reusable prefix; the follow-up matched all `64` tokens, but needed `14` appended prompt tokens, so Pocket LLM correctly fell back to full prefill and reported that reason instead of making the follow-up slower
- session-prefix cache status is now visible in runtime settings and speed status
- the latest full unit suite passes with `141` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `89%` complete. The real session-prefix mechanism exists and is guarded, but normal follow-up speed will only improve after the append path can process suffix tokens in batch or a better backend is added.

Latest batched-prefix update:
- replaced the one-token session suffix append path with a batched suffix prefill path, so a matched follow-up can process all new prompt tokens in one layer-stack pass against the stored KV prefix
- raised the default safe append limit to `64` prompt tokens for the batched path, while preserving exact token-prefix, model/profile/mode/layer, TTL, and cache-size guards
- live web proof after restart: first Quick run took about `20.7s` and stored a `64` token prefix; the follow-up reused all `64` tokens, batch-appended `14` prompt tokens, and returned `OK` in about `17.2s` wall time with no blockers
- timing proof: the reused follow-up spent about `14.1s` in `prefix_append_stack` plus about `1.1s` in decode tail, instead of doing a full prefill stack
- the latest full unit suite passes with `141` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `91%` complete. Follow-up reuse now produces a real speed win, but tensor loading still dominates, so the next big gain needs reducing repeated weight movement or changing backend.

Latest adaptive weight-movement update:
- tensor residency now has a selectable boosted preset instead of an automatic high-RAM boost; Standard stays at `256 MB` / `12` front layers, while Boosted can be chosen for `288 MB` / `13` front layers
- a larger `320 MB` / `15` front-layer setting was tested and rejected because the live follow-up slowed down and showed extra cache churn instead of a useful speed gain
- live web proof with the boosted policy active: first Quick run returned `OK` in about `19.6s`; the follow-up reused `64` prefix tokens, batch-appended `14` new prompt tokens, and returned `OK` in about `17.5s`
- the reused follow-up still spent about `12.7s` of its prefix append time loading tensors, so this phase improves policy safety and headroom use more than raw speed
- Settings now includes a Tensor Cache selector, and the chosen preset is saved under project state so users with more RAM can opt into Boosted without making it the default for weaker machines
- speed status and runtime settings now expose the active residency policy, including the chosen preset and whether the boosted policy is active
- the latest full unit suite passes with `144` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `92%` complete. The short-answer and follow-up path is stronger, but the next major speed gain should come from a different weight layout/backend strategy, not simply making caches bigger.

Latest boosted-pack update:
- added a new optional Boosted runtime pack layout that packs small repeated tensors, front `2` layers of Q/K/V projection tensors, and front `13` layers of K/V projection tensors
- the real Qwen Boosted pack is ready at `models/qwen2.5-14b-instruct/artifacts/runtime-pack-plan.boosted.small.safetensors`; it contains `153` tensors and is about `361.02 MB`
- Standard mode still selects the safer Low Memory pack at about `140.97 MB`; Boosted mode selects the new `361.02 MB` pack
- the rejected `481 MB` speed-core pack remains unselected by default, so stronger machines get the safer Boosted path first instead of the known bad large-pack path
- live Quick smoke comparison was safe but only a small win: Standard returned `OK` in about `18.5s`, while Boosted returned `OK` in about `17.9s`
- changing runtime presets now clears resident tensor cache, session-prefix cache, and exact-response cache so comparisons do not reuse stale answers from another preset
- chat responses now include current speed-status metadata, making the active pack visible after a run
- the latest full unit suite passes with `146` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `94%` complete. Optional boosted packing is real and safer than the failed large pack, but the gain is small; the next major speed improvement likely needs backend execution changes, not more safetensors repacking.

Latest backend-report update:
- added a full backend capability report for the current machine, covering Direct CPU, CUDA, DirectML, and llama.cpp/GGUF-style backend paths
- live report says Direct CPU is the active implemented fallback; CUDA is not visible to the current Torch runtime; DirectML is possible on Windows but the package is missing; llama.cpp/GGUF is the recommended next prototype but needs `llama-cpp-python` and a converted GGUF model file
- the backend report is now returned by `/api/status` and shown in the Settings screen under Backend Options
- the engine decision now carries the recommended future backend id, so Pocket can keep using the safe CPU path while pointing future acceleration work at the right target
- no large package install or model conversion was started in this phase, because GGUF prototyping would require user-approved install/conversion work
- the latest full unit suite passes with `148` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `95%` complete. Pocket now knows which backend path to pursue next, but the actual faster backend prototype still needs the GGUF package/conversion phase.

Latest GGUF adapter update:
- added the first optional GGUF backend adapter layer: Pocket can now discover local `.gguf` model files and has a prompt-runner interface for llama.cpp/GGUF
- added a Python `3.12` GGUF sidecar environment under `state/backend-envs/gguf-py312` so native backend packages do not have to run inside the main Python `3.14` app runtime
- added a sidecar runner script that can execute a GGUF prompt through `llama-cpp-python` once that package and a GGUF model file are present
- attempted a safe binary-only install for `llama-cpp-python`; no matching wheel was available for the current environment
- attempted the documented prebuilt-wheel route in the Python `3.12` sidecar, but pip fell back to source build and failed because Windows native build tools such as `nmake`/C++ compiler are not installed
- live status now reports the sidecar path, while still marking GGUF as not ready because `llama_cpp` is not installed and no GGUF Queen file exists yet
- the latest full unit suite passes with `152` tests, and compile verification is clean
- current Qwen heavy-engineering foundation estimate: about `96%` complete. Pocket has the GGUF adapter and sidecar shape ready, but the real speed prototype still needs a usable llama.cpp package/binary and a GGUF Queen artifact.

Latest Qwen 32B recovery update:
- Qwen 32B direct page runtime now has a stable default for full prompt/decode: scoped safetensor handle reuse is disabled by default for `qwen2.5-32b-instruct`, while the explicit env override still exists for debugging
- live 32B proof without any env workaround: `max_new_tokens=1` generated `Hello` in about `44.5s`, and `max_new_tokens=2` generated `Hello World` in about `77.3s`
- extended 32B proof: `max_new_tokens=4` generated `Hello World! It` in about `156.6s`, proving the default path survives multiple continuation steps
- 14B regression still passes: `max_new_tokens=4` generated the known baseline `Hello! How can` in about `69.4s`
- current Phase 2 recovery estimate: about `80%` complete. The native crash is fixed for the default path and 4-token 32B continuation is proven, but customer-facing guardrails and longer-run policy still need tightening before calling the 32B path production-stable.

Latest Qwen 32B guardrail update:
- added customer-facing direct-runtime guardrails for Qwen 32B across `/api/status`, speed status, chat responses, and the Load Model panel
- the app now reports `stable-slow`, `stable-low-headroom`, or `blocked-low-ram` for 32B, with the current free RAM, `4 GB` minimum, `5 GB` recommended headroom, and `4` proven new-token range
- the guardrail payload explicitly says scoped safetensor handle caching is disabled by default for 32B because that path caused native Windows access violations
- web regression suite passed with `32` tests
- current Phase 2 recovery estimate: about `84%` complete. The stable 32B path now has product-facing safety labels, but the next work should prove or block longer 32B runs through policy instead of relying on hidden engineering knowledge.

Latest Qwen 32B policy update:
- normal direct web chat for Qwen 32B is now capped to the proven `4` new-token range
- longer 32B output requires `allow_experimental_32b_tokens=true`, and the response keeps the guardrail warning
- web regression suite passed with `34` tests
- current Phase 2 recovery estimate: about `87%` complete. 32B now has a stable default path, customer-facing safety labels, and an enforced short-output policy.

Latest verification update:
- full test suite passed after the 32B default, guardrail, and policy work: `185 passed in 18.90s`
- current Phase 2 recovery estimate: about `88%` complete. The branch is green after the 32B stability and product-policy changes.

Latest 32B experimental-run preflight:
- skipped the experimental `8` token 32B live proof because free RAM was about `4830 MB`, below the new `5120 MB` recommended headroom
- the guardrail correctly reports `stable-low-headroom` and warns that longer-than-`4` token runs are experimental
- current Phase 2 recovery estimate remains about `88%` complete until the longer proof can run with enough RAM.

Latest Phase 2 completion update:
- after freeing RAM, Qwen 32B `full --max-new-tokens 8` passed with `Hello World! It's great to see` in about `331.0s`, ending at `1421 MB` process working set
- Qwen 14B regression still matches the baseline: `Hello! How can` in about `74.4s`
- the web policy now treats `8` new tokens as the proven local Qwen 32B range; longer 32B web replies still require `allow_experimental_32b_tokens=true`
- the latest full unit suite passes with `185` tests, and compile verification is clean
- current Phase 2 estimate: about `96%` complete. The 32B page-runtime path is proven and guarded for short output; remaining work is mainly broader long-run proof, speed, and product packaging.

Latest Phase 3 speed/reliability update:
- baseline numbers are now recorded for 14B and 32B at `1`, `4`, and `8` direct tokens; repeated tensor loading is the measured bottleneck
- normal Qwen 14B and 32B direct web replies now stay inside the proven `8` token range unless an experimental override is set
- Agent mode exists as a full-stack short-work mode and caps direct replies to `2` tokens
- live Agent proof on 14B generated `Hello!` in about `40.2s`, matching the new estimate closely
- focused web/benchmark tests pass with `43` tests, and frontend syntax check is clean
- current Phase 3 estimate: about `35%` complete. The app is safer and more honest about speed, but raw runtime speed still needs a backend or tensor-loading redesign.

Latest Agent reuse update:
- repeated Agent calls reuse session prefix correctly: the second live 14B call reused `64` prompt tokens and appended `14`
- speed did not materially improve because tensor loading still dominated: the second call took about `38.7s`
- chat responses now include a `performance_summary` field so future runs expose the bottleneck without manually reading raw timing keys
- current Phase 3 estimate: about `42%` complete. Reliability and measurement are much better; the next real speed work needs a warm runner or backend change.

Latest 32B proof-ladder update:
- Qwen 32B `12` tokens completed without a crash, but it used only `24` layers and produced bad mixed-language output, so it is not promoted
- explicit longer-than-proven direct web runs now force full-stack layer count; normal web chat still caps Qwen 14B and 32B at the proven `8` token range
- current Phase 3 estimate: about `48%` complete. Longer-run policy is safer, but true longer 32B quality still needs a slow full-stack proof or a better backend.
