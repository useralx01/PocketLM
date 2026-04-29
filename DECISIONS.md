# Decisions

## Decision Log

### Phase 2 / Setup

- Phase 2 uses `py -3.14` for Python invocations because `py -3.14 -c "import torch; import safetensors; import tokenizers; import transformers; print('ok')"` returned `ok`.
- Phase 2 uses the existing memory helper at `src/pcketlm/core/runtime/load_attempt.py`, signature `def _memory_snapshot() -> MemorySnapshot`. The helper returns `MemorySnapshot(total_bytes: int, free_bytes: int)` plus `total_gb` and `free_gb`, so Phase 2 uses `free_bytes` for runtime free-RAM decisions.

### Phase 2 / Step 1 / Hardcoded shapes that must be parameterized for 32B

- Qwen2.5-14B-Instruct reference values from `models/qwen2.5-14b-instruct/original/config.json`: `num_hidden_layers=48`, `hidden_size=5120`, `num_attention_heads=40`, `num_key_value_heads=8`, `vocab_size=152064`, `intermediate_size=13824`.
- Audit result: no hardcoded 14B architectural shape values were found in the direct page-runtime path that needed replacing. The relevant bridge/runtime code reads these values from `config.json` or safetensors metadata already.

### Phase 2 / Step 3 / Qwen2.5-32B-Instruct reference config

- Reference source: `https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/raw/main/config.json`.
- Values used by synthetic 32B-shape tests: `num_hidden_layers=64`, `hidden_size=5120`, `num_attention_heads=40`, `num_key_value_heads=8`, `vocab_size=152064`, `intermediate_size=27648`, `max_position_embeddings=32768`, `torch_dtype=bfloat16`.

### Phase 2 / Step 4 / 32B residency budget

- Model-aware residency uses the existing `_memory_snapshot()` helper, subtracts a `4 GB` safety margin from free RAM, and caps the cache at `4 GB`. This gives Qwen2.5-32B more room for reusable converted tensors while leaving space for the active layer tensors, Python process, tokenizer state, and OS pressure.
- The model-aware budget only activates when a model catalog reports a deeper model than the standard front-cache policy was designed for. Explicit `PCKETLM_TENSOR_CACHE_MB` overrides still win.

### 2026-04-23

- Working name is `pcketlm`.
- The product direction is hybrid, not single-method only.
- Users should choose high-level tradeoffs like speed, quality, memory, and capabilities.
- The system should not expose raw weights or raw token removal as the normal interface.
- `pcketlm` will use a fully custom runtime path instead of building on another inference engine.
- V1 is dense-model only and Qwen-first.
- V1 is Windows-first and desktop-first.
- Reversibility means immutable originals plus derived artifacts and recreatable profiles.
- V1 should include a minimal chat shell alongside the runtime for testing.
- Future decision discussions should be presented as 3 options with one explicit recommendation, plus pros, cons, and failure risk.
- Tracker/readme files should be updated whenever meaningful project changes are made.
- If behavior is clearly bugged, debugging becomes the immediate default instead of a 3-option decision.
- While the full model download is still in progress, the best parallel work should favor failure avoidance and core-architecture hardening over surface polish.
- The first desktop UI work should start with the model status screen because it already maps cleanly to real backend state.
- The desktop model picker should prefer registry-backed entries and only fall back to a detected local source when the registry is still empty.
- For the current machine and the full Qwen2.5-14B source, the next runtime path should be staged disk streaming rather than plain CPU loading.
- Staged-streaming units should be sized by a shared segment budget that fits both the hot window and the warm prefetch window, not by the hot window alone.
- Staged-streaming manifests should treat the current project root as canonical and auto-rewrite stale in-project cache/model paths after a move.
- Cache verification should stay lightweight by recording checksums during materialization and verifying on demand, not by constantly re-hashing segments during normal runtime use.
- Staged-streaming residency and rotation position should live in a dedicated state file instead of being inferred only from the schedule or cache index.
- Cache verification should add cache-hit and cache-miss telemetry without resetting rotation step, refill count, or the current hot/warm residency position.
- Runtime control decisions should use the latest verification deltas for cache health and keep cumulative hit/miss totals as telemetry only.
- The product should expose both a fast `Advance Stream` path and a separate verify-first `Safe Advance` path instead of forcing one behavior for every user action.
- The first tensor-aware bridge should come from safetensors header metadata and the shard index, not from loading full tensors into RAM.
- The first tensor-aware execution plan should group tensors into ordered runtime units by embeddings, per-layer norm/attention/MLP, and decode head before we attempt true tensor loading.
- The first real tensor-loading slice should stay CPU-only and target small execution units first so we prove correctness before attempting larger execution blocks.
- Before attempting any true forward-pass math, larger loaded execution units should be verified against catalog and execution-plan metadata for dtype, shape, and byte size.
- The first true forward-pass bridge should stay single-token and CPU-only so we can avoid RoPE and KV-cache complexity while still proving real model math.
- The first true forward-pass bridge should load tensors one at a time instead of holding whole verified execution units in memory, because weak-memory machines can execute the slice that way even when full-unit residency is too heavy.
- The first multi-layer bridge should reuse the proven one-layer bridge as a stack of sequential low-memory steps instead of introducing a separate multi-layer math path too early.
- The first true model-entry path should use embedding-row slicing for specific token ids instead of loading the whole embedding table, because the embedding table is too large to treat as a normal always-resident tensor on weak machines.
- The first decode tail should stream `lm_head` row chunks and build logits incrementally, because the logits vector for one token is small enough to keep while the full `lm_head` weight matrix is too large to treat as resident memory on weak machines.
- The first repeated decode loop is allowed to be greedy and context-naive as a control-path proof, but it must explicitly say that it does not yet carry true multi-token context or KV-cache state.
- The first stronger decode-state step after the context-naive loop should carry a small recent-token embedding summary explicitly, while still clearly saying that this is not equivalent to true causal attention or KV-cache state.
- The decode path should support at least one alternative to greedy so behavior can be compared on the same runtime path before deeper decode work hides control-path bugs.
- The first real K/V-aware step is allowed to be RoPE-free as long as it explicitly says it carries projected K/V tensors but is not yet a full production decode implementation.
- Once the first K/V-aware loop exists, pcketlm should keep a small comparative decode benchmark so history-summary and K/V-aware behavior can be checked on the same seed before deeper runtime changes land.
- RoPE should be added at the actual live attention entry point for the carried K/V path rather than being approximated later in the outer decode loop.
- Once the K/V loop stops being RoPE-free, the next runtime-control step should favor an explicit decode-state object over loose cache dictionaries so future decode improvements have a stable state surface.
- Once decode state becomes explicit, stop conditions should come from real model metadata (`generation_config.json` / `config.json`) instead of being left as implicit loop behavior.
- The first real prompt-entry path should use the local tokenizer files directly and prefill the current K/V session with prompt tokens before any extra chat polish is attempted.
- Prompt-quality improvements should first use the model's own local metadata (`tokenizer_config.json` and `generation_config.json`) before inventing custom prompting behavior.
- The next prompt-quality layer should expose a few real user controls first, especially system prompt override, raw-prompt mode, and repetition-aware generation, before broader chat UI work begins.
- After the first prompt/session controls exist in the runtime, the next desktop step should be a small real prompt test panel with max-new-tokens and stop controls instead of waiting for a polished full chat UI.
- Prompt quality should improve first through more conservative default runtime behavior and a less artificially shallow prompt layer budget before adding more UI controls or sampling complexity.
- For short prompt sessions, the automatic prompt layer budget should prefer a deeper carried stack by default so fidelity improves before we spend more time on chat UI polish.
- Once prompt runs use a deeper default layer budget, the next fidelity step should favor true multi-token causal prefill over token-by-token prompt stepping before adding more chat-surface polish.
- When the live machine proves it can safely carry a deeper short-prompt stack, the automatic prompt budget should be raised instead of keeping the runtime artificially capped for convenience.
- Automatic prompt-depth decisions should stay heuristic and generic, but they should still incorporate real verified machine headroom when the live runtime proves a much deeper short-prompt stack is stable.
- When a short prompt session is both verified and stable at the full carried stack on the live machine, the default short-prompt heuristic should use that full stack instead of preserving an artificially smaller ceiling.

### 2026-04-27

- The long-term product promise is broader than running dense models on weak hardware: `pcketlm` should become a local LLM control center for weak hardware, personalization, agent-oriented profiles, comparison, and model management.
- The product should support both beginners and power users. The default experience should be plain-English and simple, with deeper controls available through advanced/debug views.
- The product identity is local-first. Cloud/API support may be considered later only if the project has the resources to support it, but V1 should not depend on cloud inference.
- The first model-family priority is Qwen, then Kimi, then Kronos/Kronk-family targets if a real supported open model path exists, then Gemma, then broader families later.
- The V1 universality path should be a universal importer/manager with dense text runtime support first, rather than attempting every architecture, MoE path, or multimodal path immediately.
- The desktop app should not force one rigid workflow. After import, the user should be able to chat, personalize, compare, inspect fit/readiness, or benchmark in the order they choose.
- `pcketlm` should load and manage models directly through its own runtime and storage system. It should not require Ollama, LM Studio, or another local LLM app as the runtime layer.
- User-facing explanations should be easy plain English with optional medium-depth detail, but release-facing documentation should avoid exposing enough internal implementation detail to make simple cloning of the system easier.
- Long-term optimization should include both runtime optimization and derived optimized artifacts, while keeping original models immutable and safe.
- Quality bar is high: the project should avoid releasing rough low-quality behavior as a finished product, even if that means a slower route to multi-model support.
- Codex should act as the main technical driver: ask the owner product questions, make normal engineering decisions autonomously, debug errors directly, and stop only for major product direction, risky system changes, large downloads, credentials, or resource-heavy work.
- After runtime or desktop chat changes, verification should include a real short Qwen prompt smoke check in addition to unit tests, because passing tests alone can miss slow/frozen user-facing behavior.
- Tensor conversion reuse should be memory-capped by default. `pcketlm` should not cache all converted Qwen weights automatically, because that would undermine the weak-hardware goal even if it makes one local smoke check faster.
- The default tensor cache should use front-layer admission, not broad LRU admission, because the runtime repeats layer passes from layer 0 and broad LRU caused eviction churn with no hits under the weak-hardware cache cap.
- Runtime math should default to bfloat16 for the current direct Qwen CPU path because it is substantially faster and uses less resident cache memory on the live machine, while float32 remains available through `PCKETLM_RUNTIME_MATH_DTYPE=float32` for compatibility.
- Desktop chat should default to Quality full-stack until Balanced mode produces acceptable output, because the product should not present rough speed-preview text as the normal answer path.
- Desktop Benchmark should run measured mode comparisons in the background, because real Fast/Balanced/Quality timing is useful but slow enough that it must not block the UI thread.
- The new downloaded design should become the main app shell through a local web UI served by Pocket LLM, while the old Tkinter implementation remains in the codebase as fallback/reference during the transition.
- Web chat should carry a bounded recent transcript into the runtime for now, not unlimited conversation memory, because this gives follow-up behavior without making slow local generation and prompt length explode on weak hardware.
- The web chat default should move from `2` to `4` max new tokens because `2` is too small for normal short replies, while higher defaults are still too slow for the current Quality path.
- Qwen web chat should use real `system/user/assistant` chat tokens for recent conversation turns when the local tokenizer supports them, because instruct models respond more reliably to proper role structure than to a pasted transcript.
- Short Quality chat runs should preserve the full layer stack for normal conversation prompts under `256` tokens, because dropping layers made follow-up answers drift even when the runtime stayed technically ready.
- Web chat should run through background jobs with polling rather than one long blocking browser request, because local Quality generation can take over a minute and the UI needs cancel/retry/status controls even before deep runtime cancellation is complete.
- Runtime cancellation should be cooperative and checked between expensive phases first, because forcibly killing tensor work would risk corrupting local state while phase-boundary cancellation still makes the app responsive enough for alpha use.
- Runtime speed tuning should prefer safe CPU-thread auto-configuration before large tensor-cache expansion, because bigger projection caching failed under current machine pressure while thread tuning preserved output and improved throughput without storing more weights.
- The web launcher should use a fixed local port plus a single-instance lock, because random free ports made repeated launches accumulate duplicate Pocket LLM servers and waste the weak hardware the product is trying to protect.
- Decode-tail optimization should prefer streamed top-k selection before larger memory-heavy logits/cache tricks, because it reduces repeated work without weakening the full-stack model path or storing more large tensors.
- The next Quality speed work should target the repeated full-layer stack, because detailed timing now shows token selection and decode-tail materialization are small compared with the 48-layer prefill and continuation passes.
- Full-stack speed work may cache tiny tensors across all layers by default, because layernorm weights and projection biases are small enough to help repeated passes without turning Pocket LLM into a full-weight RAM cache.
- Normal chat should skip debug-only per-layer summary reductions by default, while keeping them available for explicit diagnostics, because the customer path needs speed and the debug path still needs visibility.
- Web status should distinguish the working direct Pocket runtime from the older full-RAM preflight, because customers need to know chat can work locally even when a full plain CPU load would still be blocked on weak hardware.
- Runtime timing/cache details should be visible in the app after runs, because future speed work should be guided from the product surface instead of hidden terminal-only measurements.
- Canceled chat runs should not become assistant transcript messages, because a canceled local generation is a user control event, not model output.
- Load Model should show honest active/planned support state before import/download controls are fully implemented, because the app should not imply universal model loading before direct runtime paths are real.
- The three free personalization profiles should be saved as per-model records before they change runtime behavior, because persisted intent is safer than pretending optimized artifacts already exist.
- Compare should begin with honest default-vs-profile target summaries tied to latest benchmark data, and only later claim profile-specific performance once profile settings affect real chat runs.
- Persistent safetensors handle caching should stay opt-in because it passed unit tests but failed unsafely on a real Qwen run; normal runtime should prefer handles that close at the end of each grouped load.
- Layer-level batching should be evidence-driven and memory-bounded: norms, Q/K/V tensors, and MLP projection weights are currently worth batching, while output projection batching was slower on the live machine and should stay separate.
- The default tensor residency front-layer window can be `12` layers under the existing `256 MB` cap, because live Qwen checks showed a large speed win, no evictions, and resident memory still stayed under the weak-hardware cache ceiling.
- Tensor residency must automatically become more conservative under low free RAM. The default low-memory residency threshold is `3 GB`; below that, the runtime should prefer `128 MB` resident cache and `6` front layers unless the user explicitly sets advanced environment overrides.
- Saved profiles may begin as light runtime-behavior defaults before optimized artifacts exist. For now that means mode and max-token defaults, not claims of profile-specific optimized quality.
- Benchmark history should use persisted measured benchmark JSON runs first, because this gives trend visibility without adding another heavy benchmark path.
- Rough Balanced/Low-Memory profile output is now a decode-quality blocker, not a profile-system blocker. The next major work should move to high effort and focus on generation quality and longer conversation behavior.
- Low Memory should not use Balanced as a customer-facing default until Balanced output quality is fixed. Weak-hardware users should get the safer Quality path with shorter outputs and memory safeguards, even when that is slower.
- Built-in saved profile records can be refreshed from their templates while optimized artifacts are not ready, because these records represent defaults rather than user-trained/custom optimized artifacts.
- Fresh measured benchmarks should persist timing summaries in addition to raw timings, because the next speed phase needs an easy app-visible before/after comparison for stack, tensor-load, and decode-tail costs.
- The next high phase should target full-stack speed first. The latest live 1-token Low Memory/Quality smoke spent about `17.7s` of `19.0s` in prefill stack, so decode tail is no longer the main bottleneck.
- Web chat should always inject runtime identity context, because a trained model does not automatically know the local app, loaded model, runtime mode, or profile unless those facts are inside the current prompt.
- Basic logic/context smoke checks should be kept short until speed improves. The model answered a 1-token logic check correctly, but an 8-token identity answer took about `184.8s`, which is too slow for broader reasoning tests.
- Pocket LLM should answer deterministic app/runtime fact questions locally when the app already knows the truth, especially current loaded model identity. This is faster and more reliable than spending a full local model generation on metadata.
- Local runtime fact shortcuts must stay narrow and transparent: normal reasoning/chat prompts still go through the Quality Qwen path and return their model strategy.
- Web chat should refuse real model generation below `2 GB` free RAM. A live run crashed the server around `1.2 GB`, so a clear memory-guard blocker is safer than attempting generation and losing the app process.
- Hot tensor lookup should use an mtime-aware name index instead of repeated full catalog scans, because it reduces runtime overhead without changing model math or memory policy.
- Persistent safetensors handle caching must remain opt-in. Re-testing it as a default passed unit tests but killed the live Qwen web server, so the product path should close shard handles after grouped loads until a safer design is proven.
- Runtime identity context should stay compact. It is necessary for model/app awareness, but every extra prompt token increases full-stack prefill cost on weak hardware.
- Request-scoped safetensors handle caching must also remain opt-in. It was safer in design than persistent caching, but real Qwen web validation still killed the server, so the release path should avoid handle reuse for now.
- Clear model-identity wording can be removed from the injected prompt because deterministic identity questions are answered by the local runtime shortcut; the model only needs compact model/mode/profile context for normal generation.
- Web chat's hard free-RAM guard should be `4 GB` for the current Qwen path. Live diagnostics showed that starting near `3 GB` can still lead to extreme memory pressure and unusable latency.
- The optimized artifact path should begin with a manifest-only runtime pack plan before writing derived weights. This keeps the original model immutable and gives tests/status a real artifact contract without making false optimization claims.
- The next major speed leap should come from a derived runtime pack or storage layout change. Current source-shard execution loaded about `25.2 GB` for a 1-token prompt under pressure, so continuing to tune prompt text or Python metadata will not be enough.
- The first real runtime pack should contain only safe repeated small tensors. This proves artifact loading without duplicating the full model or risking a huge low-RAM write.
- Artifact-aware loading must always fall back to original safetensors when a tensor is not packed, because the first packs will be partial by design.
- The next artifact tier should target projection-weight layout, because the small pack reduced file opens but cannot remove the main `16s+` tensor-load bottleneck by itself.
- Tier 2 projection packing should start with only the first `2` layers of Q/K/V tensors. This proves the large-tensor artifact route with about `141 MB` of derived data instead of duplicating the whole model.
- Tier 2 speed validation should not bypass the `4 GB` RAM guard. If the machine is below the guard, Pocket LLM should report memory pressure and wait for headroom.
- Runtime pack selection should be RAM-budgeted, not newest-file-wins. The `481 MB` front-4 Q/K/V/O pack is useful evidence but was slower and pressured RAM on the live laptop, so the current default should keep selecting the `140.97 MB` pack until a larger free-RAM budget or a different backend makes the bigger pack worthwhile.
- Larger resident tensor cache presets should not become the default without live proof. Retesting `384-512 MB` style residency caused eviction churn and no useful warm-hit improvement, so the safe `256 MB` cap remains the release path.
- Exact chat-result reuse is allowed as a narrow session speed layer because it is transparent, keyed to the full request shape, and does not pretend to answer new prompts. True follow-up speed still needs prefix/KV reuse or a different backend.
- Runtime engine choice should be reported through an explicit selector even before GPU execution exists. This prevents CPU/GPU behavior from being mysterious and gives future DirectML/Vulkan/CUDA/llama.cpp-style work a clean place to plug in.
- Cached/local/guarded chat jobs may complete synchronously in `/api/chat/start`, because the background job layer is only needed for slow model generation. The UI should not wait for a poll when the server already has a final answer.
- Quick mode may use the full layer stack with a one-token cap, because this creates an honest fast short-check path for agents without presenting rough partial-layer output as normal quality.
- Request-scoped safetensors handle reuse may be automatic only for one-token generation. Forcing it through multi-token continuation decode is not stable enough yet, so the general Quality path should keep using the safer grouped-load behavior.
- Session-prefix KV reuse must prove an exact token-prefix match before it is used. If the next prompt does not start with the stored tokens, the runtime should fall back to full prefill rather than risk corrupting the conversation.
- The current CPU suffix-append path should have a small default token cap, because it processes appended prompt tokens one at a time. Longer matched suffixes should wait for batched append support instead of making follow-ups slower.
- Batched suffix prefill may use a higher default append cap than one-token append, because it processes the follow-up suffix in one layer-stack pass while still preserving exact-prefix and same-runtime-path guards.
- Tensor residency boost should be user-selectable, not automatic. Standard remains the default for weak hardware; Boosted is available for users with more RAM.
- Larger resident cache settings must be accepted by live proof, not theory. A `320 MB` / `15` front-layer setting was rejected because it slowed the real follow-up path, so the safe boosted default is `288 MB` / `13` layers.
- The next speed path should not keep pushing larger resident caches on this laptop. The live bottleneck is repeated large tensor movement, so the next meaningful work should be a better derived weight layout or backend strategy.
- Standard runtime selection should keep using the safe Low Memory pack even when more RAM is visible. Larger packs should require an explicit Boosted or experimental choice.
- The Boosted runtime pack should target K/V projection weights across more front layers before trying full Q/O/MLP projection packing, because live proof already rejected a broader `481 MB` pack.
- Changing runtime presets must clear exact-response and session-prefix caches, because cached answers from one preset should not be used as proof for another preset.
- Backend acceleration should be capability-reported before implementation. Pocket should show Direct CPU, CUDA, DirectML, and GGUF readiness separately instead of hiding everything behind one vague engine label.
- On the current machine, the next serious speed prototype should be llama.cpp/GGUF, not more safetensors repacking, because CUDA is not visible and DirectML is not installed.
- Large backend package installs or GGUF model conversion should be treated as an approval boundary, because they can take disk, time, and possibly network resources.
- Native GGUF runtime support should use a sidecar environment when needed instead of forcing the main Pocket app off Python `3.14`.
- If `llama-cpp-python` has no matching wheel, Pocket should not silently compile native code on a beginner machine. It should report the missing build tools or use a standalone prebuilt llama.cpp binary path.
- GGUF readiness requires both runtime support and a `.gguf` Queen artifact; having only one of those is not enough to mark the backend ready.
- The first GGUF implementation should use the official standalone llama.cpp Windows runtime, because it avoids forcing native compilation on a beginner machine and works even when `llama-cpp-python` wheels are unavailable.
- The official Qwen2.5-14B-Instruct Q4_K_M GGUF artifact is acceptable as the first Queen GGUF artifact because it is small enough to run locally while preserving the real 14B model family.
- Persistent `llama-server` should be the preferred GGUF path once available, because loading the GGUF model for each prompt is too slow while a loaded server can answer short prompts quickly.
- GGUF mode should not use the old direct-runtime free-RAM guard after the llama server is loaded. That guard protects the safetensors direct runtime, while the GGUF server has already paid its RAM cost and should be managed through backend lifecycle controls instead.
- GGUF is currently a power-user backend, not the weak-hardware default, because the loaded server uses roughly `8.6 GB` to `10 GB` RAM on the current machine.
- Full GGUF PID/RAM probing should live in the status/load-model surface, not in every chat response. Chat should return lightweight backend metadata so fast loaded-server answers are not slowed by process inspection.
- GGUF mode may use a higher max-token cap than Direct CPU modes because the persistent llama.cpp server has already loaded the model and can produce short complete answers without exercising the slower safetensors direct path.
- GGUF prompts should pass Qwen chat stop markers to the backend instead of relying only on token limits, because stop markers prevent visible template leakage and reduce mid-format continuations.

## Open Decisions

- Exact support definition for the first non-Qwen model family
- Exact public/private documentation boundary for release packaging
- Exact desktop shell technology for the finished product UI
## Phase 2 / Recovery / Crash class

No 32B crash class was assigned in this recovery pass because the required Qwen 14B baseline did not pass end-to-end.

Evidence:
- `load-config`, `embedding-only`, `embed-forward`, `layer-0`, `layer-0-1`, `layer-0-7`, and `layer-0-15` succeeded on Qwen 14B.
- Qwen 14B `full` exited with `-1073741819` / `0xC0000005` before the diagnostic emitted an `after full-prompt-decode` checkpoint.
- Per the recovery brief, 32B was not touched after the failed 14B baseline.

Verdict for this session: diagnostic recovery is blocked at the full prompt/decode-tail boundary on 14B. The next diagnostic should split `full` into prompt prefill stack versus final norm + lm_head/decode tail before retrying 32B.

## Phase 2 / Recovery 2 / 14B crash class

The 14B crash did not reproduce after clearing local Ollama/model background pressure.

Evidence:
- `all-layers` passed through all 48 layers.
- `all-layers-norm` passed final norm.
- `all-layers-norm-lm` passed lm_head streaming.
- `full --max-new-tokens 4` generated the prior baseline text: `Hello! How can`.

Verdict: environmental/native pressure was the likely cause of the prior 14B access violation in this recovery path. No layer-bridge or tensor-loader behavior fix was applied because the localized runtime components passed with real output.

## Phase 2 / Recovery 2 / 32B crash class

Qwen 32B is localized to the full prompt decode/KV path.

Evidence:
- `load-config`, `embedding-only`, `embed-forward`, `layer-0`, `layer-0-1`, `layer-0-7`, `layer-0-15`, `all-layers`, `all-layers-norm`, and `all-layers-norm-lm` all passed.
- `full` crashed with `-1073741819` / `0xC0000005` before emitting an `after full-prompt-decode` checkpoint.
- Qwen 14B `full --max-new-tokens 4` passed and generated `Hello! How can`.

Verdict: bug in 32B full prompt decode/KV path, not a 32B catalog/layer/final-norm/lm-head bug and not a proven hardware ceiling. Per the Recovery 2 stop condition, leave the 32B fix for the next session because it fails at a different slice than 14B.

## Phase 2 / Recovery 3 / 32B full split crash class

Qwen 32B is now localized to the `run_prompt_decode_loop` full-wrapper path, not to KV cache size growth and not to weight reload with KV present.

Evidence:
- `prompt-prefill` passed with `return_kv_cache=True`: 31 prompt tokens, 64 KV cache layers, `8,126,464` KV bytes, `44.645s`, final working set `1404 MB`.
- `decode-step-1` passed through prefill decode tail and first-token selection: chosen token id `9707`, final working set `1403 MB`.
- `decode-step-2` passed one true continuation call through `run_kv_decode_step`: cache sequence length grew from `31` to `32`, KV bytes grew from `8,126,464` to `8,388,608`, chosen token id `4337`, operation time `46.085s`, final working set `1413 MB`.
- `full --max-new-tokens 1` still crashed natively with exit `-1073741819` / `0xC0000005` after the diagnostic emitted `before full-prompt-decode` and before it could emit `after full-prompt-decode`.

Verdict: decode-loop bug. The KV cache is small relative to system RAM and grows successfully by one token, and the continuation step successfully reloads/runs all 64 layers while carrying KV. The crash correlates with the full wrapper path rather than isolated KV budget or weight reload behavior.

Follow-up evidence:
- Running the exact full path with `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=0` passed on Qwen 32B: generated text `Hello`, token id `[9707]`, operation time `40.317s`, final working set `1410 MB`.

Updated verdict: scoped safetensor handle-cache bug. The full decode-loop math is viable when scoped handle caching is disabled, so the release-safe path should keep `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=0` for Qwen 32B/full prompt decode until the native handle lifecycle is fixed.

## Phase 2 / Recovery 4 / Scoped cache default

Default scoped safetensor handle reuse is disabled for `qwen2.5-32b-instruct`.

Why:
- The old `auto` default used scoped handles for one-token prompt runs.
- Qwen 32B crashed natively with scoped handles but passed the same full prompt/decode path when scoped handles were disabled.
- After making the default safe, Qwen 32B passed `full --max-new-tokens 1` with `Hello` in `44.501s` and `full --max-new-tokens 2` with `Hello World` in `77.28s`.
- Qwen 14B still passed the regression baseline with `Hello! How can` in `69.39s`.

Explicit override:
- `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=1` still opts into scoped handle reuse for advanced debugging.
- `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=0` still forces it off for every model.
