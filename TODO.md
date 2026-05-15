# Todo

## In Progress

- Decide whether staged streaming should auto-verify after each rotation in debug mode or only when explicitly requested
- Decide how much of the live streaming telemetry should appear in the desktop app by default versus debug views

## Next

- MoE follow-up: make Qwen3 full decode report expert hit/miss telemetry in the final full slice, then attack the `~51s` warm tensor-load wall with an expert-aware packed/runtime path instead of dense-style safetensors reloads.
- Re-scope direct paged runtime speed around first-pass tensor IO/copy reduction. Sticky residency did not improve Qwen 14B `--slice=full` at one generated token.
- Investigate architecture-level direct runtime levers: persistent per-layer weight service, memory-mapped packed weights, larger contiguous derived packs, or a backend execution change that avoids reopening/copying large projection tensors per prompt.
- Phase 4 hardening: make the loaded GGUF path smooth across app refresh/restart where practical.
- Improve RAM messaging and recovery for the loaded GGUF server because this machine can drop below `1 GB` free RAM while the fast path is loaded.
- Add a visible loaded-speed proof in the UI after GGUF server is ready: seconds/token, server RAM, and last warm prompt latency.
- Make sure app refresh/restart preserves the fact that GGUF is the default fast chat path without silently starting a cold load.
- Keep direct dense runtime available as an advanced/research path, but do not let normal users enter it accidentally.
- Extend the new GGUF file list into a selectable artifact manager if multiple complete GGUF files exist for the same model.
- Add cleanup actions for redundant GGUF split shards or old merged artifacts after the user explicitly approves deletion.
- Next speed target: design a real reduction in repeated large tensor movement. Bigger resident-cache tweaks are now low-return; the next useful step should be a better derived weight layout/backend path that avoids reloading the same projection weights so often.
- Keep the `288 MB` / `13` front-layer residency boost as a selectable preset, not an automatic default; do not expand it again without a live speed win.
- Treat the new `361.02 MB` Boosted pack as optional, not a default release path, until longer follow-up tests show a bigger win than the first roughly `0.6s` Quick improvement.
- Next major speed path should investigate backend execution changes or more direct mapped-weight execution; safetensors repacking alone is now showing limited returns.
- Phase 4: build the first agent workflow foundation with plan/run/verify state, safe local file inspection, cancellation, logs, and model/backend choice.
- Continue session-prefix/KV reuse work beyond the first guarded implementation; exact response reuse is instant for retries, and prefix reuse now exists but needs batched suffix append to speed up normal follow-ups
- Next session-speed target: reduce repeated tensor loading during batched prefix reuse, because live follow-up proof now saves prompt work but still spends most time loading the same layer weights
- Use the new engine-decision status as the base for future backend work: DirectML/Vulkan/llama.cpp-style CPU/GPU hybrid experiments should plug into this selector instead of being hardwired
- Next high phase should stop trying to win mainly through larger safetensors packs on this laptop; live proof shows the `481 MB` pack creates memory pressure and slows down the current CPU path
- Design the next speed path around execution reuse or a different backend strategy: persistent session/KV reuse across turns, quantized/hybrid runtime options, or a llama.cpp/DirectML/Vulkan-style backend experiment
- Keep the new `speed_status` surface as the customer-facing explanation layer for why a pack was selected or rejected
- Continue heavy runtime work before profile/UI polish: target repeated full-layer tensor loading, longer conversation stability, and release-grade speed without falling back to rough partial-layer answers
- Run a fresh measured benchmark before the next high phase so the new timing-summary fields become the baseline for stack/load/tail comparisons
- Next high-speed phase: reduce Quality full-stack time enough that 8-token open-ended model answers do not take several minutes
- Phase 3 next speed target: reduce repeated continuation tensor loading, which currently costs about `131s` on 14B and `302s` on 32B for 8-token direct replies
- Use the opt-in warm Agent runner as the next test harness for deeper execution reuse; the runner lifecycle is stable, but repeated tensor/layer work still dominates speed
- Investigate the full-stack bottleneck shown by live smokes: deterministic identity is now local/instant, but real Qwen logic still spends about `16.4s` to `17.3s` in tensor loading for a 1-token run
- Keep request-scoped safetensors handle reuse automatic only for one-token Quick runs; forcing it through multi-token decode is still unsafe and should stay opt-in until redesigned
- Free enough system RAM before the next real speed benchmark; below `4 GB`, Pocket LLM now blocks generation to avoid crashing or entering severe memory pressure
- Expand the first real small-tensor runtime pack into a larger derived runtime pack that reduces repeated 25+ GB tensor movement per prompt
- Use the new tensor-load diagnostics to compare source-shard reads against the future optimized artifact path
- Build true conversation/session state reuse after the runtime can keep memory stable enough for longer chats
- Design the next artifact pack tier for large projection tensors without duplicating the full model recklessly or crashing low-RAM machines
- Add a benchmark row that compares original-source tensor loading against artifact-backed tensor loading using shard opens and artifact hits
- Run the Tier 2 artifact-backed speed smoke once free RAM is above `4 GB`; current live RAM was about `2.31 GB`, so generation correctly stayed guarded
- Expand Tier 2 beyond the first `2` Q/K/V layers only after the measured smoke proves memory and speed behavior are acceptable
- Add a safer memory-pressure guard around larger grouped loads before expanding batching beyond the proven norm/QKV/MLP groups
- Improve Balanced-mode output quality before presenting it as a customer-facing speed mode
- Expand realistic prompt checks after speed improves enough to make 20+ token answers practical
- Add deeper saved profile behavior beyond mode/token defaults only after the runtime path is less fragile
- Replace remaining placeholder actions in Load Model, Personalize, Agents, and Settings with real backend actions as each backend is ready
- Define the first non-Qwen support target and what "supported" means for import, readiness checks, direct runtime loading, chat, personalization, and comparison
- Expand persisted lightweight benchmark runs with measured real prompt timings and output summaries when the user explicitly starts a slower benchmark
- Expand the first profile templates into saved profile records under each model's profile directory
- Wire Compare to real default-vs-profile summaries once saved profile records and benchmark results exist
- Use the new tensor residency benchmark stats to compare cache behavior across Fast, Balanced, and Quality modes
- Add an advanced opt-in tensor cache preset only if repeated real measurements prove it improves speed more than the safe front-layer default
- Improve Balanced-mode output quality so the faster mode can become more useful without producing rough text
- Expand benchmark history further only after fresh measured runs populate the new timing-summary fields
- Define beginner vs advanced desktop modes so plain-English status remains visible while deeper runtime telemetry stays available without overwhelming normal users
- Surface the real RAM blocker inside the desktop UI
- Surface the staged-streaming residency state, cache hits, cache misses, and refill count inside the desktop UI
- Add the first runtime component that verifies cache files automatically after a rotation when debugging is enabled
- Add the first cache-residency metadata view to the desktop app
- Add the first runtime component that advances the stream based on residency telemetry instead of one-off manual rotation
- Add a desktop button or mode that verifies cache health before advancing when the user wants a safer path
- Add the first minimal decode-oriented loop after the layer bridge proves stable
- Decide the smallest honest next step after the first RoPE-aware K/V loop: more production-like decode state layout, or broader loop coverage
- Expand the decode benchmark from small chain comparison into a more formal regression benchmark with fixed seeds, persisted baselines, and expected summaries
- Add the next repeated token loop that carries more faithful context than the current first K/V-with-RoPE implementation
- Improve the first real prompt/tokenizer entry path so prompt handling is less rough and closer to usable chat
- Add stronger prompt/session quality controls such as richer sampling controls, configurable system prompts, and prompt formatting choices
- Turn the new desktop prompt test panel into a fuller prompt/chat test surface with cleaner session summaries and easier prompt presets
- Improve runtime-side generation behavior further so the desktop prompt panel produces less rough text, especially by strengthening decode fidelity beyond the current conservative greedy + multi-token-prefill prompt profile and the new full-stack short-prompt default
- Turn the desktop status-screen spec into the first real app UI when desktop implementation starts
- Use the risk register to guide the first real load debugging pass
- Define the first code folder layout
- Define the model registry schema
- Define the profile schema
- Define the benchmark schema
- Rescope direct paged runtime speed around quantized execution, native fused CPU backend, GPU path, or a redesigned packed executor; cache/prefetch/zero-copy levers did not reach the target.
- Improve Mixtral MoE expert hit rate beyond the current `21.40%` by adding compressed expert residency or a hotter expert scheduler; Qwen3 now meets the 20-token MoE cache gate, but Mixtral does not yet have comparable reuse.
- Implement first model file validation pass
- Implement first registry write/read flow
- Implement import readiness detection from real model folders
- Add a simple model status CLI for local inspection
- Add a simple local import CLI for model folders
- Add a dedicated source download-state command
- Add a registry-backed model catalog command
- Add a safe registry record removal command
- Choose the exact first supported Qwen model variant
- Decide the initial source format support
- Design the model registry and storage layout
- Add a registry-backed runtime source command
- Add a runtime bootstrap command that can target registry entries directly
- Attempt the first real dense-model load through the custom runtime path
- Define the first benchmark suite
- Choose the desktop shell approach
- Break the runtime into concrete modules and file layout

## Backlog

- Add a richer visual desktop acquisition panel
- Add a richer desktop model catalog view
- Add auto-refresh and multi-model selection to the desktop test UI
- Add warm-window refill logic that reacts to the current hot-window head
- Add benchmarking flow
- Add presets for different hardware tiers
- Add capability-based optimization profiles
- Add advanced research mode
- Add export/import of optimization profiles
## Phase MoE Correctness Follow-Up

- Replace the invalid 12-layer MoE speed shortcut with a correctness-preserving acceleration plan.
- Rework expert residency for full-stack Qwen3; current valid 20-token run is `27.923s/token` with `1.25%` expert hit rate.
- Add a small local MoE oracle fixture or an offloaded HF reference path so checkpoint-level cosine comparisons can run without loading 60-90 GB into RAM.
- Replace GGUF Qwen 14B as the Qwen3 speculative speculator, or run Qwen3 in a consistent non-thinking verifier mode; current Qwen2.5 speculator proposes direct-answer tokens while Qwen3 verifies with `<think>`.

## Phase Q4 Streaming Follow-Up

- Replace Python Q4 dequant in the paged runtime with native/fused dequant or a larger persistent dequantized fp16 window; current Qwen 32B Q4 artifacts are correct and coherent but slower than fp16.
- Replace the first scalar ctypes Q4 dequant kernel with SIMD/threaded fused unpack+dequant or a grouped packed executor; scalar native dequant is correct but slower than PyTorch vectorized dequant on Qwen 32B.
- Move Q4 speed work from raw dequant into grouped bridge/residency loading. SIMD dequant is fast now, but full Qwen 32B still spends `56.8791s` in layer-loop tensor loading where a standalone grouped Q4 pass takes `12.629s`.
- Add a fused/native multi-token Q4 MoE expert/layer compute path for appended prompt tokens. The latest warm-runner Q4 MoE row has tensor load down to `0.234s`, so the remaining `8.304s` second-turn wall is layer/expert math, not disk loading.
- Build and benchmark a different Q4 selected-expert math kernel that reduces the number of row-dot calls or fuses gate/up/down more deeply. Simple Python payload lookup caching, torch thread tuning, and OpenMP region reshaping did not beat the accepted `8.840s` follow-up row.
- After scoped safetensors handle reuse, the Qwen3-30B-A3B Q4 same-session follow-up target moved to `7.484s`. Next speed work should attack the remaining prefix append layer/expert compute, not add RAM-heavy expert byte caches.
- Do not revisit simple generated-prefix commit or a basic one-call Q4 selected-expert batch wrapper as defaults; both were re-measured after scoped handles and regressed.
- Do not default native thread-count tuning until it is repeatable across fresh runs; the one `6.908s` row was not stable.
- Next target after Q4 MoE prefix-prefill: teach the product/chat layer to choose useful reusable prefixes automatically instead of relying on the operator to set `PCKETLM_Q4_MOE_PRIME_PROMPT`.
- Next speed target after the Qwen3 thinking warm lane: reduce the `67s` prompt-specific warmup. Visible thinking output is now in the 1-2s/token band after warmup, but startup preparation is still too long for interactive cold prompts.
- Next Q4 MoE target: build a fused multi-token selected-expert payload/load path for warmup itself. The visible continuation now has packed-cache evictions at `0`; remaining cold/warmup work is generating the primed tokens, not visible decoding.
- Before real monolithic production routing, finish per-model tensor registration into the new C-owned session store and add native layer dispatch against those registered weights. Without that, a monolithic wrapper would still bounce through Python loaders and would not remove the real crossing overhead.
- Next monolithic integration step: implement real C-side decode math using the registered tensor table, then wire `layer_bridge.py` only after Qwen 14B tiny/full regression logits match the existing Python path. The anti-bluff counter must become nonzero before any speed claim.
- Replace the synthetic monolithic `write_logits` body with real C-side layer dispatch that calls the resolved component function pointers for norm, attention/KV, dense or MoE FFN, final norm, and lm_head. Tiny Qwen3/Mixtral oracle token identity must pass before production routing is enabled.
- Replace copied monolithic tensor registration with stable non-copy tensor handles or mapped pointers, and add BF16 dtype support to the monolithic dense path. This is required before local `qwen2.5-14b-instruct` can be routed through monolithic honestly.
Native BF16 follow-up: wire real Qwen 14B tensor catalog loads into `MonolithicForwardSession.register_tensor` lazily so production can use the no-copy BF16 monolithic decode path.
Native BF16 next: add exact monolithic prompt prefill/KV handoff for Qwen 14B, then turn on production routing behind `PCKETLM_ENABLE_MONOLITHIC_QWEN14`.
- Next DeepSeek runtime step: build a fused FP8 MLA attention projection path that combines q/kv/o work or reuses absorbed `kv_b`; simple independent streamed attention is available behind `PCKETLM_ENABLE_STREAMED_FP8_ATTENTION=1` but is slower.
- Next DeepSeek speed step: replace remaining Python-level FP8 attention work with fused kernels only where timing proves a win.
- Next DeepSeek prompt speed step: extend layer-wise prefill to batch more real prompt tokens and use the new elapsed fields to separate prefill, decode, attention, MLP, and tail costs.
- Next DeepSeek native speed step: go beyond dual gate/up by fusing the down projection or batching selected experts together.
- Next MoE speed attempt should fuse selected experts in native code rather than rely on Python thread-level expert workers.
- Full FP8 MLP fusion is now available but did not materially improve DeepSeek. Next speed work should prioritize fused MLA attention and/or a truly batched selected-expert kernel that reduces repeated file reads, not another per-expert MLP wrapper.
- Next chat path step: move from bounded layer-count text probes to product-facing full-stack attempts once the FP8 layer path is faster enough.
- Next DeepSeek speed step: full `0-61` bounded proof passes, so stop spending time on layer-count proof and reduce broad per-layer cost instead.
- Build a packed/reordered FP8 artifact path for DeepSeek routed experts and attention weights so cold external-disk reads are not the normal execution path.
- Build a batched selected-expert FP8 MoE path only after payload access is improved; current timing shows routed expert payload time, not shared expert math, is the first MoE target.
- Build or prove out a fused FP8 MLA attention/projection path; simple independent streamed attention exists but is not a default win.
- Add the next timing detail only where it directly drives one of those speed cuts, such as separating FP8 payload read time from native expert math time.
- Next product step: expose the bounded FP8 chat-template mode in the app once speed is less painful.
- Next DeepSeek product step: wire the concise FP8 status summary into the desktop/web model view once that UI is the active work surface.
