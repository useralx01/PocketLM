# Todo

## In Progress

- Publish the multi-PC supervisor build to the private GitHub repository and verify its Windows CI run and downloadable prerelease.
- Decide whether staged streaming should auto-verify after each rotation in debug mode or only when explicitly requested
- Decide how much of the live streaming telemetry should appear in the desktop app by default versus debug views

## Next

- Run real model generation on each additional desktop after its local model storage is connected; installation supervision can verify software and hardware without downloading weights, but cannot claim real inference proof without them.
- Install a chosen Gemma 3 GGUF and run a real measured chat benchmark before promoting Gemma from `Missing`/fixture-verified to locally proven.
- Install a Kimi K2 GGUF only when storage permits; its production checkpoint is intentionally not downloaded by this storage-light task.
- Install the official Kronos package, Kronos-small weights, and tokenizer together, then run a real CPU forecast benchmark before promoting the adapter from fixture-verified.
- Run the existing DeepSeek CUDA proof command on a CUDA machine; this CPU-only host cannot produce a real GPU measurement.

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
- Define the profile schema
- Define the benchmark schema
- Rescope direct paged runtime speed around quantized execution, native fused CPU backend, GPU path, or a redesigned packed executor; cache/prefetch/zero-copy levers did not reach the target.
- Improve Mixtral MoE expert hit rate beyond the current `21.40%` by adding compressed expert residency or a hotter expert scheduler; Qwen3 now meets the 20-token MoE cache gate, but Mixtral does not yet have comparable reuse.
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
- Build the resumable lossless FP8 pack writer for the planned layout: persistent region, attention per layer, router per MoE layer, routed expert per layer/expert, shared expert per layer.
- Build the FP8 pack reader and route `run_fp8_moe()` expert payload reads through packed expert units when the artifact is ready.
- Add a conversion/progress state file for FP8 packing under `state/` like downloads and Q4 conversion.
- Benchmark DeepSeek layer 3 and bounded layers 0-3 from the FP8 pack before attempting another full 62-layer timing run.
- Build a batched selected-expert FP8 MoE path only after payload access is improved; current timing shows routed expert payload time, not shared expert math, is the first MoE target.
- Build or prove out a fused FP8 MLA attention/projection path; simple independent streamed attention exists but is not a default win.
- Add the next timing detail only where it directly drives one of those speed cuts, such as separating FP8 payload read time from native expert math time.
- Next product step: expose the bounded FP8 chat-template mode in the app once speed is less painful.
- Next DeepSeek product step: wire the concise FP8 status summary into the desktop/web model view once that UI is the active work surface.
- Next DeepSeek speed step: build a fused selected-expert FP8 kernel or batched expert executor; lossless packing plus hot cache passes the 50% timing gate, but `346.815s` for a bounded full-stack token is still not interactive.
- Next DeepSeek attention step: fuse MLA projection/attention work only where per-layer timing proves it beats the current materialized path.
- Next DeepSeek speed step after fused experts: attack FP8 MLA attention, now about `155s` of the `260.999s` full bounded run.
Done: PLM-1 Native FP8 dequant kernel.
Done: PLM-2 Native FP8 linear MLP gate.
- Resolve PLM-4 by adding a real DeepSeek FP8 monolithic backend or explicitly revising PLM-4 acceptance to allow Python orchestration with native FP8 kernels.
Done: PLM-5 DeepSeek C-side session + FP8 pack interface.
Done: PLM-6 Native DeepSeek router top-k kernel.
Done: PLM-7 Native DeepSeek MoE expert dispatch loop.
Done: PLM-8 C-side attention bridge for DeepSeek.
Done: PLM-9 DeepSeek monolithic forward boundary.
- Blocked: PLM-10 production routing speed gate needs a true native DeepSeek layer loop, not a C wrapper around `run_fp8_decode_loop`.
- Blocked: PLM-11 flash MLA speed gate needs fused/native q_a/q_b/kv_a/o_proj projection plus persistent attention weight residency; the online-softmax core alone is correct but not faster for the full call.
- Blocked: PLM-12 fused attention block speed gate needs BLAS-backed/batched native projection or GPU offload; the one-call C fusion is correct but slower than PyTorch/MKL on CPU.
- Blocked: PLM-13 effective `<=2s/token` needs GPU offload or a fundamentally faster full-layer engine; CPU FP8 speculative batching improved verification but misses the full-model target.
- PLM-14 follow-up only if Kaggle changes its API worker image: rerun `python tools\kaggle_gpu_smoke.py`; until then, use the browser Colab/Kaggle notebook for GPU validation.
Done: PLM-13 local resident-layer pager and header-only GPU residency estimator.
Done: PLM-13 local pager prefetch plumbing and telemetry.
Done: PLM-13 streamed local lm_head tail for resident-pager token decisions.
Done: PLM-13 local resident-layer KV cache carry.
Done: PLM-13 small local resident-pager multi-token decode loop.
Done: PLM-13 online Kaggle CUDA smoke and real DeepSeek remote resident/paged validation.
Done: PLM-13 GPU readiness gate for CUDA + local DeepSeek catalog + local FP8 pack.
- PLM-13 next: provide a CUDA worker with direct access to the `688 GB` local FP8 pack/source, then run the local resident-pager decode loop there.
- PLM-13 next: validate whether local pager prefetch overlaps pack/source reads with CUDA compute on that direct-access worker.
- DeepSeek exact CPU next: reduce the streamed `lm_head` tail, now `17.585s` of the cached 8-layer next-token row, using exact lossless caching or a native/top-k tail kernel that proves identical top-k.
- DeepSeek exact CPU next: reduce per-layer attention/FFN compute, now `5.241s` attention and `10.019s` FFN on the cached 8-layer next-token row; storage is not the current blocker because the row has `0` scattered reads.
- DeepSeek exact CPU hard unblock: replace the current PyTorch/Python FP8 attention and FFN path with a fundamentally faster exact matrix engine. After the full `lm_head` cache, cached 32-layer exact decode still spends `146.016s` in attention and `58.025s` in FFN with `0` scattered reads.
- Do not spend more time on native `lm_head` top-k, MLP span cache, small attention cache, or CPU thread tuning as defaults; all were measured and rejected for the full exact target.
- DeepSeek exact CPU only viable software unblock: a new exact attention/FFN matrix engine that is much faster than the current CPU PyTorch/native mix. The measured full 62-layer row is `386.038s`; reaching `<=10s` would require about a `38.6x` full-token speedup without dropping layers or changing precision.
- Do not claim more storage, FP8 packing, `lm_head` caching, or Smart App Control fixes can make full DeepSeek interactive on this CPU-only machine. Those walls have been removed or measured.
- After prefix attention cache, the remaining exact speed gap is still about `27.5x` from `275.327s` to `<=10s`. Any next attempt must replace the attention/FFN matrix engine itself, not just change cache shape.
- PLM-17 physical migration remains blocked until an internal NVMe volume with at least `1.38 TB` free exists, or until the source/pack storage strategy is changed without violating exact FP8 constraints. Continue PLM-18+ exact-path improvements against the current pack path and keep proof artifacts explicit.
- PLM-18 follow-up if pursuing full 61-layer residency on CPU: add a raw FP8 attention-span cache or another exact attention representation. The current dequantized attention cache cannot hold all layers inside 16 GB, so the guard correctly limits the proven resident window.
- PLM-19 follow-up only if revisiting expert kernels: the exact AVX-512 bit-decode path is safe and default, but isolated routed-expert math is still slower than scalar native on the real layer-3 microbench; future wins need a different exact reduction strategy or a batched GEMM-style route, not AVX-512 gather.
- PLM-20 follow-up: the original S4 proof averaged only `1.25` accepted tokens per verifier sweep; later MTP tree/rank work raised the current default to `[6,4]`. Keep future exact attempts focused on V3-MTP proposal acceptance or verifier sweep cost, not another single greedy chain.
- PLM-21 follow-up: the integrated exact path is proven and now defaults to the `[6,4]` rank-first top-3 continuation proof. Future exact speed work must preserve model, quantization, experts, and layers.
- MTP tree follow-up: punctuation-biased depth-2 improved the exact proof to `164.9231s/visible-token` with `2.0` accepted tokens/sweep, but has been superseded by the `60.614s/visible-token` rank-first continuation-rank3 default.
- MTP adaptive follow-up: depth-4 `punctuation-next2` improved the exact proof to `157.8086s/visible-token`, but has been superseded by the `60.614s/visible-token` rank-first continuation-rank3 default.
- MTP under-50 follow-up: current exact warm-prefill default is `37.4887s/visible-token` on the full 61-layer 10-visible-token proof, using `tree_depth=10`, `mtp_top_k=2048`, and `punctuation-rank-onepass-top2048`. Prompt-prefill warmup is exact and separately proven at `420.2098s`; measured generation uses the warmed exact prefill cache and commits all 10 visible tokens in one full verifier sweep.
- MTP next frontier: below-37 work must reduce the one remaining full verifier sweep or the top-2048 MTP-tail cost while preserving the same model, quantization, experts, layers, and exact verifier commit. Do not revisit branch pruning, broad finish depth, long rank-pattern chains, top-3 continuation, continuation-rank3 pruning, root-tail skip, final-tail depth reduction, smaller top-k/width, MTP-head warmup, inference-mode wrapping, or single-branch verifier wrapping unless new proof changes their measured blockers.
- MTP around-20 follow-up: do not revisit depth-20 widening, fixed 20-token ranks, root-tail skip, larger attention/MLP caches, direct MTP shared-head warmup, or prefill attention prefetch as defaults. The next credible route below `37.4887s/visible-token` is an exact verifier sweep speedup or a different proof prompt/long-output benchmark where V3's MTP head can honestly commit more than 10 visible tokens before the full verifier diverges.
- MTP below-34 follow-up: current exact default is now `34.2292s/visible-token` from `state/mtp-warm-prefill-mtphead-evict-before-verify-rank-onepass-top2048-depth10-full61-tenvisible.json`. The remaining measured generation cost is mainly the one verifier loop (`282.188s`, split into `145.2421s` FFN and `114.9234s` attention) plus MTP block/proposal work (`31.7227s` block after shared-head warmup). Next credible speed work should target exact verifier attention/FFN math or exact MTP block cost, not another shared-head warmup without the eviction guard.
- MTP regular tensor cache follow-up: broad default enablement was rejected at `42.6497s/visible-token`. Do not enable it by default; only revisit as a narrow filtered probe if a later profile shows a specific regular tensor read is dominating without added memory-pressure cost.
- MTP row-weighted many follow-up: reduced verifier proof did not improve FFN time, so do not promote `PCKETLM_ENABLE_FP8_MOE_ROW_WEIGHTED_MANY` without a new native kernel shape that handles multi-row route weights directly.
- MTP no-MLP-span follow-up: full proof regressed to `47.7969s/visible-token`, so keep the promoted MLP span cache defaults despite the misleading reduced-layer improvement.
- MTP attention FP32 cast-cache follow-up: reduced proof regressed to `51.2504s` verifier loop, so do not promote FP32 attention residency without a lower-memory per-layer cast reuse design.
- MTP top-k follow-up: `mtp_top_k=1304` regressed the full proof to `39.6258s/visible-token`; keep `2048` unless a new rank strategy changes the required candidate window.
- MTP MoE worker follow-up: worker parallelism regressed the reduced MTP verifier, so keep expert workers opt-in only.
- MTP attention matmul-core follow-up: full proof accepted only `6/10` candidates, so do not promote alternate attention accumulation orders unless they preserve the full 10-token verifier acceptance.
- MTP native many-MLP follow-up: disabling the native many-MLP path regressed reduced proof timing, so keep it enabled.
- MTP under-20 follow-up: current default is still `34.2292s/visible-token`; latest rejected exact probes are logged in `ERRORS.md` (blind depth-20 long prompt, one-worker pack preload, q/kv attention role cache, small role-cache overrun, and AVX-512 kill switch). Do not claim `20s cleared` without a ready full proof below `20s/visible-token`.
- MTP verifier-cost follow-up: the remaining plausible route is a new exact verifier engine or a proven longer MTP branch with high acceptance before running a full sweep. Cache reshaping and simple worker/role toggles have not cleared the gate.
- MTP AVX-512 full-MLP follow-up: do not promote `PCKETLM_ENABLE_NATIVE_FP8_MLP_AVX512_FULL`; it is bit-exact but slower on the reduced proof. Future verifier-engine work needs a different exact reduction strategy, not the current AVX-512 lane-product kernel.
- MTP row-weighted native-batch follow-up: do not promote `PCKETLM_ENABLE_FP8_MOE_ROW_WEIGHTED_MANY`; the new single-call row route kernel only shaved the reduced proof to `32.9885s/visible-token` and the full proof overran the under-20 gate before writing a ready JSON.
- MTP attention-side follow-up: do not promote narrow `kv_b` FP32 casts, q/kv role cache combined with row-weighted MoE, native FP8 attention-linear, or causal-mask caching. All were measured and rejected on reduced proofs; the mask cache was reverted.
- MTP branch/cache follow-up: do not pursue repeated-sky/counting long branches from the top-k64 reduced probes, do not disable native full-MLP, and do not re-run q/kv role-cache cap/policy variants as defaults without a new mechanism. None points at a below-20 proof.
- MTP SDPA follow-up: do not re-add the SDPA attention-core branch without a different exact layout; the direct fused SDPA attempt regressed reduced timing and was removed.
- MTP native batch-reuse follow-up: keep `PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE` opt-in. The reduced verifier-only row-weighted slice improved, but the full 61-layer proof did not emit a ready JSON before the stop gate. Future below-20 work needs a full-sweep verifier engine that reduces attention/FFN together, not only batch-reusing routed rows.
- MTP u16 projection follow-up: do not enable native u16 or Torch BF16 projection modes for the exact verifier; both changed the reduced verifier token stream and were slower.
- MTP long-rank follow-up: do not run blind long-rank proof launches as defaults. Build an incremental branch diagnostic that writes every draft step, selected rank, candidate token, and elapsed time before any full verifier sweep. Only spend a full 61-layer proof when the branch artifact shows a realistic acceptance path and a projected phase-clear window.
- MTP thread-count follow-up: do not lower the exact verifier to `PCKETLM_FP8_CPU_THREADS=4`; the reduced proof regressed versus the current default.
- MTP packed-span follow-up: do not enable `PCKETLM_ENABLE_FP8_PACK_MMAP_SPANS` for the exact MTP default; the reduced proof regressed.
- MTP router-cache follow-up: do not enable router-only `PCKETLM_FP8_REGULAR_TENSOR_CACHE_FILTER=mlp.gate.weight,mlp.gate.e_score_correction_bias` as a default; reduced timing improved, but the full exact proof regressed to `38.206s/visible-token`.
- MTP verifier telemetry follow-up: future below-20 attempts must cut full-path attention plus FFN together. The latest exact split shows `105.0389s` attention and `185.2804s` FFN, with FFN further split into `59.2928s` expert preload and `81.1853s` routed expert math.
- MTP MTP-layer cache follow-up: do not add `model.layers.61.mlp.experts` to the MLP span cache filter; the reduced proof regressed.
- MTP pack-worker follow-up: do not promote `PCKETLM_FP8_PACK_PREFETCH_WORKERS=16` from the reduced-only probe; it is not a full proof and does not project to the under-20 phase.
- MTP MoE worker follow-up: do not promote `PCKETLM_FP8_MOE_EXPERT_WORKERS=2`; the reduced proof regressed to `34.2567s/visible-token`.
- MTP native batch-reuse follow-up: do not promote batch reuse alone; `state/mtp-batchreuse-only-layers8-visible10.json` regressed to `38.8915s/visible-token` and confirms this knob is not the under-20 path.
- MTP AVX-512 full-MLP follow-up: the LUT-gather variant was exact but still slower than scalar on the reduced proof (`35.1863s/visible-token`) and was reverted; keep `PCKETLM_ENABLE_NATIVE_FP8_MLP_AVX512_FULL` opt-in.
- MTP q/kv plus pack-worker follow-up: do not combine q/kv attention role caching with worker `16`; the reduced proof regressed to `32.5517s/visible-token` versus q/kv-only.
- MTP pack-worker follow-up: do not promote worker `32`; it regressed to `34.925s/visible-token` reduced.
- MTP long-rank follow-up: do not run a full long-rank proof from the reduced window probe; it accepted only one reduced token and regressed to `36.4668s/visible-token`.
- MTP pack-worker follow-up: do not promote worker `16` without a ready full proof; the full launch for `state/mtp-packworkers16-full61-tenvisible.json` produced no JSON after 15 minutes.
- MTP native thread follow-up: do not set `PCKETLM_NATIVE_THREADS=4`; the reduced proof regressed to `36.6971s/visible-token`.
- MTP norm-cache follow-up: do not enable norm-only regular caching; `state/mtp-normcache-layers8-visible10.json` regressed to `36.1806s/visible-token`.
- MTP terminal-trim follow-up: do not enable `PCKETLM_ENABLE_FP8_MTP_TERMINAL_VERIFIER_TRIM` as a default; the full exact proof regressed to `41.4318s/visible-token` even though the trim was exact and anti-cheat passed.
- MTP hot-cache mmap follow-up: keep `PCKETLM_ENABLE_FP8_HOT_CACHE_MMAP` opt-in only. It improved the reduced proof but the full proof overran without writing JSON, so it is not a below-20 route.
- MTP cache-pressure follow-up: do not disable the FP8 hot cache as a speed route; the reduced no-hot-cache launch produced no JSON before the stop. Future cache work should add bounded pruning/telemetry, not blanket disablement.
- MTP attention dequant mmap follow-up: do not set `PCKETLM_DISABLE_FP8_DEQUANT_HOT_CACHE_MMAP=1`; the reduced copy-read probe regressed to `49.8599s/visible-token`.
- MTP thread-count follow-up: do not set `PCKETLM_FP8_CPU_THREADS=8`; it regressed reduced exact timing to `55.6239s/visible-token`, consistent with the rejected thread-4 and thread-14 rows.
- MTP inference-mode follow-up: do not wrap the exact MTP default in `torch.inference_mode()`; the correctly scoped reduced artifact regressed to `41.2255s/visible-token` versus the fresh `33.9547s` baseline.
- MTP root-miss follow-up: do not continue building long MTP branches after the root draft misses the exact verifier token; the forced-root continuation artifact regressed to `89.2289s/visible-token`.
- MTP one-pass top-k follow-up: do not enable `PCKETLM_ENABLE_FP8_MTP_ONEPASS_WINDOWED_TOPK` by default; the safe reduced artifact regressed to `36.6361s/visible-token`.
- MTP preload/MoE combo follow-up: do not combine worker-16 preloading with row-weighted MoE as a default; the reduced combo regressed to `36.32s/visible-token`.
- MTP attention-cache pin follow-up: do not pin `model.layers.61.self_attn` by default; the full proof regressed to `38.2928s/visible-token` after evicting verifier prefix attention entries.
- MTP attention-cache size follow-up: do not raise the exact attention cache to `4608 MB` with the MTP layer pin; the full proof regressed to `53.9244s/visible-token`.
- MTP MoE overlap follow-up: do not enable `PCKETLM_ENABLE_FP8_MOE_OVERLAP_SHARED_PRELOAD`; reduced timing regressed to `49.319s/visible-token` from contention.
- MTP verifier-scope MoE follow-up: do not promote verifier-only batchreuse+rowweighted, rowweighted-only, or active-pair pairweighted MoE. The first two full proofs hit memory pressure without ready JSONs, and the active-pair reduced branch did not beat rowweighted. Future below-20 work needs a broader exact verifier engine that reduces attention plus FFN without adding full-proof memory pressure.
- MTP below-20 follow-up: do not promote low-cache batchreuse+rowweighted, batchreuse+active-pair, trimshape9+batchreuse+rowweighted, verifier attention matmul-core, or `OMP_NUM_THREADS=4`. Each has a reduced artifact in `state/` and none projects to a full under-20 proof.
- MTP cache-filter follow-up: do not promote routed-all or shared-only MLP span cache filters. Warmed reduced proofs looked faster, but full exact proofs either regressed or failed to hold on default rerun (`36.5052s`, `36.1748s`, and `35.4807s`); the current default remains `34.2292s/visible-token`.
- MTP row-weighted chunk follow-up: do not promote `PCKETLM_FP8_MOE_ROW_WEIGHTED_MANY_CHUNK` or spend a full proof on chunked verifier-only batchreuse+rowweighted. Chunk8/chunk32 preserved reduced anti-cheat but regressed the clean verifier branch to about `42s`, losing the only useful unchunked signal.
- MTP long-output follow-up: do not reuse the canonical sky depth-20 branch for a phase proof. Find a different real prompt only after an incremental diagnostic proves the full DeepSeek verifier actually has a long visible continuation and V3's own MTP ranks can propose it within the exact top-k budget.
- MTP below-20 follow-up: do not pursue long-acceptance-only as the next phase route on the current verifier engine. Full length-40 verifier scaling is still `32.2436s/candidate` before MTP proposal cost, so below-20 now requires reducing exact verifier attention/FFN/router math itself.
- MTP attention follow-up: do not promote the native FP8 attention sequence projection path. `state/mtp-attn-native-linear-seq-layers8-visible10.json` changed verifier tokens and was slower; future attention work must preserve the materialized exact accumulation semantics or prove identical verifier ids first.
- DeepSeek GPU follow-up: run `python tools\deepseek_gpu_ready.py --model-id deepseek-v3 --run --json` on a CUDA machine and record the measured `run.elapsed_seconds / generated_token_count`; local host cannot produce this final CUDA speed because `torch.cuda.is_available()` is false.
