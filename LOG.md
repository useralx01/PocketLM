# Log

## Phase 2 / Setup

```text
phase-2-page-runtime-generalization
```

Working tree after checkout:

```text
On branch phase-2-page-runtime-generalization
nothing to commit, working tree clean
```

Python dependency check:

```text
py -3.14 -c "import torch; import safetensors; import tokenizers; import transformers; print('ok')"
ok
```

Disk check:

```text
C: free bytes: 719585067008
C: free GB: 670.17
Required minimum: 80 GB
Result: pass
```

## Phase 2 / Step 1

PowerShell wildcard note: the literal pytest argument `tests/test_runtime_*.py` does not expand under PowerShell, so the same test set was run with `Get-ChildItem tests -Filter 'test_runtime_*.py'`.

```text
tests/test_tensor_residency.py::test_load_resident_tensor_reuses_converted_tensor_when_within_policy PASSED [ 84%]
tests/test_tensor_residency.py::test_default_tensor_residency_policy_stays_standard_when_memory_has_headroom PASSED [ 85%]
tests/test_tensor_residency.py::test_tensor_residency_policy_boosts_when_preset_is_selected PASSED [ 86%]
tests/test_tensor_residency.py::test_default_tensor_residency_policy_stays_conservative_without_headroom PASSED [ 88%]
tests/test_tensor_residency.py::test_tensor_residency_policy_reduces_cache_when_memory_is_low PASSED [ 89%]
tests/test_tensor_residency.py::test_tensor_residency_policy_guards_below_three_gb_by_default PASSED [ 90%]
tests/test_tensor_residency.py::test_tensor_residency_policy_honors_guard_threshold_override PASSED [ 91%]
tests/test_tensor_residency.py::test_tensor_residency_policy_honors_explicit_cache_overrides_when_memory_is_low PASSED [ 92%]
tests/test_tensor_residency.py::test_load_resident_tensors_batches_misses_and_reuses_cached_results PASSED [ 94%]
tests/test_tensor_residency.py::test_load_resident_tensor_skips_tensors_larger_than_policy PASSED [ 95%]
tests/test_tensor_residency.py::test_load_resident_tensor_skips_layers_outside_front_cache_window PASSED [ 96%]
tests/test_tensor_residency.py::test_load_resident_tensor_keeps_small_tensors_across_all_layers PASSED [ 97%]
tests/test_tensor_residency.py::test_load_resident_tensor_clones_same_dtype_safetensors_view PASSED [ 98%]
tests/test_tensor_residency.py::test_clear_tensor_residency_cache_resets_counters PASSED [100%]

============================= 84 passed in 25.56s =============================
```

## Phase 2 / Step 2

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\isale\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\isale\Documents\pcketlm
configfile: pyproject.toml
collecting ... collected 3 items

tests/test_registry_catalog.py::test_build_model_catalog PASSED          [ 33%]
tests/test_registry_repository.py::test_load_model_registry_relocates_stale_project_paths PASSED [ 66%]
tests/test_registry_repository.py::test_registry_lists_qwen_14b_and_32b_entries PASSED [100%]

============================== 3 passed in 0.12s ==============================
```

## Phase 2 / Step 3

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\isale\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\isale\Documents\pcketlm
configfile: pyproject.toml
collecting ... collected 6 items

tests/test_runtime_tensor_catalog.py::test_build_tensor_catalog_reads_tensor_headers_and_groups_layers PASSED [ 16%]
tests/test_runtime_tensor_catalog.py::test_build_tensor_catalog_blocks_on_incomplete_source PASSED [ 33%]
tests/test_runtime_tensor_catalog.py::test_build_tensor_catalog_records_qwen32b_config_values PASSED [ 50%]
tests/test_runtime_tensor_execution_plan.py::test_build_tensor_execution_plan_groups_tensors_into_runtime_units PASSED [ 66%]
tests/test_runtime_tensor_execution_plan.py::test_build_tensor_execution_plan_blocks_when_catalog_is_not_ready PASSED [ 83%]
tests/test_runtime_tensor_execution_plan.py::test_build_tensor_execution_plan_handles_qwen32b_layer_count PASSED [100%]

============================== 6 passed in 1.82s ==============================
```

## Phase 2 / Step 4

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\isale\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\isale\Documents\pcketlm
configfile: pyproject.toml
collecting ... collected 24 items

tests/test_tensor_residency.py::test_tensor_residency_policy_uses_model_aware_budget_for_deep_models PASSED [ 37%]
tests/test_tensor_residency.py::test_load_resident_tensors_batches_misses_and_reuses_cached_results PASSED [ 41%]
tests/test_tensor_residency.py::test_load_resident_tensors_evicts_under_small_budget_for_32b_shaped_catalog PASSED [ 45%]
tests/test_tensor_residency.py::test_load_resident_tensor_skips_tensors_larger_than_policy PASSED [ 50%]
tests/test_tensor_residency.py::test_load_resident_tensor_skips_layers_outside_front_cache_window PASSED [ 54%]
tests/test_tensor_residency.py::test_load_resident_tensor_keeps_small_tensors_across_all_layers PASSED [ 58%]
tests/test_tensor_residency.py::test_load_resident_tensor_clones_same_dtype_safetensors_view PASSED [ 62%]
tests/test_tensor_residency.py::test_clear_tensor_residency_cache_resets_counters PASSED [ 66%]
tests/test_runtime_tensor_loader.py::test_load_tensor_by_name_reads_real_tensor PASSED [ 70%]
tests/test_runtime_tensor_loader.py::test_load_tensor_by_name_prefers_runtime_pack_when_available PASSED [ 75%]
tests/test_runtime_tensor_loader.py::test_runtime_pack_selection_snapshot_reports_selected_pack PASSED [ 79%]
tests/test_runtime_tensor_loader.py::test_load_tensors_by_name_reads_real_tensors_in_one_call PASSED [ 83%]
tests/test_runtime_tensor_loader.py::test_scoped_tensor_handle_cache_reuses_handle_across_load_calls PASSED [ 87%]
tests/test_runtime_tensor_loader.py::test_load_execution_unit_reads_grouped_runtime_unit PASSED [ 91%]
tests/test_runtime_tensor_loader.py::test_verify_loaded_tensor_matches_catalog_metadata PASSED [ 95%]
tests/test_runtime_tensor_loader.py::test_verify_execution_unit_matches_plan_metadata PASSED [100%]

============================= 24 passed in 1.95s ==============================
```

## Phase 2 / Step 5

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\isale\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\isale\Documents\pcketlm
configfile: pyproject.toml
collecting ... collected 28 items

tests/test_runtime_layer_bridge.py::test_run_minimal_layer_forward_bridge_supports_multi_token_input_with_causal_mask PASSED [ 21%]
tests/test_runtime_layer_bridge.py::test_run_minimal_layer_forward_bridge_supports_bfloat16_math_mode PASSED [ 25%]
tests/test_runtime_layer_bridge.py::test_run_layer_bridge_stack_executes_two_real_layers_in_sequence PASSED [ 28%]
tests/test_runtime_layer_bridge.py::test_run_layer_bridge_stack_iterates_qwen32b_layer_count PASSED [ 32%]
tests/test_runtime_layer_bridge.py::test_load_token_entry_hidden_state_reads_one_embedding_row_without_full_table PASSED [ 35%]
tests/test_runtime_layer_bridge.py::test_run_token_entry_layer_bridge_executes_from_token_id PASSED [ 39%]
tests/test_runtime_layer_bridge.py::test_run_decode_tail_streams_lm_head_and_returns_logits PASSED [ 42%]
tests/test_runtime_layer_bridge.py::test_run_decode_tail_can_stream_topk_without_full_logits PASSED [ 46%]
tests/test_runtime_layer_bridge.py::test_run_token_decode_step_produces_logits_from_real_token_entry PASSED [ 50%]
tests/test_runtime_layer_bridge.py::test_run_repeated_decode_loop_greedily_selects_next_tokens PASSED [ 53%]
tests/test_runtime_layer_bridge.py::test_build_history_summary_hidden_state_carries_recent_tokens PASSED [ 57%]
tests/test_runtime_layer_bridge.py::test_run_token_decode_step_can_use_history_summary_context PASSED [ 60%]
tests/test_runtime_layer_bridge.py::test_select_next_token_supports_greedy_and_top_k_sample PASSED [ 64%]
tests/test_runtime_layer_bridge.py::test_select_next_token_applies_top_p_filtering PASSED [ 67%]
tests/test_runtime_layer_bridge.py::test_run_repeated_decode_loop_can_use_top_k_sampling_policy PASSED [ 71%]
tests/test_runtime_layer_bridge.py::test_run_kv_decode_loop_carries_cache_lengths_across_steps PASSED [ 75%]
tests/test_runtime_layer_bridge.py::test_run_decode_benchmark_compares_history_and_kv_paths PASSED [ 78%]
tests/test_runtime_layer_bridge.py::test_run_kv_decode_step_can_advance_explicit_decode_state PASSED [ 82%]
tests/test_runtime_layer_bridge.py::test_run_prompt_decode_loop_uses_real_prompt_tokenization PASSED [ 85%]
tests/test_runtime_layer_bridge.py::test_run_prompt_decode_loop_can_cancel_before_heavy_generation PASSED [ 89%]
tests/test_runtime_layer_bridge.py::test_run_prompt_decode_loop_supports_raw_prompt_and_custom_system_controls PASSED [ 92%]
tests/test_runtime_layer_bridge.py::test_run_prompt_decode_loop_supports_min_new_tokens_and_stop_strings PASSED [ 96%]
tests/test_runtime_layer_bridge.py::test_prepare_prompt_text_supports_chat_wrapping_when_metadata_is_present PASSED [100%]

============================= 28 passed in 4.26s ==============================
```

## Phase 2 / Step 6 / monitored download started

The first blocking download attempt was interrupted before verification. The downloader now writes live status to `state/downloads/qwen2.5-32b-instruct.json`, and `download_status_cli --model-id qwen2.5-32b-instruct models/qwen2.5-32b-instruct/original` reports progress while the background download continues.

```text
live_status.status: downloading
live_status.expected_bytes_gb: 61.04
live_status.bytes_on_disk_gb: 0.06
live_status.progress_pct: 0.1
live_status.expected_file_count: 24
live_status.present_expected_file_count: 3
folder status: partial
folder bytes_on_disk: 110577986
missing_core_files: tokenizer.json, model.safetensors.index.json
```

Focused verification after adding the monitored downloader:

```text
53 passed in 4.02s
```

## 2026-04-28

- Added the first real GGUF/llama.cpp backend path for Queen/Qwen.
- Downloaded the official standalone llama.cpp Windows CPU runtime release `b8963` and verified `llama-cli.exe` works.
- Tried the `llama-cpp-python` package path, but the current Python `3.14` environment has no matching wheel and the Python `3.12` sidecar install tried to compile native code without Windows build tools. Pocket now records this as a blocked package route instead of forcing a beginner-machine compile.
- Downloaded the official `Qwen/Qwen2.5-14B-Instruct-GGUF` Q4_K_M split files and merged them into `models/qwen2.5-14b-instruct/artifacts/qwen2.5-14b-instruct-q4_k_m.gguf`.
- Added `pcketlm.core.runtime.gguf_backend` and `gguf_sidecar_runner`, plus runtime/backend status detection for `llama-cli`, `llama-server`, main-process `llama_cpp`, and sidecar `llama_cpp`.
- Added web `GGUF` mode and status reporting so the app can route chat through the new GGUF backend.
- Added persistent local `llama-server` use on `127.0.0.1:8767` so GGUF prompts can reuse the already-loaded model.
- Live GGUF checks: direct llama-server answered a one-token prompt in about `0.52s`; the Python adapter answered in about `0.39s`; the live web API answered `READY` with `strategy: llama-cpp-gguf-server` and `0.91s` reported generation time.
- Fixed a bug where the old direct-runtime RAM guard blocked GGUF mode after `llama-server` had already loaded the model. Added a regression test for that path.
- Re-ran verification after the GGUF guard fix: compile passed, focused web/GGUF tests passed, and the full suite passed with `158` tests.
- Added GGUF server lifecycle management around the persistent llama.cpp server: status, start/load, stop/unload, PID detection, process RAM reporting, and a small state file for the managed server.
- Wired `/api/gguf/server` into the web server and added Load/Unload controls to the Load Model screen.
- Live lifecycle proof: the API unloaded the GGUF server, confirmed port `8767` was no longer listening, then loaded it again in about `45.75s`; the loaded process reported ready with about `9.81 GB` RAM.
- A first post-control GGUF chat smoke showed the model generation was fast but the web response was still slow because chat rebuilt full process/RAM metadata after generation.
- Optimized the GGUF chat response path to return lightweight server metadata after chat while leaving full PID/RAM status for the Load Model/status surface.
- Latest live GGUF web chat proof: `Reply with SUN only.` returned `SUN` through `llama-cpp-gguf-server` with `1.07s` wall time and `0.61s` app-reported generation time.
- Re-ran verification after the control and metadata-speed work: compile passed, focused GGUF/web tests passed, and the full suite passed with `161` tests.
- Fixed the GGUF wrong-answer bug shown in the UI where `Reply with OK only.` could produce unrelated continuation text. The cause was raw completion prompting; GGUF mode now wraps Qwen prompts in the proper instruct chat format before sending them to llama.cpp.
- Live proof after the prompt-format fix: `Reply with OK only.` returned `OK`, and a follow-up formatted GGUF check returned `YES` in about `1.29s` wall time.
- Re-ran verification after the GGUF prompt-format fix: focused tests passed and the full suite passed with `162` tests.
- Added GGUF-focused measured benchmark cases for instruction following, 1/4/8/16-token generation, a basic logic check, and a short agent-style checklist.
- Added `run_gguf_measured_benchmark` and a web `/api/benchmark/gguf` route so GGUF can be benchmarked and saved without forcing the slow Direct CPU benchmark path.
- Added a `Run GGUF` button beside `Run full` on the Benchmarks screen.
- Live saved GGUF benchmark results: `GGUF 1` `0.54s` -> `OK`; `GGUF 4` `1.52s` -> `Local AI enhances efficiency`; `GGUF 8` `3.03s` -> `Pocket LLM provides concise answers and assistance`; `GGUF 16` `5.7s` -> cut off at the token limit; `GGUF Logic` `0.72s` -> `YES`; `GGUF Agent` `5.77s` -> began a numbered checklist but cut off at the token limit.
- Re-ran verification after the GGUF benchmark phase: compile passed, focused tests passed, and the full suite passed with `164` tests.
- Added GGUF stop-string support and visible stop-marker cleanup for server, Python package, sidecar, and CLI paths.
- Raised the web GGUF max-token cap to `64` while leaving direct CPU modes capped at `16`; the UI now switches the max-token input limit when GGUF mode is selected.
- Updated GGUF benchmark cases from the old cut-off `16` token/agent checks to `GGUF 32` and a `48` token agent check.
- Live longer-answer GGUF smoke: `List the next two safe steps for checking a local model...` returned two complete numbered steps through `llama-cpp-gguf-server` in about `21.57s`.
- Live saved GGUF benchmark results after tuning: `GGUF 1` `0.55s` -> `OK`; `GGUF 4` `1.56s` -> `Local AI enhances efficiency`; `GGUF 8` `4.0s` -> `Pocket LLM provides concise answers and assistance`; `GGUF 32` `10.37s` -> complete one-sentence explanation; `GGUF Logic` `0.93s` -> `YES`; `GGUF Agent` `12.63s` -> two complete numbered checking steps.
- Re-ran verification after the GGUF longer-answer phase: compile passed, focused tests passed, live smoke passed, live GGUF benchmark passed, and the full suite passed with `167` tests.

## 2026-04-23

- Created the `pcketlm` Mission Control tracking category.
- Set up the initial project tracking structure.
- Wrote the full V1 blueprint with scope, architecture, phases, and current risks.
- Added the first engineering roadmap while the initial model download was in progress.
- Added the first draft schemas and initial code layout while the model download was stalled.
- Logged the Hugging Face Xet download failure and restarted the model download with Xet disabled.
- Created the first Python package skeleton so implementation can start before the model download fully completes.
- Implemented the first registry, storage path, and Qwen file validation foundation.
- Added real source inspection so pcketlm can distinguish partial downloads from runnable model sources.
- Hardened the source inspector so partial or malformed JSON files do not break import detection.
- Added a first model status command so local source folders can be checked without opening Python internals manually.
- Stopped the stalled Qwen download path so the next acquisition attempt can start cleanly from LM Studio or another source.
- Added the first local import command so a downloaded model folder can be registered as soon as it is ready.
- Removed the temporary LM Studio / GGUF fallback logic after the official-source download path started working again.
- Added a download-state layer so pcketlm can report partial vs ready official sources more clearly.
- Added the first registry-backed catalog layer so pcketlm can list all known models and their current source states.
- Added registry removal support so stale model entries can be cleaned out when project direction changes.
- Added a standing rule that obvious bugs should be debugged immediately instead of being framed as normal option decisions.
- Debugged the registry/catalog mismatch and confirmed the live registry is now clean after stale-entry removal.
- Added a standing workflow rule: future decisions should be framed as 3 options with a clear recommendation, and tracker files should stay updated as changes are made.
- Added a product-facing acquisition layer so pcketlm can show clearer download progress, plain-English state, and next-step guidance.
- Verified the new acquisition layer against the live Qwen source folder and confirmed the official download is still partial but growing.
- Installed `pytest` so the local pcketlm test suite can be verified properly.
- Added the first runtime source-readiness layer and a runtime-source CLI so pcketlm can describe loader-facing source readiness directly.
- Verified the current runtime, acquisition, registry, and validation foundation with a passing 13-test local run.
- Confirmed the live Qwen source folder now has loader metadata present, but only 1 of 8 expected shard files is downloaded so far.
- Installed `transformers`, `tokenizers`, and `safetensors` for the first runtime correctness slice.
- Added a runtime bootstrap/preflight layer so pcketlm can separate source blockers, dependency blockers, and loader readiness before full model loading.
- Installed `torch`, reran the bootstrap checks, and confirmed the runtime is now blocked only by the incomplete shard download.
- Verified the current pcketlm foundation again with a passing 15-test local run.
- Rechecked the live Qwen download and confirmed it has reached about 17.5 GB / 27.51 GB, roughly 63.6%.
- Added an explicit risk register and tightened the blueprint/roadmap around failure avoidance while the model download continues.
- Rechecked the live Qwen download and confirmed it has reached about 21.0 GB / 27.51 GB, roughly 76.33%.
- Added the first desktop status-screen spec based on the real acquisition, runtime-source, and runtime-bootstrap states already implemented in pcketlm.
- Created a desktop shortcut at `C:\Users\isale\Desktop\pcketlm.lnk` pointing to the project folder so it can later be retargeted to the real app.
- Built the first Tkinter desktop test UI for the model status screen, verified it with a passing 9-test run, and repointed the desktop shortcut to launch `launch_pcketlm.pyw`.
- Connected the desktop UI to real registry model options and kept a detected-local-source fallback so the app stays usable before the registry is populated.
- Confirmed the official Qwen source is now fully downloaded and runtime-ready at the file level, with all 8 shards present.
- Registered `qwen2.5-14b-instruct` in the live pcketlm registry.
- Added a guarded real-load command, verified it with a passing 11-test run, and ran the first real load attempt against the full model.
- The first real CPU load was blocked honestly by available RAM: about 1.97 GB free versus an estimated 33.01 GB needed.
- Added reusable blocker category, severity, and recommended-action state so the app can explain the RAM blocker as stable product logic, not just ad-hoc text.
- Added a reduced-memory strategy planner, verified it with a passing test run, and ran it against the live Qwen model.
- The planner recommends `staged-disk-streaming` as the best next runtime path: plain CPU loading is not viable, but a streamed working-set path may be.
- Added the first staged disk-streaming planner, verified it with a passing test run, and ran it against the live Qwen model.
- The current adaptive plan suggests about a 0.86 GB hot window, 0.43 GB prefetch window, and about 33 streamed chunks, with cache state under `state/streaming/qwen2.5-14b-instruct`.
- Added the first staged-streaming manifest/bootstrap layer, verified it with a passing test run, and bootstrapped real cache state for `qwen2.5-14b-instruct`.
- The real manifest now lives at `C:\Users\isale\Documents\pcketlm\state\streaming\qwen2.5-14b-instruct\manifest.json` with hot and warm window directories beside it.
- Added the first staged-streaming manifest reader, verified it with a passing test run, and confirmed the live manifest reads back as ready with no cache-layout blockers.
- Added the first staged-streaming weight-unit mapper, fixed a test fixture issue in the unit-map tests, and verified the live unit map with a passing test run.
- The real unit map now lives at `C:\Users\isale\Documents\pcketlm\state\streaming\qwen2.5-14b-instruct\units.json` and maps all 8 shard files into streamable units.
- Copied the live `pcketlm` project out of `.openclaw` into `C:\Users\isale\Documents\pcketlm` and repointed the desktop shortcut to the new root.
- The old `.openclaw` copy is no longer the active project root and only remains as cleanup if Windows still holds a transient lock on that original folder.
- Removed the old `.openclaw` copy after the Windows lock cleared, so `C:\Users\isale\Documents\pcketlm` is now the only live project root.
- Debugged a post-move desktop status regression and found that the registry still pointed to the deleted old model folder.
- Added automatic registry relocation for stale in-project model paths, verified it with a new repository test, and confirmed the live desktop status now resolves the model from `C:\Users\isale\Documents\pcketlm\models\qwen2.5-14b-instruct\original`.
- Registry and streaming regression tests: 5/5 passed for the focused repository and streaming suite after the move fix.
- Added the first staged-streaming segment materializer, a CLI for it, and focused cache-materialization tests.
- Hit and fixed a circular import in `streaming_units.py` by importing the manifest reader directly instead of through the aggregate runtime package.
- Hit and fixed a second move-related stale-path bug in the streaming manifest so staged streaming cache paths relocate to the current project root automatically.
- Streaming tests: 9/9 passed for the focused reader, materializer, unit-map, and window-schedule suite.
- The live materializer now writes one hot cache file and one warm cache file for `qwen2.5-14b-instruct`, each about 388.5 MB, plus a `cache-index.json` file under `state/streaming/qwen2.5-14b-instruct`.
- Added lightweight write-boundary checksum recording to the cache index and an on-demand cache verification command.
- Added the first staged-streaming rotation flow, including persisted schedule loading and one-step hot/warm advancement.
- Hit and fixed a small rotation robustness gap so locked cache files return blockers instead of crashing the materializer.
- Hit and fixed a test expectation mismatch in the small rotation fixture because the segment splitter produced `6/5/5/5/5` bytes, not repeated `6`-byte segments.
- Streaming tests: 11/11 passed for the focused reader, verifier, materializer, unit-map, window-schedule, and rotation suite.
- The live Qwen cache verification passes cleanly before and after rotation, and the live rotation now advances from `unit-0001-seg-002` / `unit-0001-seg-003` to `unit-0001-seg-003` / `unit-0001-seg-004`.
- Added a persistent `residency.json` state file for staged streaming and wired materialization/rotation to keep it updated.
- Hit and fixed a small residency writer assumption so schedule-like test doubles without `overflow_units` no longer crash the materializer tests.
- Streaming tests: 13/13 passed for the focused reader, verifier, materializer, unit-map, window-schedule, rotation, and residency suite.
- The live residency state now persists the post-rotation position correctly: rotation step `1`, hot `unit-0001-seg-005`, warm `unit-0001-seg-006`, consumed `unit-0001-seg-004`, overflow head `unit-0001-seg-007`.
- Verified the first segment-level unit refactor and found two foundation issues: repo-root `pytest` still needed manual package path setup, and the original segment sizing fit the hot window but not the smaller warm prefetch window.
- Added `tests/conftest.py` so local test runs work from the repo root without manual `PYTHONPATH`.
- Added a repo-root `pcketlm` import shim so `py -m pcketlm...` commands work directly from the workspace root during development.
- Refactored the staged-streaming plan to include a shared segment budget that fits both the hot and warm windows.
- Rebootstrapped the live streaming manifest, regenerated the live unit map, and rebuilt the live window schedule with the shared segment budget.
- Streaming tests: 7/7 passed for the focused streaming manifest, reader, units, and window-schedule suite.
- The live Qwen streaming state now schedules one segment into the hot window and one segment into the warm prefetch window with no blockers.
- Added cache-hit and cache-miss accounting to cache verification and extended the residency writer so verification can preserve rotation/refill state while adding telemetry deltas.
- Added a regression test for the full `materialize -> rotate -> verify` sequence so verification no longer resets rotation step or doubles refill count.
- Streaming tests: 7/7 passed for the focused materialize, rotation, and residency suite after the verification-state fix.
- Rebuilt the live residency state from the current schedule after the old verification bug had polluted the counters, then reran verification on the fixed code path.
- The live staged-streaming residency state is now consistent again: rotation step `4`, hot `unit-0001-seg-005`, warm `unit-0001-seg-006`, consumed `unit-0001-seg-004`, refill count `4`, cache hits `2`, cache misses `0`.
- Added `last_cache_hit_delta` and `last_cache_miss_delta` to streaming residency so runtime control can react to the latest verification result instead of stale cumulative totals.
- Added a staged-streaming control layer and CLI so pcketlm can decide whether to materialize, repair cache, rotate forward, or hold position from live telemetry.
- Wired the desktop status screen to show streaming control state, current hot/warm/overflow units, and cache telemetry, and added a live `Advance Stream` button.
- Hit and fixed a controller-priority bug so current verification misses trigger `repair-cache-window` before the generic materialization path.
- Streaming and desktop tests: 12/12 passed for the focused control, residency, materialization, rotation, and status-screen suite.
- Ran one live controlled streaming step for `qwen2.5-14b-instruct`; the stream advanced cleanly from hot `unit-0001-seg-005` / warm `unit-0001-seg-006` to hot `unit-0001-seg-006` / warm `unit-0001-seg-007`.
- The live runtime control state now recommends another `rotate-forward` step with rotation step `5`, overflow head `unit-0001-seg-008`, refill count `5`, cache hits `2`, cache misses `0`.
- Debugged a desktop layout regression after adding the streaming control card and confirmed the window needed about `1369px` of vertical space while opening at only `760px`.
- Reworked the desktop app into a scrollable content area with a fixed bottom action bar so `Refresh Status`, `Open Model Folder`, `Retry Preflight`, `Advance Stream`, and `View Details` remain visible.
- Desktop status tests: 2/2 passed after the layout fix, and direct Tk widget inspection confirmed the live action buttons are mapped inside the visible window.
- Added a verify-first safe control path so the runtime can verify cache health, repair if needed, and only then advance the stream.
- Wired the safer path into both the runtime control CLI and the desktop app as `Safe Advance`.
- Streaming and desktop tests: 14/14 passed for the focused control, residency, materialization, rotation, and status-screen suite after the safe-advance work.
- Ran one live safe advance for `qwen2.5-14b-instruct`; verification passed and the stream advanced cleanly to hot `unit-0001-seg-010`, warm `unit-0002-seg-001`, overflow head `unit-0002-seg-002`.
- The live residency state after the safe advance is now: rotation step `9`, refill count `9`, cache hits `4`, cache misses `0`.
- Added a tensor-aware catalog layer that reads safetensors headers without loading whole weights and persists the result as `tensor-catalog.json` beside the streaming state.
- Added a runtime tensor-catalog CLI and focused tensor-catalog tests using tiny generated safetensors shards.
- Tensor catalog tests: 2/2 passed.
- Built the live tensor catalog for `qwen2.5-14b-instruct`; it found `579` tensors across `8` shards and `48` layers, with component-group counts of `336` attention tensors, `144` MLP tensors, `96` layer norms, plus embeddings, final norm, and lm head.
- Added a tensor-aware execution-plan layer that groups catalog entries into runtime units and persists them as `tensor-execution-plan.json`.
- Added a runtime tensor-execution-plan CLI and focused tests for grouped execution units and phase ordering.
- Tensor bridge tests: 4/4 passed for the catalog + execution-plan slice.
- Built the live tensor execution plan for `qwen2.5-14b-instruct`; it found `147` execution units with ordered phases `prefill`, `layer-entry`, `layer-attention`, `layer-mlp`, and `decode-head`.
- Added a first tensor loader slice that can load a single tensor by name or a grouped execution unit directly from the original safetensors shards.
- Added a runtime tensor-loader CLI and focused tests for both single-tensor loads and grouped execution-unit loads.
- Tensor loader tests: 6/6 passed for the catalog + execution-plan + loader slice.
- Loaded the live `layer-00-layer_norm` execution unit from the real Qwen model; it returned two real `torch.bfloat16` tensors from `model-00001-of-00008.safetensors` with no blockers.
- Added a stronger tensor verification pass for both single tensors and grouped execution units, with CLI support for `--verify-tensor` and `--verify-unit`.
- Tensor verification tests: 8/8 passed for the catalog + execution-plan + loader + verification slice.
- Verified the live `layer-00-attention` execution unit from the real Qwen model; all 7 tensors matched expected dtype, shape, and byte size.
- Loaded the live `layer-00-attention` execution unit from the real Qwen model; it returned 7 real `torch.bfloat16` tensors totaling `125843456` bytes with no blockers.
- Added a first minimal CPU-only layer bridge, CLI, and focused tests so pcketlm can run a real layer-0 execution slice instead of only loading or verifying tensors.
- Focused layer-bridge tests: 3/3 passed, and the catalog + execution-plan + loader regression suite also stayed green at 8/8 passed.
- The first live bridge attempt failed silently under memory pressure when it held full attention and MLP execution units in memory at once.
- Reworked the live layer bridge to load required norms, projections, and MLP tensors one at a time instead of keeping full verified units resident.
- The reworked live bridge now executes successfully on `qwen2.5-14b-instruct`; it produced a real `1 x 1 x 5120` `torch.float32` output tensor from the true layer-0 norm, attention, and MLP math with no blockers.
- Extended the layer bridge with a stack runner and CLI support for `--layers`, so we can chain multiple real layers without creating a second execution path.
- Focused stacked layer-bridge tests: 4/4 passed, and the catalog + execution-plan + loader regression suite stayed green at 8/8 passed.
- The live stacked bridge now executes successfully on `qwen2.5-14b-instruct` through layers `0` and `1`; it still uses synthetic single-token input and produced a real `1 x 1 x 5120` `torch.float32` output tensor with no blockers.
- Proved the stack can go deeper on the live Qwen model by running a four-layer synthetic single-token pass through layers `0` to `3`, still with no blockers.
- Added low-memory token-entry support using `safetensors.get_slice()` so pcketlm can read the needed embedding row for a token id without loading the full embedding table.
- Focused bridge tests are now 6/6 passed after adding token-entry coverage, and the catalog + execution-plan + loader regression suite stayed green at 8/8 passed.
- The live token-entry bridge now works for `qwen2.5-14b-instruct`; token id `42` is converted into a `1 x 1 x 5120` hidden state and then executed through layers `0` and `1` with no blockers.
- Added a low-memory decode tail using final norm plus streamed `lm_head` row chunks, so pcketlm can produce real logits without loading the whole output matrix into RAM.
- Focused bridge tests are now 8/8 passed after adding decode-tail coverage, and the catalog + execution-plan + loader regression suite stayed green at 8/8 passed.
- The live token-to-logits decode path now works for `qwen2.5-14b-instruct`; token id `42` is converted into a hidden state, executed through layers `0` and `1`, and decoded into a real `1 x 1 x 152064` logits tensor with no blockers.
- Added the first repeated greedy decode loop on top of the token-entry + layer-stack + decode-tail path, with explicit labeling that it is still context-naive and not KV-cache aware.
- Focused bridge tests are now 9/9 passed after adding repeated-loop coverage, and the catalog + execution-plan + loader regression suite stayed green at 8/8 passed.
- The live repeated loop now works for `qwen2.5-14b-instruct`; starting from token id `42`, it generated the two-step chain `42 -> 123571 -> 102664`.
- Strengthened the repeated loop with an explicit recent-token embedding-history summary and exposed it through `history_window` so the loop can carry more than the newest token.
- Focused bridge tests are now 11/11 passed after adding history-summary loop coverage, and the catalog + execution-plan + loader regression suite stayed green at 8/8 passed.
- The live history-summary loop now works for `qwen2.5-14b-instruct`; starting from token id `42` with `history_window=3`, it generated the two-step chain `42 -> 123571 -> 117864`.
- Added a token-selection policy layer with greedy and deterministic top-k sampling support on the live decode path.
- Focused bridge tests are now 14/14 passed after adding policy and K/V-aware coverage, and the catalog + execution-plan + loader regression suite stayed green at 8/8 passed.
- The live sampled history-summary loop now works for `qwen2.5-14b-instruct`; starting from token id `42` with `history_window=3` and `top-k-sample`, it generated the two-step chain `42 -> 123571 -> 123571`.
- Added the first real K/V-carrying decode step and repeated loop so projected keys and values persist across steps per layer.
- The live K/V-aware loop now works for `qwen2.5-14b-instruct`; starting from token id `42`, it generated the two-step chain `42 -> 123571 -> 10862` and reports cache sequence lengths of `2` for layers `0` and `1`.
- Added a small decode benchmark path so pcketlm can compare the current history-summary greedy loop, sampled loop, and K/V-aware loop on the same seed token.
- Focused bridge tests are now 15/15 passed after adding benchmark coverage, and the tensor catalog/execution-plan/loader regression suite stayed green at 8/8 passed.
- Added RoPE-aware handling on the live K/V key path inside the real attention bridge instead of only labeling the loop as K/V-aware.
- The live decode benchmark for `qwen2.5-14b-instruct` now compares `history-greedy`, `history-top-k-sample`, and `kv-greedy`; the RoPE-aware K/V case currently yields `42 -> 123571 -> 46322`.
- The live K/V-aware loop now reports `greedy-kv-cache-rope` and yields the two-step chain `42 -> 123571 -> 46322` with cache sequence lengths of `2` for layers `0` and `1`.
- Expanded the decode benchmark output so each case now records `steps_completed`, `final_token_id`, `unique_token_count`, and overall ready-case counts for regression-style comparisons.
- Reworked the live K/V path around an explicit `KVDecodeState` object so the runtime now carries next token, next position, generated chain, and cache sequence lengths as one state bundle instead of passing loose cache dicts around.
- Focused bridge tests are now 16/16 passed after adding explicit decode-state coverage, and the tensor catalog/execution-plan/loader regression suite stayed green at 8/8 passed.
- Extended the decode-state model again so it now loads EOS metadata from `generation_config.json` / `config.json`, tracks `finished` and `stop_reason`, and surfaces those values in the K/V loop and decode benchmark.
- The live Qwen benchmark and K/V loop both now report `stop_reason: step-limit` for the current 2-step runs, confirming that the runtime is treating decode as a bounded session rather than an open-ended demo chain.
- Added a local tokenizer runtime layer so pcketlm can load `tokenizer.json`, encode prompt text into token ids, and decode generated token ids back into text without leaving the local model files.
- Added the first prompt-entry runtime path so a real text prompt can prefill the current K/V session and then hand off to the existing prompt-generation loop.
- Focused bridge tests are now 17/17 passed after adding prompt-entry coverage, and the tensor catalog/execution-plan/loader regression suite stayed green at 8/8 passed.
- The first live prompt-entry runs now work for `qwen2.5-14b-instruct`; `hello world` tokenized to `[14990, 1879]` and generated `eligeçek`, while `write a short poem` tokenized to `[4934, 264, 2805, 32794]` and generated `Państwo엘`.
- Improved the prompt-entry path so it now reads tokenizer metadata, wraps plain prompts in a simple Qwen-style instruct/session format, uses local generation defaults, and forces prompt-prefill tokens instead of sampling during prefill.
- Focused bridge tests stayed green at 17/17 passed after the prompt-quality pass, and the tensor catalog/execution-plan/loader regression suite stayed green at 8/8 passed.
- The current live wrapped-prompt runs now use `top-k-sample-prompt-kv-cache-rope`; `hello world` generated `Ởgaard`, and `write a short poem` generated `อำนวยความ坐标`.
- Added the first prompt/session control layer so prompt-entry can now take a custom system prompt, raw-prompt mode, and explicit repetition penalty.
- Focused bridge tests are now 19/19 passed after adding prompt-control coverage, and the tensor catalog/execution-plan/loader regression suite stayed green at 8/8 passed.
- The live controlled prompt-entry path now works for `qwen2.5-14b-instruct`; wrapped `hello world` with `Be brief.` and repetition penalty `1.2` generated ` alguataka`, while raw `hello world` with the same repetition penalty generated `eligeuish`.
- Added explicit `max_new_tokens` and custom `stop_token_ids` support to the prompt runtime path and CLI so prompt sessions can stop more cleanly and predictably.
- Added a real desktop prompt test panel with prompt input, system prompt override, raw-prompt mode, max-new-tokens input, custom stop-token parsing, generated output, and detailed session summaries.
- Added focused desktop helper tests for stop-token parsing and prompt-result summarization.
- Desktop + prompt runtime tests: 24/24 passed across `test_desktop_main.py`, `test_desktop_status_screen.py`, and `test_runtime_layer_bridge.py`.
- Verified the updated live prompt runtime with `hello world` and `max_new_tokens=4`; it returned a ready result with strategy `top-k-sample-prompt-kv-cache-rope`, generated `setTypeものicielty`, and stopped on `step-limit`.
- Tightened prompt runtime defaults so prompt runs stay on `greedy` unless sampling is explicitly requested, even when local generation metadata advertises `do_sample`.
- Added a model-aware automatic prompt layer budget so prompt runs do not stay artificially stuck at 2 layers when no explicit layer count is provided.
- Updated the prompt runtime tests so the tiny Qwen fixture advertises sampling-capable generation metadata while the default prompt path still stays conservative and ready.
- Runtime + desktop tests: 24/24 passed again after the prompt-defaults change.
- Verified the updated live prompt runtime with `hello world` and `max_new_tokens=4`; it now returns `greedy-prompt-kv-cache-rope`, uses 4 carried layers by default on the real Qwen model, and generated ` underminlüNotFoundErroryü` with `stop_reason: step-limit`.
- Deepened the automatic prompt layer budget for short sessions so the runtime now aims higher on real models instead of staying at the earlier 4-layer default.
- Runtime + desktop tests: 24/24 passed again after the deeper auto-layer change.
- Verified the updated live prompt runtime with `hello world` and `max_new_tokens=4`; it now returns `greedy-prompt-kv-cache-rope`, uses 8 carried layers by default on the real Qwen model, and generated `findFirstOrCreateyü背上` with `stop_reason: step-limit`.
- Added real multi-token causal support to the layer bridge, including causal masking for multi-token self-attention while preserving RoPE and carried K/V handling.
- Switched prompt prefill from token-by-token stepping to a true multi-token prefill slice that builds the prompt cache in one causal pass before generation starts.
- Runtime + desktop tests: 24/24 passed again after the multi-token prefill change.
- Verified the updated live prompt runtime with `hello world` and `max_new_tokens=4`; it kept the same generated text `findFirstOrCreateyü背上`, but the live runtime check completed much faster because prompt prefill no longer walks token-by-token.
- Probed the real Qwen runtime with deeper manual prompt slices and confirmed that both 12-layer and 16-layer short prompt runs still complete successfully on this machine.
- Raised the automatic short-prompt layer budget so the default runtime now aims for 12 carried layers instead of 8 on short prompt sessions.
- Runtime + desktop tests: 24/24 passed again after the 12-layer default change.
- Verified the updated live default prompt runtime with `hello world` and `max_new_tokens=4`; it now uses 12 carried layers by default and generated ` mmcographedgeshift`.
- Probed the real Qwen runtime even deeper and confirmed that both 24-layer and 32-layer short prompt runs still complete successfully on this machine.
- Raised the automatic short-prompt layer budget again so the default runtime now aims for 24 carried layers instead of 12 on short prompt sessions, while still using prompt-length-sensitive reductions for longer prompts.
- Runtime + desktop tests: 24/24 passed again after the 24-layer default change.
- Verified the updated live default prompt runtime with `hello world` and `max_new_tokens=4`; it now uses 24 carried layers by default and generated `您好 {{--<../../../zego`.
- Probed the real Qwen runtime at 40 layers and at the full 48-layer stack; both short prompt runs completed successfully on this machine.
- Promoted the short-prompt default to the full carried stack for short prompt sessions instead of leaving the runtime below verified headroom.
- Runtime + desktop tests: 24/24 passed again after the full-stack default change.
- Verified the updated live default prompt runtime with `hello world` and `max_new_tokens=4`; it now uses all 48 carried layers by default and generated `Hello! How can`.
- 2026-04-27: Started the handoff-to-Codex workstream by running a full project health check. There is no `.git` repository in `C:\Users\isale\Documents\pcketlm`, but the Python suite passed cleanly at `74` tests before new work began.
- Added `pcketlm.core.model_families` as the first product-level model-family metadata layer, with priority order Qwen, Kimi, Kronos/Kronk, then Gemma. Qwen is marked `active`; the next families are marked `planned`.
- Updated model import to normalize known family aliases before writing registry records, so values like `qwen2`, `kronk`, and `gemma3` map to stable family keys.
- Connected the desktop status view to family labels and runtime support status instead of hardcoding Qwen in every product-facing path.
- Added a first desktop `Model Home` summary that shows the selected model as a flexible surface for chat, personalization, comparison, inspection, and benchmarking. This keeps the product direction aligned with direct model loading and non-linear user workflows.
- Added focused model-family tests and updated desktop status tests. Full verification now passes at `77` tests.
- Turned the desktop `Model Home` into clickable actions. Chat scrolls/focuses the existing local chat surface, Inspect opens the existing details view, and unfinished actions stay honest instead of pretending artifact flows exist.
- Added `pcketlm.core.benchmark.readiness` so the Benchmark action can report benchmark readiness, first planned checks, blockers, and recommended runtime path without running an expensive generation benchmark yet.
- Added built-in safe personalization profile templates under `pcketlm.core.profiles.templates`: Balanced Local, Agent Coder, and Low Memory. These are targets only; artifact generation is still not enabled.
- Changed desktop chat labels from prompt-test wording to product-facing chat wording.
- Full verification now passes at `82` tests after the model-home/profile/benchmark-readiness slice.
- Reproduced the user's suspected chat bug by running a real prompt smoke check. The Qwen runtime worked, but a 4-token `hello world` check took about two minutes, meaning the desktop appeared frozen because generation ran on the Tkinter UI thread.
- Fixed desktop chat execution by moving `run_prompt_decode_loop` into a background thread, disabling the Send button while a prompt is active, and showing a local generation working state.
- Re-ran the full test suite: `82` tests passed.
- Re-ran a real Qwen prompt smoke check with `hello world`, `max_new_tokens=2`, and repetition penalty `1.1`; it generated `Hello!`, reported `ready: true`, and had no blockers.
- Launched a fresh desktop app window after the fix so the user can test the updated chat behavior instead of the older frozen window.
- Lowered the desktop chat default from `8` to `2` max new tokens and added a visible alpha-runtime speed hint. This makes the current product easier to test while the real speed work is still unfinished.
- Re-ran unit tests: `83` passed.
- Re-ran the real Qwen smoke check after the default change. `hello world` with `max_new_tokens=2` generated `Hello!`, reported `ready: true`, and took about `67` seconds.
- Compared runtime layer budgets on the live Qwen smoke prompt. `8` layers took about `24` seconds but generated `findFirstOrCreate`; `16` layers took about `33` seconds but generated rough mixed text; `32` layers took about `48` seconds and generated Japanese-like text; full stack took about `67` seconds and generated the cleanest current `Hello!` result.
- Added a desktop runtime mode selector: Fast (`8` layers), Balanced (`32` layers), and Quality (`full stack`). Balanced is now the default for faster alpha testing, while Quality remains available for the best current output.
- Added mtime-aware caching for tensor catalog and tensor execution-plan JSON reads. This kept tests green but only slightly improved real runtime, confirming metadata reads are not the main bottleneck.
- Ran cProfile on the prompt path. For an `8`-layer, one-token smoke check, the dominant runtime costs were repeated `.float()` conversion of loaded tensors and `torch._C._nn.linear`; safetensors `get_tensor` itself was not the dominant cost.
- Re-ran unit tests: `84` passed.
- Re-ran the real Balanced-mode Qwen smoke check. `hello world` with `max_new_tokens=2`, `32` layers, and repetition penalty `1.1` completed in about `47` seconds, ready with no blockers, but quality drifted compared with full-stack mode.
- Added PyTorch `inference_mode` around the heavy prompt/layer/decode paths and cached small layer-bridge config JSON reads with mtime invalidation.
- Re-ran unit tests: `84` passed.
- Re-ran the real Balanced-mode Qwen smoke check after inference-mode/config caching. `hello world` with `max_new_tokens=2`, `32` layers, and repetition penalty `1.1` completed in about `44` seconds, ready with no blockers.
- Added persisted lightweight benchmark runs under each model's `benchmarks` directory and a `latest.lightweight-benchmark.json` snapshot. The desktop Benchmark action now saves this lightweight result before showing the summary.
- Re-ran unit tests: `85` passed.
- Re-ran the real Balanced-mode Qwen smoke check after benchmark persistence. The prompt path stayed ready with no blockers.
- Added `pcketlm.core.runtime.tensor_residency`, a memory-capped cache for converted CPU tensors with environment-tunable limits, LRU eviction, and runtime counters.
- Changed the layer bridge so `_load_required_tensor` uses the residency layer instead of directly returning `loaded.tensor.float()`. This keeps the generation math path stable while making conversion reuse tunable.
- Added focused tensor residency tests covering reuse, skip behavior for oversized tensors, and cache reset behavior.
- Re-ran the full unit suite after tensor residency: `88` tests passed.
- Re-ran the real Balanced-mode Qwen smoke check after tensor residency. `hello world` with `max_new_tokens=2`, `min_new_tokens=1`, `32` layers, and repetition penalty `1.1` completed in about `48` seconds, reported `ready: true`, and had no blockers.
- Added tensor residency stats to lightweight benchmark result JSON and the desktop Benchmark message, including hits, misses, resident tensors, resident MB, evictions, and skips.
- Re-ran the full unit suite after benchmark-stat wiring: `88` tests passed.
- Re-ran the real Balanced-mode Qwen smoke check after benchmark-stat wiring. `hello world` with `max_new_tokens=2`, `min_new_tokens=1`, `32` layers, and repetition penalty `1.1` completed in about `48` seconds, reported `ready: true`, and had no blockers.
- Measured the first broad tensor cache policy in-process on the real Balanced Qwen path. It completed in about `47` seconds but showed `0` cache hits, `770` misses, `450` stores, `406` evictions, and about `252 MB` resident tensors, proving the policy was safe but not useful.
- Changed tensor residency admission to keep only a front-layer warm window by default. The current defaults are `256 MB` total cache, `32 MB` per tensor, and `6` front layers, all tunable through environment variables.
- Measured the tuned front-layer policy in-process on the real Balanced Qwen path. It completed in about `46` seconds and showed `43` hits, `727` misses, `43` stores, `0` evictions, and about `252 MB` resident tensors.
- Tested a larger projection-cache variant with `1` front layer, `128 MB` per tensor, and `256 MB` total cache. It completed in about `46` seconds with fewer hits, so the safer six-front-layer default remains the chosen policy.
- Re-ran the full unit suite after front-layer cache tuning: `89` tests passed.
- Re-ran the real Balanced-mode Qwen CLI smoke check after front-layer cache tuning. `hello world` with `max_new_tokens=2`, `min_new_tokens=1`, `32` layers, and repetition penalty `1.1` completed in about `50` seconds, reported `ready: true`, and had no blockers.
- Measured decode-tail chunk sizes on the real Balanced Qwen CLI path. `8192` rows produced the same output and measured faster than `4096`, while `16384` was slower on this machine.
- Changed the default lm_head decode-tail chunk size from `4096` to `8192` rows and exported the default through the runtime package so the CLI and app share one value.
- Measured a real Qwen layer-0 projection and found bfloat16 linear math was far faster than float32 for that projection, with close mean output magnitude.
- Added `PCKETLM_RUNTIME_MATH_DTYPE` support, including `bfloat16` and `float32` modes, then verified the tiny runtime fixture in bfloat16 mode.
- Found and fixed a bfloat16 safety issue where same-dtype tensors could keep safetensors-backed storage after the shard handle closed. The residency loader now clones same-dtype converted tensors into owned CPU memory.
- Verified isolated real Qwen layer-0 and decode-tail execution in bfloat16 mode after the residency clone fix.
- Verified the real in-process Balanced Qwen prompt path in bfloat16 mode: it completed in about `36.5` seconds, returned the same generated token ids `[89015, 107162]`, and had no blockers.
- Promoted bfloat16 to the default runtime math dtype while keeping `PCKETLM_RUNTIME_MATH_DTYPE=float32` as an escape hatch.
- Re-ran the full unit suite after bfloat16 became the default: `91` tests passed.
- Re-ran the real Balanced-mode Qwen CLI smoke check with default bfloat16 math. `hello world` with `max_new_tokens=2`, `min_new_tokens=1`, `32` layers, and repetition penalty `1.1` completed in about `38.5` seconds, reported `ready: true`, and had no blockers.
- Added runtime setting visibility to lightweight benchmark result JSON and the desktop Benchmark message, including math dtype and lm_head chunk rows.
- Rechecked the Quality full-stack path after the bfloat16 speed change. `hello world` with `max_new_tokens=2`, `min_new_tokens=1`, `48` layers, and repetition penalty `1.1` completed in about `61` seconds, generated `Hello!`, and had no blockers.
- Changed the desktop chat default from Balanced to Quality because Balanced is faster but still drifts on the live Qwen smoke check, while Quality produces the clean current answer.
- Updated desktop runtime hints so the estimated time changes by mode: Fast is under 30 seconds, Balanced is about 40 seconds, and Quality is about 1 minute for the current 2-token default.
- Re-ran the full unit suite after the desktop default/mode-hint change: `91` tests passed.
- Re-ran the real default Quality-mode Qwen smoke check without forcing a layer count. `hello world` with `max_new_tokens=2`, `min_new_tokens=1`, and repetition penalty `1.1` completed in about `60` seconds, generated `Hello!`, and had no blockers.
- Added a measured benchmark backend that times Fast, Balanced, and Quality prompt runs, stores generated text, token ids, blockers, stop reason, runtime settings, and tensor residency stats, then writes `latest.measured-benchmark.json`.
- Wired the desktop Benchmark action to run the measured benchmark in a background thread and show a compact comparison when it completes.
- The first real measured benchmark exposed a readiness bug where plain CPU RAM warnings were treated as blockers even though the measured runtime path completed. Fixed that logic so completed measured cases can be ready while RAM limitations stay visible as warnings.
- Re-ran the full unit suite after measured benchmark wiring and readiness-status correction: `93` tests passed.
- Ran the real measured Qwen benchmark and saved it under `models/qwen2.5-14b-instruct/benchmarks`. Results: Fast `12.33s` -> `findFirstOrCreate`; Balanced `39.72s` -> `こんにちは世界的`; Quality `57.14s` -> `Hello!`; benchmark status `Measured benchmark ready`.
- Imported the new Pocket LLM design direction into the product by adding a local web UI shell with the same dark sidebar structure: Chat, Load Model, Personalize, Benchmarks, Agents, and Settings.
- Added `pcketlm.app.web.main`, a small stdlib HTTP server that serves the web assets and exposes local API routes for `/api/status`, `/api/chat`, and `/api/benchmark`.
- Repointed `launch_pcketlm.pyw` to launch the new local web UI instead of the older Tkinter desktop screen.
- Replaced the template's fake Claude/demo chat behavior with calls to the real Pocket LLM local runtime through `/api/chat`.
- Wired the web Benchmark screen to the real measured benchmark backend through `/api/benchmark`.
- Added focused web tests for mode mapping and prompt-result serialization.
- Re-ran the full unit suite after adding the web UI shell: `95` tests passed.
- Ran a local web-server smoke check. The new HTML served successfully, `/api/status` resolved `qwen2.5-14b-instruct`, the model list contained one model, and the profile list contained the three built-in templates.
- Ran a real Qwen smoke check through the new web API route. Quality mode generated `Hello!` from `hello world`, reported ready, and had no blockers.
- Added bounded recent-history support to `/api/chat`, including UI role normalization, placeholder filtering, and a compact transcript wrapper before the current user message.
- Updated the web chat UI so it sends the last few local turns, shows a live elapsed-time status while Qwen is running, and defaults to `4` max new tokens instead of `2`.
- Added focused web tests for chat history normalization and conversation prompt construction.
- Re-ran the full unit suite after conversation-history wiring: `98` tests passed.
- Ran a real Qwen web API smoke check with two previous conversation turns. The route returned HTTP `200`, `ready: true`, no blockers, `conversation_turn_count: 2`, and completed in about `59` seconds for `2` generated tokens. The tiny output was still mixed-language, so the next runtime concern is answer quality rather than the web history path.
- Switched web chat history from a plain transcript wrapper to real Qwen chat-message formatting when `<|im_start|>` and `<|im_end|>` are available locally, while keeping the transcript wrapper as fallback for other model families.
- Added a Pocket LLM web-system prompt so default web replies are instructed to stay English, brief, and direct.
- Tested the first formatted-history Quality check with `4` tokens. It was ready with no blockers, but still produced mixed-language text and an invented `defaultManager`, which showed the web formatting alone was not enough.
- Found that the automatic prompt budget was reducing chat-history prompts below the full stack once they passed `32` prompt tokens. Changed short-answer runs to keep the full stack for prompts under `256` tokens.
- Re-ran the full unit suite after the quality fix: `101` tests passed.
- Re-ran the real Qwen web memory check with two previous turns and `max_new_tokens=4`. The route returned HTTP `200`, `ready: true`, no blockers, `preformatted_chat: true`, `prompt_token_count: 61`, `48` cache layers, and generated `Your name is Sam` in about `99` seconds.
- Added a background chat-job layer for the web server. `/api/chat/start` creates a job, `/api/chat/status` polls it, and `/api/chat/cancel` marks it canceled while preserving `/api/chat` as the simple synchronous compatibility route.
- Updated the web chat UI to use start/status polling, show elapsed job state, expose Cancel during active generation, and expose Retry after a previous request.
- Re-ran the full unit suite after the background-job UI/API work: `103` tests passed.
- Ran a real Qwen background-job smoke check through `/api/chat/start` and `/api/chat/status`. The job completed with HTTP `202` on start, final status `completed`, `ready: true`, no blockers, and generated `Hello` from `hello world` in about `25` seconds for `1` token.
- Added cooperative cancellation hooks across the runtime path used by web chat: prompt decode loop checks, layer-stack per-layer checks, K/V decode-step checks, and streamed `lm_head` chunk checks in the decode tail.
- Passed the web job's cancel state into `run_prompt_decode_loop`, so a canceled job can stop inside the active runtime path instead of waiting until the full response is produced.
- Added focused cancellation coverage for prompt decode and web chat job payloads.
- Re-ran the full unit suite after runtime cancellation wiring: `104` tests passed.
- Ran a real Qwen cancel smoke check through `/api/chat/start` and `/api/chat/cancel`. The job reached final status `canceled`, with `cancel_requested: true`, no kept result, and about `4` seconds wall time.
- Re-ran a normal real Qwen background-job completion after the cancellation smoke. It completed with `ready: true`, no blockers, and generated `Hello` in about `25` seconds for `1` token.
- Cleaned up many duplicate `launch_pcketlm.pyw` processes that had accumulated from repeated relaunch checks. This removed resource pressure that was distorting runtime experiments.
- Tested unsafe large-tensor cache variants and rejected them after the process failed early under larger projection caching. The current weak-hardware cache policy remains conservative.
- Measured PyTorch CPU thread counts on the real Qwen Quality path. On the current 16-core machine, `14` threads gave the best observed 1-token result while preserving the same generated output.
- Added automatic runtime thread configuration with `PCKETLM_TORCH_THREADS` as an override. Auto mode keeps two cores free on larger machines and caps the default at `14` threads.
- Re-ran the real Qwen Quality `hello world` check after thread auto-configuration. The 2-token path generated `Hello!`, ready with no blockers, in about `57` seconds.
- Added a fixed web app port (`8765`) and single-instance lock port (`8764`) so repeated launches reuse the existing Pocket LLM server instead of accumulating duplicate local runtimes.
- Re-ran the full unit suite after the speed and singleton-launch work: `107` tests passed.
- Verified a double-launch check: one server process owned both the app port and lock port, and `/api/status` resolved `qwen2.5-14b-instruct`.
- Ran a live web-job Qwen smoke check against `http://127.0.0.1:8765`: Quality generated `Hello`, ready with no blockers, in about `25.6` seconds for `1` token.
- Added runtime phase timings to the prompt path and K/V continuation step, including stack totals, average layer time, and slowest layer time.
- Added a streamed top-k decode-tail path for policies that can select directly from top-k candidates, avoiding full logits materialization on greedy and top-k-sample paths.
- Re-ran focused runtime/web tests after the timing and top-k work: `37` tests passed.
- Re-ran the full unit suite after the timing and top-k work: `108` tests passed.
- Re-ran a clean real Qwen Quality timing check. `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `49.6` seconds. Timing showed prefill stack about `24.3s`, prefill decode tail about `1.0s`, continuation stack about `23.0s`, and continuation decode tail about `1.0s`.
- Restarted the updated web app on `http://127.0.0.1:8765` and verified the background-job route. Quality generated `Hello`, ready with no blockers.
- Added a safe full-stack speed pass: all-layer small-tensor residency, cached RoPE trig tables, math-dtype token entry, and optional layer-summary reductions so normal chat does less debug-only work.
- Re-ran focused runtime/web/tensor tests after the speed pass: `43` tests passed.
- Re-ran the full unit suite after the speed pass: `109` tests passed.
- Re-ran a clean real Qwen Quality timing check. `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `42.7` seconds. Timing showed prefill stack about `20.6s`, prefill decode tail about `0.9s`, continuation stack about `20.0s`, and continuation decode tail about `0.9s`.
- Restarted the updated web app on `http://127.0.0.1:8765` and verified the background-job route. Quality generated `Hello`, ready with no blockers, in about `21.0` seconds for `1` token.
- Added runtime visibility to the web UI: completed chat runs now expose timing/cache/runtime settings in a collapsible runtime details panel, and Settings shows the current runtime dtype, thread count, lm-head chunk size, and cache counters.
- Changed `/api/status` to include a direct-runtime status that can say `Working` when the Pocket direct runtime path is usable, even if the older full-RAM/Transformers preflight still warns.
- Re-ran the full unit suite after runtime visibility/status cleanup: `109` tests passed.
- Restarted the web app on `http://127.0.0.1:8765`. `/api/status` reported `qwen2.5-14b-instruct`, direct runtime `Working`, `bfloat16` math, `14` torch threads, and `8192` lm-head chunk rows.
- Ran a live web background-job smoke check after the cleanup. Quality generated `Hello`, ready with no blockers, in about `22.9` seconds for `1` token and returned timing/runtime settings data.
- Polished web chat cancellation so canceled jobs remove the temporary assistant placeholder from the transcript and leave Retry available.
- Upgraded the Benchmarks screen to show richer result cards with generated text, layer count, ready/blocked state, tensor-cache hits, resident cache size, and latest runtime settings.
- Reworked the Load Model screen to show active direct runtime status, the local model folder, and the current support plan for Qwen, Kimi, Kronos/Kronk, and Gemma.
- Re-ran the full unit suite after chat/benchmark/load product cleanup: `109` tests passed.
- Restarted the web app on `http://127.0.0.1:8765` and verified the new runtime/load/benchmark page assets are served.
- Ran a live cancel smoke check after the UI cleanup. The job reached final status `canceled`, kept no result, and had no error.
- Ran a live normal web background-job smoke check after the cancellation check. Quality generated `Hello`, ready with no blockers, in about `23.6` seconds for `1` token.
- Added persisted per-model profile records under `models/<model_id>/profiles`, seeded from the three free templates.
- Added profile comparison summaries that compare saved profile targets against the current default Quality baseline and latest benchmark cases when available.
- Updated `/api/status` to return saved profile records and the profile comparison payload.
- Added a Compare screen to the web UI and changed Personalize to render saved profile records instead of unsaved templates.
- Re-ran focused profile/web tests after saved profile and Compare wiring: `15` tests passed.
- Re-ran the full unit suite after saved profile and Compare wiring: `111` tests passed.
- Restarted the web app on `http://127.0.0.1:8765`; `/api/status` returned saved profile ids `balanced-local`, `agent-coder`, and `low-memory`, with `profile_compare.profile_count: 3`.
- Ran a live normal web background-job smoke check after the profile/Compare wiring. Quality generated `Hello`, ready with no blockers, in about `25.1` seconds for `1` token.
- Added operation-level timing inside the layer bridge so full-stack runs now report tensor-load time, norms, Q/K/V projection, RoPE, attention, output projection, MLP, and decode-tail costs.
- Added scaled-dot-product attention as the default attention implementation with manual attention as a fallback, plus cached causal masks for repeated decode shapes.
- Tested persistent safetensors handle caching against the real Qwen model. Unit tests passed, but the real process exited silently, so the cache was made opt-in through `PCKETLM_SAFETENSOR_HANDLE_CACHE=1` instead of default.
- Added safe batched tensor loading by shard and a batched residency path for cache misses.
- Wired the layer bridge to batch-load norms, Q/K/V tensors, and MLP projection weights. A measured attempt to also batch the output projection was slower, so it was reverted.
- Re-ran focused runtime/tensor tests after the heavy loading pass: `38` tests passed.
- Re-ran the full unit suite after the heavy loading pass: `113` tests passed.
- Re-ran a clean real Qwen Quality timing check. `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `40.5` seconds. Timing showed prefill stack about `19.9s`, continuation stack about `18.4s`, and each decode tail below `1.0s`.
- Restarted the web app on `http://127.0.0.1:8765` and verified `/api/status` still reports the direct Pocket runtime as working.
- Re-ran a live web background-job smoke check after the heavy loading pass. Quality generated `Hello`, ready with no blockers, in about `20.7` seconds for `1` token.
- Tested a larger front-layer residency window under the existing `256 MB` cap. `PCKETLM_TENSOR_CACHE_FRONT_LAYERS=12` kept resident cache at about `253 MB`, had no evictions, and moved the clean real Qwen Quality 2-token timing to about `35.3` seconds.
- Changed the default front-layer residency window from `6` to `12` while keeping the total cache cap unchanged.
- Added regression coverage for the default `12`-front-layer policy.
- Re-ran focused runtime/tensor tests after the residency-window pass: `39` tests passed.
- Re-ran a clean real Qwen Quality timing check after making the `12`-layer window default. `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `36.1` seconds. Cache stats showed `265` stores, `265` hits, `0` evictions, and about `253 MB` resident.
- Re-ran the full unit suite after the residency-window pass: `114` tests passed.
- Re-ran a longer real Qwen Quality check with `max_new_tokens=4`. It generated `Hello! How can`, completed all `4` steps, stayed ready with no blockers, and took about `85.0` seconds.
- Restarted the web app on `http://127.0.0.1:8765`; `/api/status` reported the direct Pocket runtime as working.
- Re-ran a live web background-job smoke check after the residency-window pass. Quality generated `Hello`, ready with no blockers, in about `26.9` seconds for `1` token. The web process reported very low free RAM at status time, so this smoke confirms stability more than best-case speed.
- Added a low-memory guard to tensor residency. When free RAM is below `2 GB`, the default policy now downgrades to `128 MB` resident cache and `6` front layers, while explicit `PCKETLM_TENSOR_CACHE_MB` and `PCKETLM_TENSOR_CACHE_FRONT_LAYERS` overrides are still honored.
- Added a short-lived memory-snapshot cache so the guard does not call the Windows memory API for every tensor load. A 1000-policy-check timing took about `0.005s` after the cache.
- Exposed the effective residency policy in web runtime settings as `tensor_residency_policy`, including max cache, front-layer count, free memory, and whether the guard is active.
- Re-ran focused runtime/web/tensor tests after the low-memory guard: `47` tests passed.
- Re-ran the full unit suite after the low-memory guard: `116` tests passed.
- Re-ran the standalone real Qwen Quality timing check after the guard. `hello world` with `max_new_tokens=2` generated `Hello!`, ready with no blockers, in about `41.5` seconds. Policy was not guarded on that run because free RAM was about `4.05 GB`; resident cache was about `253 MB`, with no evictions.
- Restarted the web app on `http://127.0.0.1:8765`; `/api/status` showed the direct runtime working and included the effective tensor residency policy.
- Re-ran a live web background-job smoke check after the guard pass. Quality generated `Hello`, ready with no blockers, in about `21.9` seconds for `1` token.
- Added `build_measured_benchmark_history`, which reads persisted measured benchmark JSON files and summarizes best, average, and worst timings for each mode.
- Added benchmark history to `/api/status` and rendered it on the Benchmarks screen.
- Added `get_saved_profile` and light saved-profile behavior in web chat. A chat request can now include `profile_id`; the backend can apply profile runtime mode and default max-new-token settings, and the result reports the active profile id/label.
- Added a profile picker to the web Chat composer. Selecting a profile updates mode/default token controls on the client and sends the profile id to the backend.
- Added cache policy and cache cap to runtime detail grids so users can see when the low-RAM guard changes behavior.
- Re-ran focused benchmark/profile/web tests after the medium groundwork pass: `20` tests passed.
- Re-ran the full unit suite after benchmark-history and profile behavior wiring: `118` tests passed.
- Re-ran two short real Qwen CLI prompt checks beyond `hello world`. Both were ready with no blockers, but 2-token outputs (`Pocket L`, `Certainly!`) confirmed that quality cannot be judged from tiny token budgets and should move to the high-effort decode-quality phase.
- Restarted the web app on `http://127.0.0.1:8765`; `/api/status` reported direct runtime `Working`, `2` benchmark history runs, `3` saved profiles, and memory guard inactive.
- Ran a web smoke through the `low-memory` profile without explicit mode/token fields. It completed ready with no blockers, used `profile_id: low-memory`, carried `32` layers, generated `3` tokens, and took about `47.6` seconds. Output was mixed-language/rough, so profile routing works but Balanced/Low-Memory quality needs the next high-effort decode pass.
- Ran the high-phase quality pass for the first Qwen path. The main finding was that Low Memory was customer-facing but still using the known-rough 32-layer Balanced preview path.
- Changed Low Memory defaults to Quality while preserving short responses and high memory priority, so weak-hardware users get the safer full-stack answer path instead of mixed-language partial-layer output.
- Added migration for saved built-in profiles: when a built-in default profile exists locally but its template defaults changed and no optimized artifact is ready, `ensure_default_profiles` refreshes the saved runtime/settings defaults.
- Added default Qwen chat stop markers to web chat requests and trimmed generated stop markers from visible assistant text in the runtime result.
- Re-ran focused runtime/profile/web tests after the high-phase quality pass: `44` tests passed.
- Re-ran the full unit suite after the high-phase quality pass: `120` tests passed.
- Restarted the web app on `http://127.0.0.1:8765`. Status reported direct runtime `Working`, Low Memory runtime mode `Quality`, and the saved profile comparison now maps Low Memory to the Quality baseline.
- Ran a 3-token real Low Memory web smoke after the profile change. It generated `Hello! How`, ready with no blockers, in about `57.6` seconds.
- Restarted again after the stop-marker trim and ran a shorter real Low Memory web smoke. It generated `Hello`, ready with no blockers, in about `26.6` seconds for `1` token.
- Updated the current Qwen heavy-engineering foundation estimate to about `52%`: profile quality routing and stop-marker cleanup are better, but speed, longer conversation quality, and real optimized profile artifacts remain unfinished.
- Ran the medium prep phase before the next high-speed phase. The live status was healthy: direct runtime `Working`, Low Memory mode `Quality`, and benchmark history had `2` measured runs.
- Added persisted timing summaries to measured benchmark cases. Fresh benchmark cases now include stack time, tensor-load time, decode-tail time, total time, and the current bottleneck alongside the raw runtime timings.
- Added timing-summary aggregation to benchmark history. Fresh history labels can now show average stack, average tensor-load, and average decode-tail time per mode.
- Updated the Benchmarks screen so fresh benchmark data displays timing chips in the app surface, not only in terminal/debug output.
- Re-ran focused benchmark and web tests after the timing-summary pass: `16` tests passed.
- Re-ran the full unit suite after the timing-summary pass: `120` tests passed.
- Ran `python -m compileall src tests -q` cleanly.
- Restarted the web app on `http://127.0.0.1:8765` and ran a real Low Memory web smoke. It generated `Hello`, ready with no blockers, in about `19.0` seconds for `1` token. The returned timings showed total about `19.0s` and prefill stack about `17.7s`, confirming the next high phase should target full-stack speed.
- Updated the current Qwen heavy-engineering foundation estimate to about `56%`: measurement and next-phase visibility are better, while the next high phase still needs actual speed gains.
- Ran the high-phase runtime identity/fidelity pass. The main user-visible issue was that Qwen did not know it was running inside Pocket LLM or which model was loaded unless that context was explicitly in the prompt.
- Added a runtime identity system prompt for web chat. It tells the model it is answering inside Pocket LLM, gives the current loaded model id/name, runtime mode, active profile, and tells it how to answer model-identity questions.
- Added `runtime_context` to web chat responses so the frontend/debug surface can confirm the actual model/profile/mode used.
- Added a small safe speed cleanup around metadata: Qwen chat-token support is cached per model id, and tokenizer/generation JSON reads now use an mtime-aware cache.
- Re-ran focused web/runtime tests after the identity pass: `40` tests passed.
- Re-ran the full unit suite after the identity pass: `121` tests passed.
- Ran `python -m compileall src tests -q` cleanly.
- Restarted the web app on `http://127.0.0.1:8765` and ran a model-identity smoke. It generated `Qwen2.5-14`, ready with no blockers, in about `184.8` seconds for `8` tokens, while `runtime_context.model_label` reported `Qwen2.5-14B-Instruct`.
- Ran a basic logic smoke. It generated `YES` for `If 2 plus 3 equals 5, answer YES only.`, ready with no blockers, in about `27.0` seconds for `1` token.
- Updated the current Qwen heavy-engineering foundation estimate to about `61%`: context/identity and basic logic are now proven, but speed remains the central blocker because 8 tokens took over three minutes.
- Started the Quality speed phase by inspecting the live runtime status and stack/tensor paths. Status was healthy: direct runtime `Working`, Low Memory mode `Quality`, and the resident cache held about `253 MB`.
- Added a local runtime-context answer path for clear model-identity questions. Pocket LLM now answers known app/runtime facts itself instead of sending them through a full Qwen decode.
- Added mtime-aware caching for the built layer bridge config object, reducing repeated config construction during full-stack layer passes.
- Added regression coverage proving model-identity questions do not call `run_prompt_decode_loop`.
- Re-ran focused web/runtime tests after the speed shortcut: `42` tests passed.
- Re-ran the full unit suite after the speed shortcut: `123` tests passed.
- Ran `python -m compileall src tests -q` cleanly.
- Restarted the web app on `http://127.0.0.1:8765`. `Which model do you run on?` now returned `Qwen2.5-14B-Instruct (qwen2.5-14b-instruct)` in about `0.015` seconds with strategy `local-runtime-context-answer`, instead of the previous Qwen path that took about `184.8` seconds for a partial identity answer.
- Re-ran the real Qwen logic path after the shortcut. `If 2 plus 3 equals 5, answer YES only.` generated `YES`, ready with no blockers, in about `29.0` seconds for `1` token, with prefill stack about `27.4s`.
- Updated the current Qwen heavy-engineering foundation estimate to about `64%`: deterministic runtime facts are now fast and reliable, but general Quality generation is still slow and needs deeper full-stack optimization.
- Continued the full-stack speed phase with a real timing probe. A normal Qwen logic request caused the web connection to close after about `93s`; checking status showed the local server process was gone.
- Restarted with the residency guard raised to `3 GB`, then tried the logic prompt while status reported about `1.21 GB` free RAM. The server still crashed, proving that very low RAM needs a hard pre-run block, not only a smaller residency cache.
- Added a hard web-chat memory guard: below `2 GB` free RAM, Pocket LLM now returns a clear `memory-guard` blocker instead of starting model generation.
- Added `PCKETLM_TENSOR_CACHE_LOW_MEMORY_GUARD_MB` so the conservative residency threshold can be overridden for advanced experiments, while defaulting to `3 GB`.
- Re-ran focused web/tensor/runtime tests after the hard memory guard: `56` tests passed.
- Re-ran the full unit suite after the hard memory guard: `127` tests passed.
- Ran `python -m compileall src tests -q` cleanly.
- Restarted the web app on `http://127.0.0.1:8765`. With about `0.74 GB` free RAM, `Which model do you run on?` still returned instantly through the local context path, and the real Qwen logic prompt returned a clean `memory-guard` blocker instead of crashing.
- Updated the current Qwen heavy-engineering foundation estimate to about `65%`: low-RAM stability improved, but real general-generation speed still requires enough free RAM to run benchmarks safely.
- Continued the Quality speed phase after freeing RAM. A fresh Low Memory/Quality logic smoke returned `YES` in about `24.6s`; timings showed tensor loading as the largest cost at about `17.7s`.
- Added an mtime-aware tensor-name index and wired it through tensor catalog lookup users, replacing repeated full-list scans in hot runtime paths.
- Re-ran focused runtime/tensor tests after the tensor-index pass: `19` tests passed.
- Re-ran the full unit suite after the tensor-index pass: `127` tests passed.
- Restarted the web app and re-ran the real logic smoke. The stable path remained correct (`YES`), with a warm repeat around `21.9s`, though cold timings remained noisy.
- Tested persistent safetensors handle caching as a default speed lever in batched tensor loading. It passed focused and full tests, but the real Qwen web request dropped the connection and killed the server after about `24s`, so the default was reverted to opt-in.
- Shortened the web runtime identity prompt to reduce repeated prompt prefill work while still telling Qwen the app, model, mode, profile, and model-identity answer.
- Re-ran focused web/runtime/tensor tests after the prompt-speed pass: `36` tests passed.
- Re-ran the full unit suite after the prompt-speed pass: `127` tests passed.
- Restarted the web app and re-ran the same Low Memory/Quality logic smoke twice. It returned `YES` with no blockers, prompt tokens dropped from `135` to `121`, cold-ish wall time was about `23.1s`, and warm wall time was about `22.1s`.
- Updated the current Qwen heavy-engineering foundation estimate to about `68%`: the app is safer and a little faster, but full-stack tensor loading is still the central speed blocker.
- Added a request-scoped safetensors handle cache as a safer alternative to persistent handle caching. Unit coverage proved it can reuse one shard handle across multiple load calls inside a scope.
- Wired the request-scoped handle cache into the prompt loop and re-ran focused tests: `50` tests passed, then the full suite passed with `128` tests.
- Real web validation showed the request-scoped handle cache still crashed the server, even with enough free RAM. It is now guarded behind `PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=1` and remains off by default.
- Raised the hard web-chat memory guard from `2 GB` to `3 GB` because a live run crashed around `2.8 GB` free RAM.
- Tightened the runtime identity prompt further to reduce repeated prefill cost. Focused tests passed (`50`), then the full suite passed (`128`).
- Restarted the web app and re-ran the same Low Memory/Quality logic smoke twice. It returned `YES` with no blockers, prompt tokens dropped to `76`, cold-ish wall time was about `22.8s`, and warm wall time was about `21.3s`.
- Updated the current Qwen heavy-engineering foundation estimate to about `70%`: prompt cost and crash resistance improved, while tensor loading remains the main speed bottleneck.
- Added cumulative tensor-load diagnostics to the runtime. Web status and chat results now expose tensor requests, loaded bytes, shard opens, and handle-reuse counters.
- Added lightweight conversation-state metadata to chat results so the product can distinguish today's bounded prompt replay from future true KV/session reuse.
- Added the first optimized-artifact planning module and tests. The manifest is planning-only by design: it creates a reversible artifact target without pretending a derived weight pack exists yet.
- Built the Qwen Low Memory runtime-pack plan. It wrote `models/qwen2.5-14b-instruct/artifacts/runtime-pack-plan.low-memory.artifact.json`, ready with `579` source tensors, `8` shards, and `48` layers.
- Re-ran focused diagnostics/artifact/web tests: `24` tests passed.
- Re-ran the full suite after the artifact pass: `129` tests passed.
- Restarted the app and confirmed status exposes the ready artifact plan and zeroed tensor-load counters on fresh launch.
- Ran a real Qwen Low Memory/Quality smoke while free RAM was marginal. It returned `YES`, ready with no blockers, but took about `97.7s`; diagnostics showed `577` tensors loaded, about `25.2 GB` moved, and `198` shard opens.
- Raised the hard web-chat generation guard from `3 GB` to `4 GB` because live evidence showed `3 GB` still allowed severe memory-pressure runs.
- Restarted the app again and verified the guard: at about `3.13 GB` free RAM, the same prompt returned `memory-guard` immediately instead of starting generation.
- Re-ran the full suite after the final guard change: `129` tests passed.
- Updated the current Qwen heavy-engineering foundation estimate to about `73%`: diagnostics, guard behavior, conversation-state groundwork, and artifact planning are stronger, but real speed still needs a derived runtime pack.
- Extended the optimized artifact layer from manifest-only to a real small-tensor safetensors pack. The builder clones tensor data out of the original safetensors handles before saving so the pack materialization stays stable.
- Added artifact-aware tensor loading with safe fallback. Packed tensors are loaded from the Pocket runtime pack when present; all missing tensors still load from the original source shards.
- Added diagnostics for artifact pack opens and artifact tensor hits.
- Added tests for pack materialization and artifact-backed tensor loading.
- Built the real Qwen Low Memory small pack. It contains `97` repeated small tensors, is about `0.95 MB`, and is recorded in `runtime-pack-plan.low-memory.artifact.json`.
- Re-ran focused runtime/artifact/web tests after artifact-aware loading: `26` tests passed.
- Re-ran the full suite after artifact-aware loading: `131` tests passed.
- Restarted the app and verified `/api/status` reports the artifact ready with `97` packed tensors and zeroed tensor-load counters.
- Ran the real Low Memory/Quality logic smoke through the artifact path. It returned `YES`, ready with no blockers, in about `22.3s`; diagnostics showed `97` artifact tensor hits and `148` original shard opens.
- Ran a warm repeat. It returned `YES`, ready with no blockers, in about `20.9s`; small tensors were resident by then, so the pack did not need to be hit again.
- Updated the current Qwen heavy-engineering foundation estimate to about `78%`: real artifact loading exists and is proven, but the remaining speed bottleneck is still the large projection tensors.
- Added Tier 2 artifact pack selection for front-layer Q/K/V projection tensors. The default Tier 2 target is the first `2` layers, bounded so the pack does not become a full-model duplicate.
- Added regression coverage for front-attention packing and verified that only the configured front layer is included in the fixture test.
- Built the real Qwen Tier 2 Low Memory pack. It now contains `109` tensors and is about `140.97 MB`, combining the small repeated tensors with front `2` layers of Q/K/V projection tensors.
- Re-ran focused artifact/runtime/web tests after Tier 2: `27` tests passed.
- Re-ran the full suite after Tier 2: `132` tests passed.
- Restarted the app and verified status reports direct runtime `Working`, artifact ready, `109` packed tensors, and about `140.97 MB` packed size.
- Live RAM was below the `4 GB` guard after restart, around `2.31 GB`, so a real generation speed run was correctly blocked. The Qwen logic prompt returned `memory-guard` instead of starting a pressure run.
- Updated the current Qwen heavy-engineering foundation estimate to about `81%`: Tier 2 artifact creation and loading support are in place, while speed proof requires enough RAM for a guarded run.
- Ran the next speed-core phase with enough RAM and built a larger `speed-core` runtime pack. It contains `125` tensors and is about `481.0 MB`.
- Added runtime-pack auto-selection so the app chooses a safe pack for current free RAM. The bigger pack is retained as an artifact, but not selected under the current weak-hardware budget.
- Added `speed_status` to web status so the app can report selected pack, shard opens, artifact hits, resident cache state, and loaded MB.
- Live validation rejected the bigger pack as a default: the same `OK` smoke slowed to about `54.9s` and free RAM fell to about `1.77 GB`.
- Re-tested larger adaptive residency and rolled it back after cache churn produced no useful warm-hit improvement.
- Final safe real web smoke returned `OK`, ready with no blockers, in about `37.4s` for `2` tokens, with the safer `140.97 MB` Low Memory pack selected.
- Updated the current Qwen heavy-engineering foundation estimate to about `83%`: the runtime is better guarded and instrumented, but real customer-grade speed needs a deeper execution/backend phase rather than simply larger safetensors packs.
- Added exact-result response reuse for chat requests. It keys on model, profile, mode, prompt, normalized messages, system prompt, generation settings, and stop strings.
- Added response-cache status to runtime settings and speed status.
- Added runtime engine decision diagnostics. Current status reports `direct-cpu` / `torch-cpu` because no supported GPU backend is visible to the current runtime.
- Live validation: after one real Queen `OK` run, the same `/api/chat` request returned from cache in about `0.454s` wall time with `response_reuse.hit: true`, strategy `+response-cache`, and model elapsed `0.0s`.
- Re-ran the full unit suite after the reuse/engine phase: `138` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `85%`: exact retries are fast and engine choice is explicit, but new general prompts still need true session-prefix/KV reuse or a different backend.
- Extended exact-result reuse to the `/api/chat/start` background-job flow. Cached results are now returned as already completed jobs, so the UI does not wait for the polling interval.
- Updated the web chat UI to finish immediately for already completed jobs and label reused output as cached.
- Live validation: one real Queen `OK` run filled the cache in about `37.83s`; the repeated `/api/chat/start` request returned a completed cached job in about `0.007s` with `job_fast_path: response-cache` and model elapsed `0.0s`.
- Re-ran the full unit suite after the instant cached-job phase: `139` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `86%`: product-path retries are instant, while new prompts still require true prefix/KV reuse or backend acceleration.
- Added `Quick` mode as a full-stack short-answer path: it keeps Quality's full layer count and caps output to `1` token.
- Request-scoped safetensors handle reuse is now automatic only for one-token runs; live two-token tests proved it must not be forced across continuation decode yet.
- Selected runtime-pack handles now participate in the scoped one-token path, reducing the live Quick smoke to `8` shard opens and `1` artifact-pack open.
- Live web API proof: Quick returned `OK`, ready with no blockers, in about `19.4s` wall time; Quality remained stable at about `36.1s` for `2` tokens.
- Re-ran the full unit suite after the Quick-speed phase: `140` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `88%`: the first fresh short-answer path is now in the 20-second target zone, but longer new prompts still need session-prefix reuse or backend work.
- Added the first real session-prefix KV reuse mechanism. Web chat now sends a session id, successful runs can store an in-memory KV prefix, and the next run can pass that prefix back into the prompt runtime.
- The prompt runtime now validates exact token-prefix matches before reuse and falls back to full prefill if model/profile/mode/layer path or token prefix does not match.
- Added a default append safety cap of `8` tokens because the current CPU append path processes suffix tokens one-by-one; longer suffixes can be slower than normal full prefill.
- Live proof after restart: first Quick run stored a `64` token reusable prefix; the follow-up matched the `64` token prefix but needed `14` append tokens, so Pocket LLM safely used full prefill and reported the reason.
- Re-ran the full unit suite after the session-prefix phase: `141` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `89%`: session-prefix machinery is real and guarded, but normal follow-up speed needs batched suffix append or a backend change before it becomes a big speed win.
- Replaced the one-token prefix append path with batched suffix prefill. The runtime now processes all matched follow-up suffix tokens in one layer-stack pass against the stored KV cache.
- Raised the default prefix append limit to `64` tokens for the batched path.
- Live proof after restart: first Quick prompt took about `20.7s`; the follow-up reused `64` prefix tokens, batch-appended `14` new prompt tokens, and returned `OK` in about `17.2s` with no blockers.
- The reused follow-up spent about `14.1s` in the batched prefix stack and about `1.1s` in decode tail; tensor loading remains the dominant wall.
- Re-ran the full unit suite after the batched-prefix phase: `141` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `91%`: follow-up reuse now creates a real speed win, while the next large gain needs reducing repeated weight movement or using a different backend.
- Added a selectable tensor residency boost. The safe default remains conservative, and users can choose the Boosted preset for `288 MB` and `13` front layers when they have more RAM.
- Rejected the larger `320 MB` / `15` layer residency experiment after live validation showed it made the follow-up path slower instead of faster.
- Restarted the web app and live-tested the adaptive path. First Quick `OK` took about `19.6s`; the follow-up reused `64` prefix tokens, batch-appended `14`, and returned `OK` in about `17.5s`.
- Added a Settings screen selector and saved runtime setting for the tensor cache preset, so the boost is opt-in instead of automatic.
- Re-ran the full unit suite after the selectable residency phase: `144` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `92%`: the policy is more adaptive, but future speed work needs a deeper weight-layout or backend step.
- Added a front K/V projection pack tier for the optional Boosted path.
- Built the real Qwen Boosted pack with `153` tensors and about `361.02 MB` of derived weights.
- Verified selector behavior after restart: Standard selects the `140.97 MB` Low Memory pack; Boosted selects the `361.02 MB` Boosted pack.
- Verified the rejected `481 MB` speed-core pack no longer becomes the normal high-RAM path.
- Ran live Quick smokes: Standard returned `OK` in about `18.5s`; Boosted returned `OK` in about `17.9s`.
- Added cache clearing when runtime presets change and included speed status in chat responses after real runs.
- Re-ran the full unit suite after the boosted-pack phase: `146` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `94%`: the optional boosted pack is real, but it is not a breakthrough speed layer.
- Added local backend capability reporting for Direct CPU, CUDA, DirectML, and llama.cpp/GGUF.
- Live backend report: Direct CPU is active; CUDA is not visible to Torch; DirectML package is missing; llama.cpp/GGUF is recommended but needs `llama-cpp-python` and a GGUF model file.
- Added backend report output to `/api/status` and the Settings screen.
- Re-ran the full unit suite after the backend-report phase: `148` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `95%`: the app now knows the next backend target, but the actual backend prototype requires an install/conversion decision.
- Added `gguf_backend.py` with local GGUF discovery, readiness status, and a prompt-runner interface.
- Added `gguf_sidecar_runner.py` for a future Python `3.12` llama.cpp sidecar.
- Created the sidecar environment under `state/backend-envs/gguf-py312`.
- Tried installing `llama-cpp-python` in the main Python `3.14` runtime with binary wheels only; no matching distribution was available.
- Tried installing through the documented prebuilt-wheel extra index in the Python `3.12` sidecar; pip downloaded source and failed at native build setup because `nmake`/C++ compiler are missing.
- Live status now reports the sidecar Python path and still correctly marks GGUF not ready.
- Re-ran the full unit suite after the GGUF adapter phase: `152` tests passed, and compile verification was clean.
- Updated the current Qwen heavy-engineering foundation estimate to about `96%`: implementation is ready to plug in a real GGUF runtime once the native package/binary and model artifact are available.
## Phase 2 / Step 6 / app download meter

- Added a Load Model download meter that reads `state/downloads/*.json` through `/api/status`.
- The meter shows model id, status, percent complete, downloaded GB / expected GB, expected file count, and last update time.
- The web UI now refreshes `/api/status` every 5 seconds while a download is active, so the user can watch progress while other engineering work continues.

Focused verification:

```text
tests/test_web_main.py::test_status_payload_includes_live_download_meter PASSED [ 92%]
============================= 28 passed in 7.26s ==============================
```

Compile verification:

```text
py -3.14 -m compileall src tests -q
exit code 0
```

Live download status while this was implemented:

```text
status: downloading
progress_pct: 2.03
bytes_on_disk_gb: 1.24
expected_bytes_gb: 61.04
present_expected_file_count: 3
expected_file_count: 24
```
## Phase 2 / Step 7 / run measurement helper

- Added `--measure-memory` to the layer-bridge CLI so the Qwen 32B first run can report elapsed seconds, free RAM at start/end, minimum free RAM, and peak RAM delta.
- This uses the existing `_memory_snapshot()` helper from `src/pcketlm/core/runtime/load_attempt.py`; no psutil or new dependency was added.

Focused verification:

```text
tests/test_runtime_layer_bridge_cli.py::test_runtime_layer_bridge_cli_measure_memory PASSED [100%]
============================== 1 passed in 1.43s ==============================
```

Compile verification:

```text
py -3.14 -m compileall src tests -q
exit code 0
```

Live download status while this was implemented:

```text
status: downloading
progress_pct: 3.54
bytes_on_disk_gb: 2.16
expected_bytes_gb: 61.04
present_expected_file_count: 3
expected_file_count: 24
```
## Phase 2 / Step 6

Qwen2.5-32B-Instruct download verification:

```text
download status: complete
progress_pct: 100.0
bytes_on_disk_gb: 61.04
expected_bytes_gb: 61.04
present_expected_file_count: 24
expected_file_count: 24
folder status: ready
present_shards: 17
expected_shards: 17
missing_core_files: []
total file bytes: 65539393631
```

Files in `models/qwen2.5-32b-instruct/original`:

```text
config.json 663
generation_config.json 242
merges.txt 1671839
model.safetensors.index.json 63248
model-00001-of-00017.safetensors 3916539832
model-00002-of-00017.safetensors 3900847496
model-00003-of-00017.safetensors 3900847480
model-00004-of-00017.safetensors 3900847544
model-00005-of-00017.safetensors 3900847544
model-00006-of-00017.safetensors 3900847544
model-00007-of-00017.safetensors 3900847544
model-00008-of-00017.safetensors 3900847544
model-00009-of-00017.safetensors 3900847544
model-00010-of-00017.safetensors 3900847544
model-00011-of-00017.safetensors 3900847544
model-00012-of-00017.safetensors 3900847544
model-00013-of-00017.safetensors 3900847544
model-00014-of-00017.safetensors 3900847544
model-00015-of-00017.safetensors 3900847544
model-00016-of-00017.safetensors 3900847544
model-00017-of-00017.safetensors 3098588976
tokenizer.json 7031645
tokenizer_config.json 7305
vocab.json 2776833
```
## Phase 2 / Step 7 / STOP

Qwen 32B first run was blocked by a native crash.

Command:

```text
py -3.14 -m pcketlm.app.chat_shell.runtime_layer_bridge_cli qwen2.5-32b-instruct --prompt "hello world" --policy greedy --max-new-tokens 1 --measure-memory
```

First corrected run after building the real 32B tensor catalog exited with empty stdout and empty stderr. A controlled retry with an exit-code wrapper produced:

```text
exit code: -1073741819
hex: 0xC0000005
meaning: Windows access violation
stdout length: 0
stderr length: 0
```

Real 32B tensor metadata before the run:

```text
catalog_ready: True
tensor_count: 771
shard_count: 17
layer_count: 64
catalog_blockers: []
plan_ready: True
unit_count: 195
plan_blockers: []
```

RAM and residency policy captured after the failed controlled run:

```text
free_memory_gb: 4.4
total_memory_gb: 15.31
max_resident_mb: 414.0
front_layer_count: 12
tensor_cache_preset: standard
memory_guard_active: false
model_aware_budget_active: true
```

32B config reference for the failed target:

```text
num_hidden_layers: 64
hidden_size: 5120
num_attention_heads: 40
num_key_value_heads: 8
intermediate_size: 27648
vocab_size: 152064
torch_dtype: bfloat16
```

Observed layer progress:

```text
No layer-progress result was emitted before the native crash. The target run was the full 64-layer prompt path; observed completed layer count is unknown / 0 emitted.
```

STOP reason:

```text
Native access violation during Qwen 32B first run. No more full-run retries in this session.
```
## Phase 2 / Recovery / Setup

```text
git checkout phase-2-page-runtime-generalization
fatal: Unable to create 'C:/Users/isale/Documents/pcketlm/.git/index.lock': File exists.

git branch --show-current
phase-2-page-runtime-generalization

stale lock recheck:
NO_LOCK

git branch --show-current
phase-2-page-runtime-generalization
```
## Phase 2 / Recovery / Step 2 / Qwen 14B baseline

14B baseline was required to pass before touching 32B. It did not pass end-to-end.

Environment note:

```text
Ollama runner was holding about 4439.5 MB working set.
Free RAM before stopping Ollama: 0.54 GB.
Free RAM after stopping Ollama: 4.83 GB.
```

Diagnostic instrumentation verification:

```text
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_load_config_outputs_checkpoints PASSED [100%]
============================== 1 passed in 2.78s ==============================
```

14B slice results:

```text
slice=load-config
exit=0
elapsed_seconds=0.003
free_ram_mb=4598
process_working_set_mb=203
ready=true
hidden_size=5120
num_hidden_layers=48

slice=embedding-only
exit=0
elapsed_seconds=0.043
free_ram_mb=2947
process_working_set_mb=206
ready=true
tensor_name=model.embed_tokens.weight
loaded_nbytes=1557135360
shape=[152064, 5120]

slice=embed-forward
exit=0
elapsed_seconds=4.523
free_ram_mb=3577
process_working_set_mb=398
ready=true
token_id=14990
shape=[1, 1, 5120]

slice=layer-0
exit=0
elapsed_seconds=3.495
free_ram_mb=3765
process_working_set_mb=474
ready=true
executed_layers=[0]
output_shape=[1, 1, 5120]

slice=layer-0-1
exit=0
elapsed_seconds=3.384
free_ram_mb=4096
process_working_set_mb=494
ready=true
executed_layers=[0, 1]
output_shape=[1, 1, 5120]

slice=layer-0-7
exit=0
elapsed_seconds=5.967
free_ram_mb=3508
process_working_set_mb=1020
ready=true
executed_layers=[0, 1, 2, 3, 4, 5, 6, 7]
output_shape=[1, 1, 5120]

slice=layer-0-15
exit=0
elapsed_seconds=11.511
free_ram_mb=3897
process_working_set_mb=1054
ready=true
executed_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_shape=[1, 1, 5120]

slice=full
exit=-1073741819
last_checkpoint=before full-prompt-decode
elapsed_seconds=0.001
free_ram_mb=4592
process_working_set_mb=203
stdout stopped before an after/complete event
stderr was empty
```

STOP:

```text
The 14B baseline failed at the full prompt path with Windows native access violation 0xC0000005.
Per the recovery brief, 32B recovery was not run because the diagnostic baseline did not pass end-to-end.
```
## Phase 2 / Recovery 2 / Setup

```text
git checkout phase-2-page-runtime-generalization
Already on 'phase-2-page-runtime-generalization'

git branch --show-current
phase-2-page-runtime-generalization
```
## Phase 2 / Recovery 2 / 14B Localization

14B finer slices after clearing local Ollama background processes:

```text
slice=all-layers
exit=0
elapsed_seconds=28.172
free_ram_start_mb=3045
free_ram_last_mb=2912
process_working_set_last_mb=936
executed_layers=[0..47]
output_shape=[1, 1, 5120]

slice=all-layers-norm
exit=0
elapsed_seconds=30.505
free_ram_last_mb=2140
process_working_set_last_mb=1057
executed_layers=[0..47]
normalized_shape=[1, 1, 5120]

slice=all-layers-norm-lm
exit=0
elapsed_seconds=29.969
free_ram_last_mb=1906
process_working_set_last_mb=936
executed_layers=[0..47]
lm_head_chunk_count=19
top_token_ids=[3837, 11, 1052, 5019, 284]

slice=full
exit=0
elapsed_seconds=22.917
free_ram_last_mb=8509
process_working_set_last_mb=517
generated_text="Hello"
```

14B prior baseline check:

```text
command: py -3.14 -m pcketlm.app.chat_shell.runtime_diagnose_cli --model qwen2.5-14b-instruct --slice full --max-new-tokens 4
exit=0
free_ram_start_mb=7360
free_ram_after_mb=5767
process_working_set_after_mb=1020
operation_seconds=69.759
generated_token_ids=[9707, 0, 2585, 646]
generated_text="Hello! How can"
timings.total=69.7585
timings.prefill_stack=17.094
timings.prefill_decode_tail=0.9317
timings.continuation_stack=48.5209
timings.continuation_decode_tail=2.8514
```

Recovery 2 localization result:

```text
The previous 14B native crash did not reproduce after clearing local background model pressure.
all-layers, final norm, lm_head, and full decode all passed.
No surgical runtime fix was applied because no code regression was reproducible in the localized slices.
```
## Phase 2 / Recovery 2 / Tests

```text
py -3.14 -m pytest tests/ -v
============================ 177 passed in 20.29s =============================

py -3.14 -m compileall src tests -q
exit code 0
```

## Phase 2 / Recovery 2 / 32B Localization

32B full retry:

```text
slice=full
exit=-1073741819
last_checkpoint=before full-prompt-decode
free_ram_mb=6062
process_working_set_mb=203
stderr=empty
```

32B progressive slices:

```text
slice=load-config
exit=0
elapsed_seconds=0.003
free_ram_mb=7117
process_working_set_mb=204
num_hidden_layers=64

slice=embedding-only
exit=0
elapsed_seconds=0.028
free_ram_mb=7117
process_working_set_mb=206
tensor_name=model.embed_tokens.weight
shape=[152064, 5120]
loaded_nbytes=1557135360

slice=embed-forward
exit=0
elapsed_seconds=7.538
free_ram_mb=6922
process_working_set_mb=400
token_id=14990
shape=[1, 1, 5120]

slice=layer-0
exit=0
elapsed_seconds=4.307
free_ram_mb=6512
process_working_set_mb=475
executed_layers=[0]

slice=layer-0-1
exit=0
elapsed_seconds=4.703
free_ram_mb=6476
process_working_set_mb=495
executed_layers=[0, 1]

slice=layer-0-7
exit=0
elapsed_seconds=9.702
free_ram_mb=5259
process_working_set_mb=1426
executed_layers=[0..7]

slice=layer-0-15
exit=0
elapsed_seconds=15.827
free_ram_mb=5550
process_working_set_mb=1459
executed_layers=[0..15]

slice=all-layers
exit=0
elapsed_seconds=49.281
free_ram_mb=5421
process_working_set_mb=1465
executed_layers=[0..63]

slice=all-layers-norm
exit=0
elapsed_seconds=49.272
free_ram_mb=5472
process_working_set_mb=1459
executed_layers=[0..63]
normalized_shape=[1, 1, 5120]

slice=all-layers-norm-lm
exit=0
elapsed_seconds=51.126
free_ram_mb=5896
process_working_set_mb=1460
executed_layers=[0..63]
lm_head_chunk_count=19
top_token_ids=[284, 600, 11, 358, 5019]
```

STOP:

```text
14B full is green and generated the baseline "Hello! How can".
32B passes all smaller slices but crashes only at full prompt decode.
This is a different failing slice than the recovered 14B path, so Recovery 2 stops here and leaves the 32B decode-loop/KV issue for the next session.
```

## Phase 2 / Recovery 3 / 32B full split

Setup:

```text
phase-2-page-runtime-generalization
```

Diagnostic test:

```text
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_load_config_outputs_checkpoints PASSED
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_full_honors_max_new_tokens PASSED
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_summarizes_kv_cache_bytes PASSED
3 passed in 1.83s
```

Qwen 32B prompt-prefill, prompt `hello world`, greedy, max_new_tokens=1:

```text
exit=0
start: free_ram_mb=5160, process_working_set_mb=203
prepare-prompt-and-generation after: free_ram_mb=5115, process_working_set_mb=241, prompt_token_count=31, num_hidden_layers=64
prompt-token-entry after: free_ram_mb=5113, process_working_set_mb=244, shape=[1, 31, 5120], dtype=torch.bfloat16
prompt-prefill-stack-with-kv after: elapsed_seconds=44.921, operation_seconds=44.645, free_ram_mb=4292, process_working_set_mb=1404
kv_cache_layers=64, kv_cache_total_bytes=8126464, kv_cache_total_mb=8, cache_sequence_lengths=31 for layers 0..63
```

Qwen 32B decode-step-1, prompt `hello world`, greedy, max_new_tokens=1:

```text
exit=0
start: free_ram_mb=5560, process_working_set_mb=203
prompt-prefill-stack-with-kv after: elapsed_seconds=44.518, operation_seconds=44.242, free_ram_mb=4967, process_working_set_mb=1406
prefill-decode-tail after: elapsed_seconds=45.836, operation_seconds=1.318, free_ram_mb=4961, process_working_set_mb=1403
first-token-selection after: elapsed_seconds=45.843, operation_seconds=0.007, free_ram_mb=4962, process_working_set_mb=1403
chosen_token_id=9707
kv_cache_layers=64, kv_cache_total_bytes=8126464, kv_cache_total_mb=8, cache_sequence_lengths=31 for layers 0..63
```

Qwen 32B decode-step-2, prompt `hello world`, greedy, max_new_tokens=2:

```text
exit=0
start: free_ram_mb=6094, process_working_set_mb=204
prompt-prefill-stack-with-kv after: elapsed_seconds=46.926, operation_seconds=46.639, free_ram_mb=5649, process_working_set_mb=1406
prefill-decode-tail after: elapsed_seconds=48.261, operation_seconds=1.335, free_ram_mb=5717, process_working_set_mb=1400
first-token-selection after: elapsed_seconds=48.268, operation_seconds=0.007, free_ram_mb=5719, process_working_set_mb=1401, chosen_token_id=9707
kv-decode-step-2 after: elapsed_seconds=94.357, operation_seconds=46.085, free_ram_mb=5857, process_working_set_mb=1413
chosen_token_id=4337
kv_cache_layers=64, kv_cache_total_bytes=8388608, kv_cache_total_mb=8, cache_sequence_lengths=32 for layers 0..63
```

Qwen 32B full confirmation, prompt `hello world`, greedy, max_new_tokens=1:

```text
start: free_ram_mb=7957, process_working_set_mb=204
before full-prompt-decode: free_ram_mb=7957, process_working_set_mb=204
exit=-1073741819 / 0xC0000005
```

STOP:

```text
The split path proves prompt prefill with KV, first-token tail/selection, and one true KV continuation step all pass on Qwen 32B.
The original full wrapper still crashes natively before it can emit an after full-prompt-decode checkpoint.
max_new_tokens=2 full was not run because max_new_tokens=1 already localized the native crash to the full decode-loop wrapper path.
```

## Phase 2 / Recovery 3 / scoped safetensor cache disabled check

Command:

```text
$env:PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE='0'
python -m pcketlm.app.chat_shell.runtime_diagnose_cli --model qwen2.5-32b-instruct --slice full --max-new-tokens 1
```

Result:

```text
exit=0
start: free_ram_mb=3888, process_working_set_mb=204
before full-prompt-decode: free_ram_mb=3888, process_working_set_mb=204
after full-prompt-decode: elapsed_seconds=40.319, operation_seconds=40.317, free_ram_mb=3617, process_working_set_mb=1410
generated_token_ids=[9707]
generated_text="Hello"
steps_completed=1
cache_sequence_lengths=31 for layers 0..63
prefill_stack_total=38.9542
prefill_decode_tail=1.0183
```

Updated localization:

```text
Disabling PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE made the same 32B full path pass.
The native crash correlates with scoped safetensor handle caching, not the full decode-loop math itself.
```

## Phase 2 / Recovery 4 / safe scoped-cache default

Change:

```text
Default scoped safetensor handle reuse is now disabled for qwen2.5-32b-instruct.
Explicit PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE=1 still opts into the old scoped-handle path.
Qwen 14B one-token auto behavior remains unchanged.
```

Focused tests:

```text
tests/test_runtime_layer_bridge.py::test_scoped_safetensor_handles_default_off_for_qwen_32b PASSED
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_load_config_outputs_checkpoints PASSED
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_full_honors_max_new_tokens PASSED
tests/test_runtime_diagnose_cli.py::test_runtime_diagnose_cli_summarizes_kv_cache_bytes PASSED
4 passed in 2.42s
```

Qwen 32B full, default environment, prompt `hello world`, greedy, max_new_tokens=1:

```text
exit=0
start: free_ram_mb=3636, process_working_set_mb=203
after full-prompt-decode: elapsed_seconds=44.502, operation_seconds=44.501, free_ram_mb=3455, process_working_set_mb=1410
generated_text="Hello"
generated_token_ids=[9707]
steps_completed=1
cache_sequence_lengths=31 for layers 0..63
prefill_stack_total=43.027
prefill_decode_tail=1.1513
```

Qwen 32B full, default environment, prompt `hello world`, greedy, max_new_tokens=2:

```text
exit=0
start: free_ram_mb=4463, process_working_set_mb=204
after full-prompt-decode: elapsed_seconds=77.281, operation_seconds=77.28, free_ram_mb=4400, process_working_set_mb=1418
generated_text="Hello World"
generated_token_ids=[9707, 4337]
steps_completed=2
cache_sequence_lengths=32 for layers 0..63
prefill_stack_total=37.9764
continuation_stack_total=36.9173
```

Qwen 14B regression, default environment, prompt `hello world`, greedy, max_new_tokens=4:

```text
exit=0
start: free_ram_mb=5269, process_working_set_mb=203
after full-prompt-decode: elapsed_seconds=69.39, operation_seconds=69.39, free_ram_mb=4949, process_working_set_mb=1022
generated_text="Hello! How can"
generated_token_ids=[9707, 0, 2585, 646]
steps_completed=4
cache_sequence_lengths=34 for layers 0..47
```

## Phase 2 / Recovery 5 / 32B four-token stability proof

Self-prompt:

```text
Prove the default-safe 32B path beyond two tokens without changing code: run max_new_tokens=4, record timing/RAM/output, and commit evidence if it passes.
```

Qwen 32B full, default environment, prompt `hello world`, greedy, max_new_tokens=4:

```text
exit=0
start: free_ram_mb=5411, process_working_set_mb=204
after full-prompt-decode: elapsed_seconds=156.6, operation_seconds=156.6, free_ram_mb=5305, process_working_set_mb=1278
generated_text="Hello World! It"
generated_token_ids=[9707, 4337, 0, 1084]
steps_completed=4
cache_sequence_lengths=34 for layers 0..63
prefill_stack_total=37.9279
prefill_decode_tail=1.0355
continuation_stack_total=114.1655
continuation_steps=117.3197
```

## Phase 2 / Recovery 6 / 32B customer guardrails

Self-prompt:

```text
Make customer-facing 32B guardrails, so the app clearly knows when 32B is stable, slow, or too risky instead of treating it like normal chat.
```

Change:

```text
Added a direct model guardrail payload for Qwen 32B.
It reports RAM floor, recommended free RAM, proven token length, scoped safetensor handle-cache safety, status, warnings, and blockers.
The payload is now included in status, speed status, direct chat responses, GGUF chat responses, local identity answers, and memory-guard responses.
The Load Model panel now renders the guardrail status and key metrics.
```

Focused test evidence:

```text
tests/test_web_main.py::test_direct_model_guardrails_marks_qwen_32b_as_stable_slow PASSED
tests/test_web_main.py::test_direct_model_guardrails_warns_on_unproven_qwen_32b_length PASSED
tests/test_web_main.py::test_direct_model_guardrails_blocks_qwen_32b_below_ram_floor PASSED
tests/test_web_main.py::test_run_chat_payload_includes_qwen_32b_guardrails PASSED
4 passed in 2.42s
```

Web regression evidence:

```text
tests/test_web_main.py
32 passed in 7.11s
```

## Phase 2 / Recovery 7 / 32B proven-token policy

Self-prompt:

```text
Turn the 32B proven-token guardrail into actual web-chat policy: keep normal 32B direct chat inside the proven 4-token range unless the request explicitly opts into experimental longer output, and prove it with tests.
```

Change:

```text
Normal direct web chat for qwen2.5-32b-instruct is capped to 4 new tokens.
Requests can opt into longer experimental 32B output with allow_experimental_32b_tokens=true.
Experimental longer requests keep the guardrail warning in the response payload.
```

Focused test evidence:

```text
tests/test_web_main.py::test_run_chat_payload_caps_qwen_32b_to_proven_token_range PASSED
tests/test_web_main.py::test_run_chat_payload_allows_explicit_experimental_qwen_32b_length PASSED
2 passed in 1.44s
```

Web regression evidence:

```text
tests/test_web_main.py
34 passed in 7.03s
```

## Phase 2 / Recovery 8 / full suite after 32B policy

Self-prompt:

```text
Before moving deeper, run the full test suite because this branch now touches runtime defaults and web product policy.
```

Full suite evidence:

```text
py -3.14 -m pytest tests/ -v
185 passed in 18.90s
```

## Phase 2 / Recovery 9 / guardrail UI smoke

Self-prompt:

```text
Since the Load Model UI changed, run a lightweight frontend syntax check and a live guardrail payload smoke without touching the model.
```

Evidence:

```text
node --check src/pcketlm/app/web/static/app.js
exit=0

_direct_model_guardrails("qwen2.5-32b-instruct", 4)
status=stable-low-headroom
proven_max_new_tokens=4
scoped_safetensor_handle_cache.default_enabled=False
```

## Phase 2 / Recovery 10 / experimental 8-token preflight

Self-prompt:

```text
Before trying any longer 32B proof, check current RAM against the new guardrail. If the machine is below the recommended 5 GB headroom, do not start a 5-minute experimental model run; record the blocker instead.
```

Preflight evidence:

```text
model_id=qwen2.5-32b-instruct
requested_max_new_tokens=8
status=stable-low-headroom
ready=True
free_ram_mb=4830
min_free_ram_mb=4096
recommended_free_ram_mb=5120
warning=Qwen 32B is proven to 4 new tokens on this machine; longer runs are experimental.
decision=do not run experimental 8-token live proof until free RAM is at least 5120 MB
```

## Phase 2 / Recovery 11 / Qwen 32B eight-token proof

Self-prompt:

```text
After freeing enough RAM, run the real Qwen 32B full prompt/decode path for 8 new tokens. If it passes, promote the product guardrail from 4 to 8; if it crashes, localize the failure before changing policy.
```

Live 32B evidence:

```text
py -3.14 -m pcketlm.app.chat_shell.runtime_diagnose_cli --model qwen2.5-32b-instruct --slice full --max-new-tokens 8
exit=0
free_ram_start_mb=9200
free_ram_end_mb=8117
process_working_set_end_mb=1421
operation_seconds=330.955
generated_text=Hello World! It's great to see
generated_token_ids=[9707, 4337, 0, 1084, 594, 2244, 311, 1490]
cache_sequence_length_each_layer=38
```

Focused verification:

```text
py -3.14 -m pytest tests/test_web_main.py tests/test_runtime_layer_bridge.py::test_scoped_safetensor_handles_default_off_for_qwen_32b -v
35 passed in 8.21s

node --check src/pcketlm/app/web/static/app.js
exit=0
```

Full verification:

```text
py -3.14 -m pytest tests/ -v
185 passed in 20.00s

py -3.14 -m compileall -q src tests
exit=0
```

## Phase 3C / GGUF speed proof

Discovery:

```text
GGUF artifact exists:
models/qwen2.5-14b-instruct/artifacts/qwen2.5-14b-instruct-q4_k_m.gguf
size=8.37 GB

llama.cpp binaries exist:
state/backend-runtimes/llama.cpp/llama-cli.exe
state/backend-runtimes/llama.cpp/llama-server.exe
```

Backend readiness:

```text
ready=true
main_package_available=false
sidecar_package_available=false
llama_cli_available=true
llama_server_available=false before load
blockers=[]
```

Cold GGUF call:

```text
prompt=Reply OK only.
max_tokens=4
ready=true
backend=llama-cpp-gguf-server
generated_text=No need to explain
elapsed_seconds=52.26
server predicted_n=4
server predicted_per_second=2.4543
```

Warm GGUF server call:

```text
server before call:
running=true
pid=11312
working_set_gb=8.64

prompt=Reply OK only.
max_tokens=4
ready=true
backend=llama-cpp-gguf-server
generated_text=No need to say
elapsed_seconds=2.36
server predicted_n=4
server predicted_per_second=2.2966
```

Cleanup:

```text
stop_gguf_server('qwen2.5-14b-instruct')
running=false
pid=null
working_set_gb=null
```

Verdict:

```text
This is the first real speed jump. Direct CPU page runtime remains important for the dense-model research path, but customer-speed chat/agent work should route through loaded GGUF when the artifact is available and enough RAM exists.
```

## Phase 3C / GGUF recommendation routing

Change:

```text
Backend report now marks llama.cpp/GGUF as ready and implemented when the local GGUF adapter is ready.
Engine decision keeps Direct CPU as the dense fallback but recommends llama-cpp-gguf for faster chat.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_runtime_engine_selector.py tests/test_web_main.py -v
45 passed in 12.53s

py -3.14 -m compileall -q src\pcketlm\core\runtime\engine_selector.py tests\test_runtime_engine_selector.py
exit=0
```

Live selector smoke:

```text
build_runtime_backend_report('qwen2.5-14b-instruct').recommended_backend_id
llama-cpp-gguf

select_runtime_engine('qwen2.5-14b-instruct'):
selected_engine=direct-cpu
selected_backend=torch-cpu
recommended_backend_id=llama-cpp-gguf
summary=GGUF is ready and recommended for faster chat; Direct CPU remains the dense fallback and research path.
```

Full verification:

```text
py -3.14 -m pytest tests/ -v
197 passed in 21.55s

py -3.14 -m compileall -q src tests
exit=0
```

## Phase 3B / Warm Agent controls

Change:

```text
Added /api/warm-runner with start, stop, and status actions.
Settings now has Start Agent runner and Stop buttons.
Stopping the runner clears session-prefix and exact-response caches.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_web_main.py tests/test_warm_runner.py -v
45 passed in 8.55s

node --check src\pcketlm\app\web\static\app.js
exit=0

py -3.14 -m compileall -q src\pcketlm\app\web\main.py tests\test_web_main.py
exit=0
```

Live control smoke:

```text
py -3.14 -c "from pcketlm.app.web.main import _warm_runner_control_payload; print(_warm_runner_control_payload({'action':'start','model_id':'qwen2.5-14b-instruct'})['warm_runner']['state']); print(_warm_runner_control_payload({'action':'stop','model_id':'qwen2.5-14b-instruct'})['warm_runner']['state'])"
ready
stopped
```

Full verification:

```text
py -3.14 -m pytest tests/ -v
196 passed in 21.40s

py -3.14 -m compileall -q src tests
exit=0
```

## Phase 3B / Warm Agent status surface

Change:

```text
/api/status now includes warm_runner.
Settings runtime grid shows Agent runner state, request count, last run, prefix readiness, and process working set.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_web_main.py tests/test_warm_runner.py -v
44 passed in 8.57s

node --check src\pcketlm\app\web\static\app.js
exit=0

py -3.14 -m compileall -q src\pcketlm\app\web\main.py tests\test_web_main.py
exit=0
```

Live status smoke:

```text
py -3.14 -c "from pcketlm.app.web.main import _status_payload; p=_status_payload(); print({'warm_state': p['warm_runner']['state'], 'requests': p['warm_runner'].get('request_count'), 'agent_setting': p['runtime_settings'].get('agent_warm_runner')})"
{'warm_state': 'ready', 'requests': 2, 'agent_setting': 'safe'}
```

Full verification:

```text
py -3.14 -m pytest tests/ -v
195 passed in 21.42s

py -3.14 -m compileall -q src tests
exit=0
```

Qwen 14B regression:

```text
py -3.14 -m pcketlm.app.chat_shell.runtime_diagnose_cli --model qwen2.5-14b-instruct --slice full --max-new-tokens 4
exit=0
free_ram_start_mb=9009
free_ram_end_mb=8269
process_working_set_end_mb=1020
operation_seconds=74.428
generated_text=Hello! How can
generated_token_ids=[9707, 0, 2585, 646]
```

## Phase 3 / Setup

```text
branch=phase-3-speed-reliability-productization
starting_point=8f2c8f4 phase2/recovery11: prove 32b eight token path
untracked_left_untouched=MEMORY.md, project_pcketlm.md
```

## Phase 3 / Baseline benchmark

Self-prompt:

```text
Before speed-policy changes, measure the real 14B and 32B direct full prompt/decode path at 1, 4, and 8 new tokens. Use these numbers to choose policy instead of guessing.
```

Baseline evidence:

```text
qwen2.5-14b-instruct / 1 token
exit=0
operation_seconds=19.494
free_ram_start_mb=8999
free_ram_end_mb=11086
working_set_mb=577
generated_text=Hello
generated_token_ids=[9707]
prefill_stack_seconds=16.765
tensor_load_seconds=14.110

qwen2.5-14b-instruct / 4 tokens
exit=0
operation_seconds=77.568
free_ram_start_mb=10956
free_ram_end_mb=9809
working_set_mb=1022
generated_text=Hello! How can
generated_token_ids=[9707, 0, 2585, 646]
prefill_stack_seconds=19.023
continuation_steps_seconds=57.134
tensor_load_seconds=65.046

qwen2.5-14b-instruct / 8 tokens
exit=0
operation_seconds=155.456
free_ram_start_mb=10573
free_ram_end_mb=9048
working_set_mb=1017
generated_text=Hello! How can I assist you today
generated_token_ids=[9707, 0, 2585, 646, 358, 7789, 498, 3351]
prefill_stack_seconds=19.182
continuation_steps_seconds=134.853
tensor_load_seconds=131.068

qwen2.5-32b-instruct / 1 token
exit=0
operation_seconds=48.254
free_ram_start_mb=9769
free_ram_end_mb=8119
working_set_mb=1410
generated_text=Hello
generated_token_ids=[9707]
prefill_stack_seconds=46.720
tensor_load_seconds=40.870

qwen2.5-32b-instruct / 4 tokens
exit=0
operation_seconds=180.573
free_ram_start_mb=9278
free_ram_end_mb=8406
working_set_mb=1318
generated_text=Hello World! It
generated_token_ids=[9707, 4337, 0, 1084]
prefill_stack_seconds=46.735
continuation_steps_seconds=132.351
tensor_load_seconds=156.956

qwen2.5-32b-instruct / 8 tokens
exit=0
operation_seconds=346.482
free_ram_start_mb=9580
free_ram_end_mb=8121
working_set_mb=1437
generated_text=Hello World! It's great to see
generated_token_ids=[9707, 4337, 0, 1084, 594, 2244, 311, 1490]
prefill_stack_seconds=43.931
continuation_steps_seconds=301.021
tensor_load_seconds=301.708
```

Bottleneck verdict:

```text
The dominant cost is repeated tensor loading, not prompt formatting or decode tail.
14B 8-token tensor loading: about 131.1s of 155.5s.
32B 8-token tensor loading: about 301.7s of 346.5s.
```

## Phase 3 / Speed policy and Agent mode

Change:

```text
Added measured direct-runtime baselines for Qwen 14B and Qwen 32B.
Qwen 14B and 32B normal direct web chat now cap at the proven 8-token local range unless allow_experimental_direct_tokens=true.
Added Agent mode, which uses the full stack but caps direct replies to 2 new tokens for repeated local work.
Benchmarks now include Agent between Quick and Fast.
Guardrails now expose baseline_estimate, bottleneck, recommended_mode, and token policy.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_benchmark_runs.py tests/test_web_main.py -v
43 passed in 7.73s

node --check src/pcketlm/app/web/static/app.js
exit=0
```

Live Agent smoke:

```text
_run_chat_payload({model_id=qwen2.5-14b-instruct, mode=Agent, max_new_tokens=8, prompt=hello world})
ready=True
elapsed_seconds=40.16
generated_text=Hello!
max_new_tokens=2
steps_completed=2
generated_token_ids=[9707, 0]
guard_status=stable-slow
guard_estimate_seconds=39.0
guard_bottleneck=tensor loading
```

## Phase 3 / Agent reuse and performance summary

Self-prompt:

```text
Measure repeated Agent calls in one web-runtime process. If prefix reuse works but speed barely improves, do not fake a speed win; expose the bottleneck structurally in response payloads.
```

Live repeated Agent evidence:

```text
first Agent call:
ready=True
elapsed_seconds=40.58
generated_text=Hello!
prefix_reuse.used=False

second Agent call, same session with chat history:
ready=True
elapsed_seconds=38.72
generated_text=Ok
prefix_reuse.used=True
matched_token_count=64
appended_token_count=14
prefix_append_stack_op_load_tensors=16.3822
continuation_stack_op_load_tensors=16.2546
```

Verdict:

```text
Session-prefix reuse works, but Agent calls are still dominated by repeated tensor loading.
Boosted was not promoted automatically because the live Agent run stayed around 40.72s and reported the standard saved preset.
Response payloads now include performance_summary so every future live run records stack seconds, tensor-load seconds, decode-tail seconds, tensor-load share, and bottleneck.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_web_main.py tests/test_benchmark_runs.py -v
43 passed in 8.34s

node --check src/pcketlm/app/web/static/app.js
exit=0
```

## Phase 3 / 32B twelve-token ladder

Self-prompt:

```text
Try the next 32B proof-ladder step, but do not promote it if quality or layer coverage is not release-grade.
```

Preflight:

```text
requested_max_new_tokens=12
free_ram_mb=8943
estimated_seconds=519.7
warning=Qwen2.5-32B-Instruct is proven to 8 new tokens on this machine; longer runs are experimental.
```

Live diagnostic evidence:

```text
py -3.14 -m pcketlm.app.chat_shell.runtime_diagnose_cli --model qwen2.5-32b-instruct --slice full --max-new-tokens 12
exit=0
operation_seconds=202.704
free_ram_start_mb=8907
free_ram_end_mb=7996
working_set_mb=1415
generated_text=您好战росл无论是其ПетерLLU SQETCHing查看全文长长长长
generated_token_ids=[111308, 119921, 137360, 117918, 135873, 45110, 51618, 15836, 287, 118214, 117012, 117012]
prefill_stack_layer_count=24
continuation_stack_layer_count=264
```

Verdict:

```text
This is not promoted. The process did not crash, but the auto layer budget dropped to 24 layers above 8 tokens and output quality degraded badly.
```

Fix:

```text
Explicit experimental direct web runs above the proven token range now force the model's full layer count instead of silently using the rough reduced-layer path.
Normal web chat still caps Qwen 14B and 32B to 8 new tokens.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_web_main.py -v
39 passed in 7.77s
```

## Phase 3 / Persistent handle cache rejection

Self-prompt:

```text
Before designing a warm runner from scratch, test the existing persistent safetensor handle cache on 14B Agent reuse. Promote it only if the second call is both faster and stable.
```

Live evidence:

```text
PCKETLM_SAFETENSOR_HANDLE_CACHE=1
first Agent call:
ready=True
elapsed_seconds=38.7
generated_text=Hello!

second Agent call:
ready=False
elapsed_seconds=0.0

load_stats:
shard_opens=8
artifact_pack_opens=65
artifact_tensor_hits=145
persistent_handle_reuses=284
loaded_mb=50402.0
```

Verdict:

```text
Rejected as a default. Persistent handle reuse reduced shard opens, but the two-call Agent path was not stable enough to promote.
```

## Phase 3B / Conservative warm Agent runner

Self-prompt:

```text
Build a controlled warm runner for short Agent work, keep it opt-in, avoid persistent safetensor handle caching, surface state in the web app, and prove it with a real Qwen 14B two-call run.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_warm_runner.py tests/test_web_main.py -v
43 passed in 7.68s

node --check src\pcketlm\app\web\static\app.js
exit=0

py -3.14 -m compileall -q src\pcketlm\core\runtime\warm_runner.py src\pcketlm\app\chat_shell\warm_runner_cli.py src\pcketlm\app\web\main.py
exit=0
```

Live 14B warm proof:

```text
py -3.14 -m pcketlm.app.chat_shell.warm_runner_cli sequence --model qwen2.5-14b-instruct --prompt "hello world" --second-prompt "hello world" --max-new-tokens 2

start:
free_ram_mb=5350
process_working_set_mb=203

first:
ready=true
generated_text=Hello!
generated_token_ids=[9707, 0]
elapsed_seconds=40.274
free_ram_before_mb=5350
free_ram_after_mb=5315
process_working_set_before_mb=203
process_working_set_after_mb=1003
tensor_load_seconds=33.503
tensor_load_share=0.83
bottleneck=layer stack
reusable_token_count=32

second:
ready=true
generated_text=Hello!
generated_token_ids=[9707, 0]
elapsed_seconds=39.177
free_ram_before_mb=5315
free_ram_after_mb=5541
process_working_set_before_mb=1003
process_working_set_after_mb=1009
tensor_load_seconds=32.875
tensor_load_share=0.84
bottleneck=layer stack
prefix_reuse.enabled=true
prefix_reuse.used=false
prefix_reuse.summary=Supplied prefix did not safely match this prompt; full prefill was used.
```

Verdict:

```text
Promoted as an opt-in lifecycle/product path, not as a speed win. The runner keeps Agent state and telemetry alive safely, but the measured bottleneck remains repeated tensor/layer work.
```

Full verification:

```text
py -3.14 -m pytest tests/ -v
194 passed in 19.97s

py -3.14 -m compileall -q src tests
exit=0
```

## Phase 3B / Warm Agent web prefix reuse fix

Problem found:

```text
The first opt-in web Agent prompt was formatted for Qwen, but the warm runner still let the runtime apply chat formatting again.
That double-wrapping prevented safe prefix reuse on the second web-style turn.
```

Fix:

```text
Warm runner accepts apply_chat_format.
Web Agent warm path sends apply_chat_format=false after it prepares the Qwen chat prompt.
First warm Agent turn now uses the Qwen chat frame even when there is no prior message history.
```

Focused verification:

```text
py -3.14 -m pytest tests/test_web_main.py tests/test_warm_runner.py -v
44 passed in 8.61s

py -3.14 -m compileall -q src\pcketlm\core\runtime\warm_runner.py src\pcketlm\app\web\main.py tests\test_web_main.py tests\test_warm_runner.py
exit=0
```

Live web-style two-turn proof:

```text
first:
ready=true
generated_text=Hello!
elapsed_seconds=51.69
preformatted_chat=true
reusable_token_count=64
free_ram_after_mb=6790
process_working_set_after_mb=1051

second:
ready=true
generated_text=Ok<|im_end|>
elapsed_seconds=47.06
prefix_reuse.enabled=true
prefix_reuse.used=true
prefix_reuse.matched_token_count=64
prefix_reuse.appended_token_count=14
prefix_reuse.summary=Reused 64 prompt tokens and batch-appended 14 new prompt tokens.
reusable_token_count=79
free_ram_after_mb=6779
process_working_set_after_mb=1069
```

Full verification:

```text
py -3.14 -m pytest tests/ -v
195 passed in 21.39s

py -3.14 -m compileall -q src tests
exit=0
```
