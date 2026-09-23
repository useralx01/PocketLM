# pcketlm Risk Register

This file exists to keep likely failure points visible before they turn into expensive surprises.

## How to use this file

- list the failure clearly
- explain why it matters
- define how we detect it early
- define what we do to reduce or contain it

## Active Risks

### 1. Incomplete or unstable model acquisition

Why it matters:
- if users cannot get a model in reliably, the whole product feels broken before runtime even starts

Early signs:
- downloads stall
- progress appears frozen
- partial folders contain metadata but no usable shard set
- repeated resume failures or cache corruption

Mitigation:
- keep acquisition state separate from runtime state
- show plain-English source state, not just bytes
- detect partial vs ready vs downloading clearly
- avoid calling a model runnable until all required runtime files and shards are present

Current status:
- partially mitigated

### 2. Runtime dependencies drift from actual loader needs

Why it matters:
- a folder can look valid while the runtime still cannot use it because libraries are missing or imports fail

Early signs:
- config reads but tokenizer or weights cannot load
- environment installs succeed but preflight still fails

Mitigation:
- keep a runtime bootstrap/preflight layer
- report dependency blockers separately from source blockers
- verify config/tokenizer loading before attempting full model load

Current status:
- partially mitigated

### 3. Model family assumptions leak into generic code

Why it matters:
- if V1 Qwen logic gets hard-coded everywhere, later family support becomes a rewrite

Early signs:
- Qwen-specific names show up in generic runtime modules
- validation and loading logic become tightly coupled to one layout everywhere

Mitigation:
- keep family-specific logic behind clear source/runtime helpers
- keep shared registry/session/acquisition layers model-family aware, not Qwen-named

Current status:
- active

### 4. Runtime correctness gets mixed with optimization too early

Why it matters:
- if we start compressing or offloading before baseline correctness is proven, failures become hard to debug

Early signs:
- generation bugs appear before a clean baseline load exists
- artifact logic starts affecting original model load path

Mitigation:
- prove baseline load path first
- keep original source path and optimized artifact path separate
- make optimization an additive layer after correctness

Current status:
- active

### 5. Disk usage grows faster than expected

Why it matters:
- local users will hit storage pain quickly once originals, artifacts, caches, and benchmarks pile up

Early signs:
- duplicate full model copies
- hidden caches grow without limits
- artifacts become expensive to recreate or compare

Mitigation:
- keep originals immutable
- prefer metadata-driven derived artifacts over naive full copies where possible
- track artifact paths and storage layout explicitly

Current status:
- active

### 6. Benchmark outputs become misleading

Why it matters:
- if pcketlm claims an optimization is “better” without honest comparison, users will lose trust fast

Early signs:
- speed improves while quality quietly collapses
- comparisons use inconsistent prompts or hardware state
- benchmark language gets vague

Mitigation:
- use fixed benchmark prompts
- compare original vs variant side by side
- keep plain-English summaries tied to real measurements

Current status:
- active

### 7. UI gets polished before core behavior is honest

Why it matters:
- pretty screens can hide weak foundations and create rework

Early signs:
- screen work starts before load, bootstrap, and validation states are settled
- UI labels need constant rewrites because backend terms are unstable

Mitigation:
- use terminal/CLI proof first
- treat desktop UI as a consumer of stable state models
- only promote states to UI once acquisition/runtime logic is trustworthy

Current status:
- active

### 8. Real hardware constraints invalidate the promise

Why it matters:
- pcketlm is built around limited hardware, so if memory math or load strategy is unrealistic, the product promise breaks

Early signs:
- baseline model load requires more RAM/VRAM than expected
- preflight passes but real loading fails under memory pressure

Mitigation:
- keep correctness and memory tracking close together
- treat preflight as necessary but not sufficient
- benchmark on real hardware before making product claims

Current status:
- active

## Current Top Priorities

1. Finish acquiring the full official model source
2. Attempt the first real baseline model load
3. Record the exact blockers from that first real load
4. Keep optimization work behind the correctness milestone
