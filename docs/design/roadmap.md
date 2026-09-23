# pcketlm Engineering Roadmap

## Current priority

Turn the V1 blueprint into the first concrete build sequence.

## Phase 1: Foundation

Goal:
- create the project skeleton and runtime boundaries before model work begins

Tasks:
- define repository layout
- define internal naming conventions
- define model registry schema
- define profile schema
- define benchmark result schema
- define artifact storage layout

Deliverable:
- stable project file layout and data contracts

## Phase 2: Model Import Core

Goal:
- load and register one supported Qwen-family dense model safely

Tasks:
- detect required model files
- validate model folder structure
- read config metadata
- register immutable original model
- record source information in project metadata

Deliverable:
- imported original model with metadata and validation results

## Phase 3: Runtime Correctness Prototype

Goal:
- make the custom runtime produce correct output from the imported model

Tasks:
- implement tokenizer loading
- implement config loading
- implement weight discovery
- implement baseline dense inference path
- implement basic generation loop
- record RAM and VRAM use

Deliverable:
- one dense Qwen model can generate text correctly through the custom runtime

## Phase 4: First Optimization Pipeline

Goal:
- generate the first derived artifact from the immutable original

Tasks:
- define optimization profile input
- implement first safe optimization pass
- create derived artifact folder
- persist profile metadata
- link artifact to original model

Deliverable:
- one optimized runnable artifact with reversible lineage

## Phase 5: Benchmark Core

Goal:
- compare original vs optimized variants honestly

Tasks:
- define benchmark prompts
- define metric collection
- collect latency and throughput
- collect RAM and VRAM use
- store results in benchmark files
- generate plain-English summaries

Deliverable:
- first side-by-side benchmark output

## Phase 6: Minimal Chat Shell

Goal:
- make runtime testing usable without waiting for the full app

Tasks:
- choose simple shell format
- load original or optimized target
- send prompt
- stream output
- show profile and benchmark context

Deliverable:
- testable chat shell for manual comparison

## Phase 7: Desktop Workflow

Goal:
- connect import, optimize, benchmark, and chat into one flow

Tasks:
- choose shell technology
- build model import screen
- build category selection screen
- build profile creation flow
- build benchmark viewer
- build chat comparison flow

Deliverable:
- first usable desktop product flow

## Immediate next tasks

- pick the first Qwen model source format we will support
- write the model registry schema
- write the profile schema
- write the benchmark schema
- define the initial code folder layout

## Blockers

- full baseline model load still depends on the remaining official shard files arriving
- the first real load attempt will define the next runtime debugging loop

## Failure Avoidance Focus

- keep acquisition truth separate from runtime truth
- keep runtime preflight separate from actual loading
- do not start optimization passes before baseline correctness is proven
- keep family-specific logic contained so later allrounder support does not become a rewrite
