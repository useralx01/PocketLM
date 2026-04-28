# pcketlm V1 Blueprint

## Product Definition

`pcketlm` is a desktop tool for power users that lets them run and personalize dense local models, starting with Qwen-class models, by creating reversible optimized runtime profiles for limited hardware.

## V1 Goal

Build a custom local runtime and optimizer that can:

- import a dense Qwen-family text model
- preserve the untouched original model
- create one or more optimized derived runtime artifacts
- let the user choose what capabilities to keep or weaken
- benchmark original vs optimized variants
- chat with both for direct comparison

## What V1 Is

- Windows-first desktop software
- dense-model only
- Qwen-first
- local-only
- single-user
- reversible by design
- correctness first, optimization second

## What V1 Is Not

- not MoE support
- not multimodal
- not cloud-hosted
- not a marketplace
- not raw weight editing for end users
- not raw token deletion for end users
- not a giant polished product UI on day one

## Primary User

Power users who want to run and shape local models on limited hardware and are willing to trade some convenience for control.

## Core Promise

- keep the original model safe
- let the user create optimized variants
- explain tradeoffs in plain English
- show real benchmark results before and after changes

## User Experience

### Main Flow

1. User imports a supported Qwen-family dense text model.
2. `pcketlm` scans available hardware.
3. User chooses goals and capability priorities.
4. `pcketlm` builds an optimized runtime profile and derived artifact.
5. User benchmarks original vs optimized variant.
6. User opens a minimal chat shell to test both.

### User-Facing Optimization Categories

- Coding
- Planning
- Plain English Chat
- Math and Logic
- Tool Use
- Long Context
- Speed
- Memory Savings
- Creativity
- Multilingual
- Stability
- Agent Role Mode

### Example User Choices

- keep Coding high, reduce Creativity and Multilingual
- keep Plain English Chat and Planning, reduce Long Context
- create a Coder Agent profile for 16 GB hardware

## Reversibility Model

Reversibility does not depend on undoing every low-level change manually.

Instead:

- the original model stays immutable
- optimized artifacts are derived and disposable
- profiles and metadata describe how the artifact was produced
- a variant can be deleted and recreated without harming the source model

## Storage Model

Each imported model should have three logical layers.

### 1. Original

- untouched source model
- immutable
- never modified directly

### 2. Artifact

- optimized working variant
- this is what actually runs
- can be regenerated

### 3. Profile

- metadata describing the optimization choices
- hardware target
- benchmark results
- capability priorities
- runtime strategy

### Example Structure

```text
pcketlm/
  models/
    qwen-example/
      original/
      artifacts/
        balanced-16gb/
        coder-16gb/
      profiles/
        balanced-16gb.profile.json
        coder-16gb.profile.json
      benchmarks/
        balanced-16gb.benchmark.json
        coder-16gb.benchmark.json
```

## Technical Architecture

V1 should be split into six major parts.

### 1. Model Import Layer

Responsibilities:

- detect supported Qwen-family dense models
- validate source files
- register model metadata
- preserve immutable originals

### 2. Optimization Engine

Responsibilities:

- convert user priorities into optimization plans
- apply reversible optimization passes
- produce derived runtime artifacts

Allowed V1 optimization directions:

- quantization
- runtime offload policy
- cache policy
- context and KV controls
- capability-aware profile tuning
- optional safe structural reductions if proven stable

Not V1:

- arbitrary raw weight removal UI
- arbitrary token deletion UI
- distillation pipeline
- MoE expert pruning

### 3. Runtime Engine

Responsibilities:

- load derived artifacts
- manage memory
- schedule inference
- run dense text generation correctly

V1 principles:

- correctness before speed
- family-aware internals
- dense-only assumptions are acceptable in V1

### 4. Benchmark Engine

Responsibilities:

- compare original vs optimized
- record speed, memory, and capability impact
- produce plain-English summaries

Benchmark dimensions:

- load success
- VRAM use
- RAM use
- tokens per second
- prompt handling
- category-specific comparison
- stability

### 5. Minimal Chat Shell

Responsibilities:

- let the user actually test the model
- compare original and optimized behavior
- support direct runtime verification

V1 shell should be minimal, not fully polished.

### 6. Desktop App Layer

Responsibilities:

- import models
- select optimization goals
- create profiles
- launch benchmarks
- open chat shell

## Why Qwen-First

Qwen is the first target family because it appears highly relevant to the intended user path and gives a strong first lane for dense-model support.

The architecture should still stay family-aware so later support can extend to:

- other dense Qwen-family variants
- Llama-family dense models
- Mistral-family dense models
- later MoE families

## Agent-Role Future

V1 should not implement full multi-agent orchestration first.

But V1 should be designed so future optimization profiles can target roles like:

- coder
- planner
- translator
- router

This matters because different agents do not need the same capability balance.

## Platform Plan

- V1: Windows first
- V1.5 or early V2: Apple support as soon as practical

## Import Strategy

Primary initial direction:

- import supported local model files from common open-weight ecosystems

The internal `pcketlm` storage and profile format should be its own system.

## Success Criteria For V1

V1 is successful if it can:

- import at least one supported dense Qwen-family model correctly
- preserve the untouched original model
- generate at least one optimized runnable artifact
- run chat inference through the custom runtime
- benchmark original vs optimized variant
- explain tradeoffs in plain English

## Build Phases

### Phase 0: Specification

- lock product definition
- lock V1 boundaries
- lock storage model
- lock benchmark goals

### Phase 1: Core Import And Registry

- model registration
- source validation
- immutable original storage

### Phase 2: Runtime Correctness Prototype

- load one supported dense model
- run generation correctly
- basic memory tracking

### Phase 3: First Optimization Pipeline

- build reversible derived artifact
- produce first profile type
- compare baseline vs optimized

### Phase 4: Benchmark Engine

- automate comparisons
- write benchmark summaries

### Phase 5: Minimal Chat Shell

- original vs optimized chat testing

### Phase 6: Desktop Workflow Integration

- import
- optimize
- benchmark
- test

## Current Risks

- full custom runtime is a deep systems problem
- optimization quality may be hard to preserve early
- disk usage can grow quickly if artifacts are duplicated naively
- Apple support will require careful runtime planning later
- benchmark quality evaluation must stay honest and repeatable

## Failure Avoidance Rules

- keep acquisition, runtime preflight, and real loading as separate truth layers
- do not call a source runnable until all required files and shards are present
- do not begin optimization work until baseline dense loading is proven
- treat desktop UI as a consumer of stable backend states, not as the place where states are invented
- log blockers in plain English as soon as they appear so the project does not drift into vague failures

## Current Open Questions

- exact first supported Qwen model variant
- exact source format support for first import pass
- exact safe set of V1 optimization passes
- exact desktop shell technology
