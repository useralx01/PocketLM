# pcketlm Initial Schemas

## Purpose

These are the first draft data contracts for V1.

They are meant to keep the project consistent while the runtime is still being built.

## 1. Model Registry Schema

Each imported model should have one registry record.

### Example

```json
{
  "id": "qwen2.5-14b-instruct",
  "label": "Qwen2.5 14B Instruct",
  "family": "qwen",
  "type": "dense",
  "task": "text-generation",
  "source": {
    "kind": "local-folder",
    "path": "C:/.../models/qwen2.5-14b-instruct/original",
    "origin": "huggingface",
    "repoId": "Qwen/Qwen2.5-14B-Instruct"
  },
  "original": {
    "path": "C:/.../models/qwen2.5-14b-instruct/original",
    "immutable": true,
    "format": "safetensors-sharded"
  },
  "tokenizer": {
    "tokenizerJson": true,
    "vocabJson": true,
    "mergesTxt": true
  },
  "config": {
    "contextLength": null,
    "hiddenSize": null,
    "numLayers": null,
    "numAttentionHeads": null
  },
  "status": {
    "imported": false,
    "validated": false,
    "runnable": false
  },
  "createdAt": "",
  "updatedAt": ""
}
```

### Required fields

- `id`
- `label`
- `family`
- `type`
- `source`
- `original`
- `status`

## 2. Optimization Profile Schema

Each optimized variant should have one profile record.

### Example

```json
{
  "id": "coder-16gb-v1",
  "modelId": "qwen2.5-14b-instruct",
  "label": "Coder 16GB V1",
  "hardwareTarget": {
    "platform": "windows",
    "vramGb": 16,
    "ramGb": null
  },
  "goals": {
    "fitWithinHardware": true,
    "preserveOriginal": true,
    "reversible": true
  },
  "categories": {
    "coding": "high",
    "planning": "medium",
    "plainEnglishChat": "medium",
    "mathAndLogic": "medium",
    "toolUse": "medium",
    "longContext": "low",
    "speed": "high",
    "memorySavings": "high",
    "creativity": "low",
    "multilingual": "low",
    "stability": "high",
    "agentRoleMode": "coder"
  },
  "runtimeStrategy": {
    "mode": "hybrid",
    "offloadPolicy": "adaptive",
    "cachePolicy": "balanced",
    "kvPolicy": "bounded"
  },
  "optimizationPlan": {
    "quantization": null,
    "contextCap": null,
    "safeStructuralReduction": false
  },
  "artifact": {
    "path": "C:/.../artifacts/coder-16gb-v1",
    "ready": false
  },
  "benchmarks": {
    "latest": null
  },
  "createdAt": "",
  "updatedAt": ""
}
```

### Category value rules

For V1, category strength should be one of:

- `off`
- `low`
- `medium`
- `high`
- `critical`

## 3. Benchmark Result Schema

Each benchmark run should create one result file.

### Example

```json
{
  "id": "benchmark-2026-04-23T19-10-00",
  "modelId": "qwen2.5-14b-instruct",
  "profileId": "coder-16gb-v1",
  "baseline": {
    "label": "original",
    "loadSuccess": true,
    "tokensPerSecond": null,
    "vramGb": null,
    "ramGb": null
  },
  "variant": {
    "label": "coder-16gb-v1",
    "loadSuccess": true,
    "tokensPerSecond": null,
    "vramGb": null,
    "ramGb": null
  },
  "comparisons": {
    "speedDeltaPct": null,
    "vramDeltaPct": null,
    "ramDeltaPct": null
  },
  "quality": {
    "coding": null,
    "planning": null,
    "plainEnglishChat": null,
    "mathAndLogic": null,
    "toolUse": null,
    "longContext": null,
    "creativity": null,
    "multilingual": null,
    "stability": null
  },
  "summary": {
    "plainEnglish": ""
  },
  "createdAt": ""
}
```

## 4. Runtime Session Schema

Each runtime launch should have a session record.

### Example

```json
{
  "id": "session-2026-04-23T19-20-00",
  "modelId": "qwen2.5-14b-instruct",
  "profileId": "coder-16gb-v1",
  "mode": "chat",
  "status": "starting",
  "runtime": {
    "device": "cuda:0",
    "vramBudgetGb": 16,
    "ramBudgetGb": null
  },
  "stats": {
    "promptCount": 0,
    "generatedTokens": 0,
    "errors": 0
  },
  "createdAt": "",
  "updatedAt": ""
}
```

## 5. Validation Result Schema

Import validation should produce a record even if it fails.

### Example

```json
{
  "modelId": "qwen2.5-14b-instruct",
  "sourcePath": "C:/.../original",
  "checks": {
    "configJson": true,
    "tokenizerJson": true,
    "weightIndex": true,
    "weightsPresent": false
  },
  "result": "partial",
  "errors": [],
  "warnings": [
    "Weight index found but all shard files are not present yet."
  ],
  "createdAt": ""
}
```

## Notes

- These schemas are intentionally simple for V1.
- We can evolve them once the runtime core is alive.
- The main purpose right now is consistency, reversibility, and traceability.

## Current implementation note

The first implemented runtime/import slice now includes:

- required file validation for Qwen imports
- source inspection for config and shard metadata
- early source download-state estimation
- a product-facing acquisition snapshot with plain-English state and progress visualization
- a runtime source descriptor for loader-facing readiness checks
- a runtime bootstrap result for dependency checks, preflight checks, and first session creation
- a reduced-memory runtime strategy plan for choosing between plain load, staged streaming, and artifact-first paths
- a staged disk-streaming plan with working-window and cache-layout metadata
- a staged-streaming manifest on disk that persists cache layout, chunk plan, and adaptive window sizes
- a streaming-manifest reader view that validates cache directories and exposes runtime-usable chunk metadata
- a streaming unit map that converts actual shard files into streamable runtime units
- a window schedule that decides which streamable units belong in the hot and warm windows at a given moment
- model registry persistence
- registry-backed model catalog view
- registry record removal
- model and profile record objects
- stable project path helpers
