# pcketlm Initial Code Layout

## Goal

Keep V1 understandable while the runtime is still young.

## Suggested project layout

```text
pcketlm/
  src/
    pcketlm/
      app/
        desktop/
        chat_shell/
      core/
        model_import/
        registry/
        profiles/
        runtime/
        optimize/
        benchmark/
        validation/
        storage/
  models/
  state/
  docs/
  tests/
```

## Folder meanings

### `src/pcketlm/app/desktop`

- desktop application shell
- import flow
- optimization flow
- benchmark viewer

### `src/pcketlm/app/chat_shell`

- minimal testing interface
- compare original vs optimized variants

### `src/pcketlm/core/model_import`

- source model intake
- file discovery
- config and tokenizer loading

### `src/pcketlm/core/registry`

- model registry records
- lookup and persistence

### `src/pcketlm/core/profiles`

- optimization profile definitions
- profile parsing and validation

### `src/pcketlm/core/runtime`

- runtime session manager
- dense inference path
- memory and cache control

### `src/pcketlm/core/optimize`

- optimization planning
- artifact generation
- reversible derivation flow

### `src/pcketlm/core/benchmark`

- metric collection
- prompt suites
- comparison reports

### `src/pcketlm/core/validation`

- import validation
- artifact validation
- runtime readiness checks

### `src/pcketlm/core/storage`

- path management
- artifact layout
- metadata persistence

### `models`

- imported originals
- derived artifacts
- profile files
- benchmark outputs

### `state`

- runtime session state
- temporary internal state

### `docs`

- product docs
- internal design notes

### `tests`

- schema tests
- validation tests
- runtime correctness tests

## First code priority

The first folders we actually need are:

- `src/pcketlm/core/model_import`
- `src/pcketlm/core/registry`
- `src/pcketlm/core/validation`
- `src/pcketlm/core/runtime`
- `src/pcketlm/core/storage`

Everything else can grow around that.
