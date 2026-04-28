# pcketlm Desktop Status Screen V1

## Purpose

This is the first desktop screen concept for `pcketlm`.

It should answer the most important user question immediately:

**"Can this model run yet, and if not, what exactly is blocking it?"**

This screen is intentionally based on real backend states that already exist in `pcketlm`.

## Screen Goal

Show one model and its current state clearly:

- acquisition state
- runtime readiness
- blockers
- next step

The screen should feel practical, not decorative.

## Main User Promise

The user should not need to guess:

- whether a model is still downloading
- whether it is import-ready
- whether the runtime can attempt a load
- whether the blocker is the source files or the runtime environment

## V1 Layout

```text
+----------------------------------------------------------------------------------+
| pcketlm                                                                   [v0.1] |
| Model Status                                                                      |
+----------------------------------------------------------------------------------+
| Model: Qwen2.5-14B-Instruct                                                       |
| Family: Qwen                          Format: safetensors-sharded                 |
| Source: Official local import                                                     |
+----------------------------------------------------------------------------------+
| Acquisition                                                                       |
| Status: Downloading                                                               |
| Progress: [##################------] 76.33%                                       |
| Downloaded: 21.0 GB / 27.51 GB                                                    |
| Shards: 1 / 8                                                                     |
| Summary: The download is active and has reached about 76.33% of the expected size.|
+----------------------------------------------------------------------------------+
| Runtime Readiness                                                                 |
| Status: Blocked                                                                   |
| Config: Ready                                                                     |
| Tokenizer: Ready                                                                  |
| Weights: Incomplete                                                               |
| Runtime libs: Ready                                                               |
| Summary: The runtime is blocked because only 1 of 8 expected shard files is here. |
+----------------------------------------------------------------------------------+
| Current Blockers                                                                  |
| - Only 1 of 8 expected shard files are present                                    |
+----------------------------------------------------------------------------------+
| Recommended Next Step                                                             |
| Let the download continue, then retry runtime load when the shard set is complete |
+----------------------------------------------------------------------------------+
| Actions                                                                           |
| [Refresh Status]   [Open Model Folder]   [Retry Preflight]   [View Details]       |
+----------------------------------------------------------------------------------+
```

## Section Breakdown

### 1. Header

Purpose:
- make it obvious the user is in the model-status area

Content:
- app name
- lightweight version label
- screen title

### 2. Model Identity Row

Purpose:
- anchor the user in what model they are looking at

Content:
- model label
- family
- format
- source origin

### 3. Acquisition Card

Purpose:
- answer download/import progress clearly

Backed by:
- acquisition snapshot
- download-state logic

Required fields:
- acquisition status
- progress bar
- bytes downloaded
- expected bytes when known
- shard count
- plain-English summary

### 4. Runtime Readiness Card

Purpose:
- answer whether the runtime can actually try loading

Backed by:
- runtime source descriptor
- runtime bootstrap result

Required fields:
- overall runtime status
- config readiness
- tokenizer readiness
- weight readiness
- runtime dependency readiness
- plain-English summary

### 5. Blockers Panel

Purpose:
- isolate actionable problems

Rule:
- only show real blockers
- do not mix warnings and blockers together

Examples:
- missing shard files
- missing runtime libraries
- tokenizer load failure
- config load failure

### 6. Recommended Next Step

Purpose:
- tell the user what to do without technical guesswork

Examples:
- let the download continue
- install missing runtime dependency
- retry runtime preflight
- import the now-ready model

### 7. Actions Row

V1 actions:
- `Refresh Status`
- `Open Model Folder`
- `Retry Preflight`
- `View Details`

Not V1:
- full optimization controls
- chat view embedded on this screen
- benchmark viewer on this screen

## Visual Direction

This should look like:

- clean
- confident
- technical but readable
- not over-designed

## Style Notes

- use strong status labels like `Downloading`, `Blocked`, `Ready`
- make the progress bar obvious
- make blockers red or high-contrast
- make summaries plain English, not backend jargon
- use cards, not giant tables

## Color Behavior

- downloading: amber or blue
- ready: green
- blocked: red
- warning: amber
- neutral metadata: slate/gray

## Why This Screen First

This is the best first desktop screen because it depends on states `pcketlm` already has:

- acquisition state
- runtime source readiness
- runtime bootstrap blockers

That means we are not inventing UI without backend truth.

## Notable V1 Non-Goals

- not a final polished visual system
- not the optimization workflow
- not benchmark comparison UI
- not chat UI

## What This Unlocks Later

Once this screen exists, it becomes the basis for:

- import progress view
- load troubleshooting view
- benchmark launch entry point
- profile creation entry point
- model catalog cards
