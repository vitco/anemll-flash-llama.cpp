# Native Slot-Bank Porting Guide

This guide is for adding the Flash-MoE native partial-bank path to another `llama.cpp` model architecture.

The important distinction is:

- the storage/runtime pieces are mostly shared
- the native handoff into the routed graph is model-specific

Qwen35MoE is the current reference implementation.
`deepseek2`/Kimi should be treated as the next port, not as something that automatically inherits the Qwen path.

## What Is Shared Already

These parts should stay reusable across models:

- Flash-MoE sidecar manifest and tensor override loading
- slot-bank storage layout and per-layer slot ownership
- miss install logic from SSD or resident packed source
- oracle replay and temporal prefetch machinery
- trace capture and cache-estimator tooling

In practice, the shared runtime lives mostly in:

- `src/llama-context.cpp`
- `src/llama-model.cpp`
- `src/llama-model-loader.cpp`
- `tools/flashmoe-sidecar/`

## What Is Model-Specific

The native slot-bank path becomes model-specific at the routed-expert graph boundary.

For each model, answer these questions first:

1. Where are routed top-k expert ids produced?
2. Which routed tensors consume those ids?
3. Which other graph paths still need true expert ids, not slot ids?
4. Are routed gate/up/down/bias/shared tensors all aligned to the same expert-id contract?
5. Can slot ids be produced from top-k inside the graph without breaking gating weights or LoRA paths?

If those answers are not clear, do not start coding yet.

## Current Qwen Pattern

Qwen35MoE now uses the native partial-bank path as the reference design.

The shape is:

1. The graph computes `ffn_moe_topk` normally.
2. The runtime decides whether that layer uses native slot mapping.
3. If yes, the graph builds a slot-id tensor from `ffn_moe_topk` through a custom op.
4. Routed expert matmuls consume slot ids, while the original expert ids remain available for gating-weight lookups.
5. The old callback/input-tensor path remains available as a fallback.

This is intentionally smaller than a deep `ggml` operator rewrite.

## Porting Checklist

### 1. Confirm routed tensor geometry

Before touching the graph:

- verify routed expert tensors are already slot-bank-shaped virtual tensors in slot-bank mode
- verify all routed families that must share slot ownership stay aligned
- verify the model does not hide a separate routed bias or expert-specific scaling path that still expects true expert ids

### 2. Map the existing graph contract

Document:

- the top-k tensor name and shape
- the routed weight/weight-scale lookup path
- all `mul_mat_id` / `add_id` style consumers
- any LoRA or adapter paths wrapping routed expert matmuls

Do not replace ids until you know which consumers need:

- true expert ids
- slot ids

### 3. Add an architecture gate

Keep native slot-bank opt-in per architecture.

The runtime should expose something equivalent to:

- `uses_layer(layer)`
- `uses_native_slot_map(layer)`
- `build_slot_ids_tensor(ctx0, selected_experts, layer)`

This lets one model use the native path while others keep the callback path.

### 4. Preserve the fallback path

Keep the old callback/input-tensor route while bringing up a new architecture.

Use it for:

- A/B validation
- correctness comparison
- immediate rollback if outputs drift

An env-controlled fallback like the current Qwen path is a good pattern.

### 5. Build slot ids inside the graph

The first native cut should usually be:

- keep top-k computation unchanged
- build slot ids from top-k inside the graph/runtime boundary
- avoid an external graph input tensor for slot ids on the native path

That moves the handoff boundary without rewriting the whole operator stack.

### 6. Reuse the shared install path

Do not fork storage logic per model unless absolutely necessary.

The model-specific work should be the graph contract, not:

- sidecar bytes
- slot install mechanics
- resident packed-source handling
- trace/oracle plumbing

### 7. Validate correctness first

Run the same prompt and same settings through:

- old callback path
- new native path

Check:

- output quality
- routed summary metrics
- slot-bank miss behavior
- oracle replay compatibility if used

### 8. Then validate speed

For the first architecture port, compare at least:

- `slot-bank`
- `resident-slot-bank`
- `oracle-all-hit`

If generation speed barely moves, the handoff did not really improve.

## Common Pitfalls

### Overwriting expert ids too early

This is the easiest way to get silent wrong answers.

Many MoE graphs need true expert ids for:

- gating-weight lookup
- expert-weight scaling
- bias/id-based post-processing
- LoRA wrapped routed matmuls

Slot ids should only replace the consumers that truly operate on the slot bank.

### Assuming all MoE models are wired the same

They are not.

Even if two models both use routed experts, the graph may differ in:

- where top-k is computed
- whether group routing exists
- whether shared experts are fused separately
- whether routed bias or scaling uses id-based ops

### Breaking oracle or trace semantics

If the model changes what counts as the routed handoff boundary, make sure:

- trace capture still reflects true expert selection
- oracle replay still replays the same logical routed calls

### Going straight to a deep operator rewrite

That may be necessary later, but it is usually not the right first step.

Start with:

- native slot-id creation inside the graph
- shared slot-bank install path
- architecture-specific gating

Only go deeper if the simpler native handoff still leaves too much overhead.

## Recommended Port Order

1. Qwen35MoE
   Already done. Treat this as the reference implementation.

2. `deepseek2` / Kimi
   Next priority because it is the main target for larger streamed runs.

3. Other MoE architectures
   Only after Qwen and Kimi validate the shared contract.

## Suggested Validation Ladder For A New Model

Use the same prompt and same token budget for all runs.

1. old callback path
2. new native path
3. `resident-slot-bank` with the same bank size
4. full-bank resident ceiling if practical
5. `oracle-all-hit`

What you want to learn:

- did correctness stay intact
- did generation speed improve
- is the remaining gap mostly storage or still handoff/consume

## When To Generalize

Only generalize after the second model.

If both:

- Qwen35MoE
- `deepseek2` / Kimi

can share the same native slot-bank contract cleanly, then it is worth pulling more of the pattern into common helpers.

Until then, prefer small architecture-gated changes over a broad abstraction.
