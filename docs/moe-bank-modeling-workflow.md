# MoE Bank Modeling Workflow

This is the reusable workflow for estimating persistent Flash-MoE residency on a new model or backend.
It is written for the current `llama.cpp` GGUF sidecar path, but the same sequence applies to MLX and future ports.

## Goals

The workflow is meant to answer these questions in order:

- what routed expert unit are we banking
- how much resident memory does that unit cost
- how much miss traffic does a given bank policy leave on the table
- whether the current runtime is miss-limited or hit-limited
- whether the next lever is more residency, a better consume path, or prefetch / predictor work

## Checklist

1. Build a resident anchor first.
   Prove correctness and record a backend-native upper bound before any streaming or banking work.

2. Define the routed expert unit from exact manifests, not assumptions.
   For GGUF sidecars, that means `layer`, `tensor_family`, `quant_type`, `exact_byte_length`, and `bytes_per_expert`.

3. Build the cost model from the manifest.
   For a per-layer bank, resident bytes are:
   `sum_over_layers(slot_bank_size * sum(bytes_per_expert for routed families in that layer))`

4. Collect routing traces under realistic workloads.
   Use real prompts, longer generations, and if relevant structured or tool-heavy prompts.

5. Normalize the routing workload into metrics.
   At minimum track:
   `unique experts per layer`
   `expert frequency`
   `misses/token`
   `bytes/token`
   `full-hit layer-call rate`

6. Simulate multiple policies.
   Always compare at least:
   `uniform per-layer static`
   `global static under a byte budget`
   `refillable LRU`

7. Separate hit-path and miss-path value.
   Persistent-cache simulation tells us how much routed traffic survives.
   Runtime anchors tell us how much that traffic matters in tokens/second.

8. Calibrate with real runtime anchors.
   Use:
   `baseline streamed runtime`
   `current banked runtime`
   `oracle all-hit replay`
   `oracle one-step replay` if prefetch is relevant

9. Treat outliers explicitly.
   Shared experts, BF16/Q5/Q6 outliers, or non-uniform routed families should be modeled as separate candidate classes instead of being forced into a uniform slot assumption.

10. Turn the model into a design decision.
    Once the estimate is calibrated, decide whether the next step is more RAM, a better hit path, better miss commit, or predictor work.

## Current `llama.cpp` Tooling

Two tools live under [`llama.cpp/tools/flashmoe-sidecar/`](/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/llama.cpp/tools/flashmoe-sidecar):

- [`flashmoe_sidecar.py`](/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py)
  Extracts, verifies, and inspects GGUF sidecars.
- [`flashmoe_cache_estimator.py`](/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/llama.cpp/tools/flashmoe-sidecar/flashmoe_cache_estimator.py)
  Reads a sidecar manifest and optional `--moe-trace` JSONL to estimate resident-bank cost and coverage.

## Kimi-K2.5 Example

### 1. Inspect the routed geometry

```bash
python3 ./llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --families routed
```

### 2. Capture a speed-oriented routing trace

This keeps the current Kimi testing shape:

Use `-ub 1` for the current Kimi shape so routing traces match the sane single-token prefill path.

```bash
./llama.cpp/build-flashmoe/bin/llama-cli \
  -m /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --moe-slot-bank 64 \
  --moe-topk 4 \
  --no-warmup \
  -ub 1 -b 64 \
  -ngl 0 \
  -c 256 \
  --seed 123 --temp 0 \
  --moe-trace /tmp/kimi-k25-trace.jsonl \
  -p "What is Apple Neural Engine?" \
  -n 256 -st
```

### 3. Run a longer trace when decode is slow

For bank modeling, a longer trace is usually more valuable than a perfect short benchmark.
If you want roughly an hour of decode on the current Kimi setup, use the Flash-MoE raw trace harness on top of `llama-cli`.
This bypasses the normal chat loop and chat-template stop behavior, so the run keeps going until the token budget is actually spent.

```bash
mkdir -p /Users/anemll/Models/flash/logs

nohup ./llama.cpp/build-flashmoe/bin/llama-cli \
  -m /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --moe-slot-bank 64 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --moe-trace-harness \
  --no-warmup \
  -ub 1 -b 64 \
  -ngl 0 \
  -c 256 \
  --context-shift \
  --seed 123 --temp 0 \
  --ignore-eos \
  --moe-trace /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  -p "What is Apple Neural Engine?" \
  -n 12000 \
  > /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.log 2>&1 &
```

Quick status checks:

```bash
pgrep -af "llama-cli.*Kimi-K2.5.*--moe-trace-harness"
wc -l /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.jsonl
tail -n 5 /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.log
```

### 4. Estimate persistent-bank cost and coverage

```bash
python3 ./llama.cpp/tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  --banks 4 --banks 8 --banks 16 --banks 32 --banks 64 \
  --byte-budget-gib 8 --byte-budget-gib 16 --byte-budget-gib 24 --byte-budget-gib 32
```

## First-Pass Kimi-K2.5 Geometry

From the current routed manifest:

- 60 routed layers
- 384 experts
- native routed top-k `8`
- tested reduced top-k `4`
- routed sidecar bytes: about `232.58 GiB`

Resident cost for a uniform per-layer persistent bank on the current Kimi sidecar:

- bank `4`: about `2.26 GiB`
- bank `8`: about `4.51 GiB`
- bank `16`: about `9.03 GiB`
- bank `32`: about `18.05 GiB`
- bank `64`: about `36.10 GiB`

That table is model geometry only.
The longer trace is what tells us how much of the real routed traffic those budgets absorb.

## Notes

- The estimator treats each traced layer call as the current runtime unit, which matches the present `slot-bank` union-over-ubatch behavior in `llama.cpp`.
- `uniform per-layer static` is the closest model for a simple persistent resident bank.
- `global static under byte budget` is the better fit once layer costs differ materially.
- `refillable LRU` is useful as a miss-path reference, but it is not the same thing as a persistent cache.
- Throughput estimation should only be trusted after adding runtime anchors such as baseline streamed tok/s and oracle all-hit tok/s.
