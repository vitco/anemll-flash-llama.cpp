# Flash-MoE GGUF Sidecar Tools

These tools implement the first Flash-MoE GGUF workflow for the vendored `llama.cpp` tree:

- keep the original GGUF as the canonical model artifact
- extract routed MoE tensors into a sidecar manifest plus raw bank files
- preserve exact tensor payload sizes and exact GGUF quantized bytes
- reload those expert tensors through `--moe-sidecar`
- support both a resident packed-bank path and an experimental streamed slot-bank path
- estimate persistent-bank cost and coverage from exact sidecar bytes plus `--moe-trace` output

By default, the helper scripts keep generated sidecars outside the repo under `/Users/anemll/Models/flash`.
Override that root with `FLASH_ROOT=/some/other/path` or set `SIDECAR_DIR` directly.

Modeling workflow: [`docs/moe-bank-modeling-workflow.md`](/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/docs/moe-bank-modeling-workflow.md)

## Current scope

- Supported runtime modes in this build:
  - `stock`
  - `resident`
  - `resident-bank`
  - `slot-bank`
  - `oracle-all-hit`
  - `oracle-prefetch`
- Manifest layout implemented here:
  - `layer_major_whole_tensor`
- Future work still not implemented end-to-end:
  - dynamic quant bank switching

## Extract a Qwen3.5 sidecar

```bash
PYTHON=python3 \
./llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model /Users/anemll/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir /Users/anemll/Models/flash/qwen35
```

## Verify the sidecar

```bash
PYTHON=python3 \
./llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model /Users/anemll/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --sidecar /Users/anemll/Models/flash/qwen35
```

## Inspect a model or subset

```bash
PYTHON=python3 \
./llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --layers 1-2 \
  --families routed \
  --include-shared
```

## Extract only selected layers

```bash
PYTHON=python3 \
./llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --out-dir /Users/anemll/Models/flash/kimi-layer1 \
  --layers 1 \
  --families all \
  --include-shared
```

## Estimate persistent-bank cost and coverage

```bash
python3 ./llama.cpp/tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /tmp/kimi-k25-trace.jsonl \
  --banks 4 --banks 8 --banks 16 --banks 32 --banks 64 \
  --byte-budget-gib 8 --byte-budget-gib 16 --byte-budget-gib 24 --byte-budget-gib 32
```

Live terminal dashboard:

```bash
python3 ./llama.cpp/tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  --banks 4 --banks 16 --banks 64 \
  --byte-budget-gib 64 --byte-budget-gib 72 --byte-budget-gib 96 \
  --watch 20
```

Optional dashboard export:

```bash
python3 ./llama.cpp/tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /tmp/kimi-k25-trace.jsonl \
  --banks 4 --banks 16 --banks 64 \
  --byte-budget-gib 64 --byte-budget-gib 72 --byte-budget-gib 96 \
  --svg-out /Users/anemll/Models/flash/logs/kimi-k25-cache-dashboard.svg
```

Long Kimi trace run without the normal `llama-cli` chat-loop exit:

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
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  -c 256 \
  --context-shift \
  --seed 123 --temp 0 \
  --ignore-eos \
  --moe-trace /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  -p "What is Apple Neural Engine?" \
  -n 12000 \
  > /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.log 2>&1 &
```

The default build includes `-DLLAMA_FLASH_MOE_GPU_BANK=ON`, and Kimi/DeepSeek2 GPU-bank placement is enabled by default at runtime.
`-ngl 999` offloads dense/shared tensors to GPU; routed expert bytes come from the sidecar slot-bank path.
Keep `--fit` enabled so dense/shared offload is clamped against the routed slot-bank reserve.
Use `-ub 1` for correct Kimi output; multi-token routed prefill produces degraded results.
Set `LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1` to force the host-backed path if you hit hangs or memory pressure.

## Run with the sidecar

```bash
./llama.cpp/build-flashmoe/bin/llama-cli \
  -m /Users/anemll/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --moe-mode resident-bank \
  --moe-sidecar /Users/anemll/Models/flash/qwen35 \
  --moe-verify-sidecar \
  --seed 123 --temp 0 \
  -p "Summarize Flash-MoE in two sentences." \
  -n 48
```

## Run a streamed slot-bank smoke test

```bash
./llama.cpp/build-flashmoe/bin/llama-cli \
  -m /Users/anemll/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/flash/qwen35 \
  --moe-slot-bank 32 \
  --moe-topk 4 \
  -fit on \
  -ub 1 -b 32 \
  --seed 123 --temp 0 \
  -ngl 999 \
  -p "What is Apple Neural Engine" \
  -n 32 -st
```

In the default build of this fork, Qwen `slot-bank` is expected to use `-ngl 999`.
You do not need to know how many layers are routed MoE versus dense/shared; routed expert tensors are virtualized out of the normal GGUF loader and continue to come from the sidecar path. Keep `--fit` on so the fork clamps dense/shared offload against the routed bank budget on unified-memory systems.

If you want a shared-expert-only control run with the same dense/shared placement, add `--moe-shared-only`. That bypasses routed experts at graph build time while leaving shared experts active, which is useful for MLX-style prefill and dense/shared diagnostic comparisons.

## Notes

- The extractor writes one raw bank file per sparse layer: `layer_XXX.bin`.
- Each file concatenates whole-tensor expert payloads in a stable layer/family order.
- `resident-bank` overrides expert tensors at load time and keeps the dense graph untouched.
- `slot-bank` reads experts on demand with `pread()` into a small resident slot bank. It is still experimental.
- The default build includes `-DLLAMA_FLASH_MOE_GPU_BANK=ON`. In slot-bank mode, routed experts stream from SSD via the sidecar path regardless of `-ngl`. Dense/shared weights are offloaded to GPU via `-ngl 99`.
- For sidecar runs with `-ngl > 0`, keep `--fit` enabled. The fitter is the supported end-user path for clamping dense/shared offload against the routed slot-bank reserve. If you need to deliberately bypass it for a supervised manual test, set `LLAMA_FLASH_MOE_ALLOW_UNFIT_OFFLOAD=1`.
- `--moe-topk N` is an experimental reduction-only runtime override for routed experts per token. It must be less than or equal to the GGUF model's native MoE top-k.
- `--moe-shared-only` is an experimental shared-expert-only diagnostic. It is most meaningful on MoE architectures that actually have shared experts, such as Qwen3.5 MoE and Kimi/DeepSeek2.
- For Qwen3.5 small-memory smoke tests, prefer `--moe-slot-bank 32` or `64`. A slot bank of `256` effectively reserves a full routed-expert bank for that model.
- In a GPU-bank build, DeepSeek2/Kimi routed GPU-bank placement is attempted by default. If a machine hits hangs or memory pressure, set `LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1` to fall back to the older host-backed routed path.
- Partial sidecars are supported: any tensor not present in the manifest continues to load from the original GGUF.
- `flashmoe_cache_estimator.py` models persistent-bank cost from the exact manifest and, when given a trace, reports static-bank, global-budget, and LRU coverage/miss estimates.
- The default estimator output is terminal-friendly text with ASCII bars; `--svg-out` adds a self-contained dashboard file.
- `--watch N` turns the estimator into a live terminal dashboard that refreshes every `N` seconds until interrupted.
