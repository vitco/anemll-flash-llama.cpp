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

## Model index

Per-model extract + run recipes in this document:

| Model | Arch | Extract | Run |
|---|---|---|---|
| Qwen3.5-35B-A3B | qwen3moe | [Extract](#extract-a-qwen35-sidecar) | [Run](#run-with-the-sidecar) |
| Gemma4-26B-A4B | gemma4 | [Extract](#extract-a-gemma4-26b-a4b-sidecar) | [Run](#run-gemma4-26b-a4b-with-the-sidecar) |
| Kimi K2 / K2.5 | deepseek2 (MLA) | [Extract](#extract-only-selected-layers) | [Run](#estimate-persistent-bank-cost-and-coverage) |
| **GLM-5.1** | **glm-dsa (MLA + DSA indexer)** | [**Extract**](#extract-a-glm-51-sidecar) | [**Run**](#run-glm-51-with-the-sidecar) |

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
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model /Users/anemll/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir /Users/anemll/Models/flash/qwen35
```

## Extract a Gemma4-26B-A4B sidecar

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model /Users/anemll/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --out-dir /Users/anemll/Models/gemma4/packed_experts \
  --force
```

## Extract a GLM-5.1 sidecar

GLM-5.1 (`glm-dsa` arch) is a 256×22B MoE with MLA attention + DeepSeek-style sparse-attention indexer.
The extractor walks the 6 GGUF shards automatically — point it at the first shard.

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model /Users/anemll/Models/GLM/GLM-5.1-UD-IQ1_M-00001-of-00006.gguf \
  --out-dir /Users/anemll/Models/flash/GLM-5.1-sidecar
```

The sidecar is roughly 177 GiB at IQ1_M/IQ2_XXS (76 MoE layers × 256 experts × gate/up/down). Make sure the target SSD has the room.

## Run GLM-5.1 with the sidecar

Daily-driver slot-bank recipe (mirrors the Kimi K2.5 fast-path: `--moe-prefetch-temporal`, `-b 64`, `-ub 1`, `--no-warmup`):

```bash
./build/bin/llama-cli \
  -m /Users/anemll/Models/GLM/GLM-5.1-UD-IQ1_M-00001-of-00006.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/flash/GLM-5.1-sidecar \
  --moe-slot-bank 64 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --no-warmup \
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  -c 4096 \
  --seed 123 --temp 0 \
  -p "What is Apple Neural Engine?" \
  -n 128 -st
```

GLM-5.1 specific notes:

- **`--moe-topk 4`** is a reduction-only override of the model's native K=8. On IQ1_M/IQ2_XXS quants the K=4 vs K=8 quality gap is within noise for general use, while halving per-token expert I/O for ~2× decode. Drop the flag to use native K=8 if you need maximum fidelity.
- **`--moe-slot-bank 64`** is the starting point. With native K=8 the bank has only 8× headroom; if you have RAM, try `128` or `256` for higher reuse on warm caches.
- **`--moe-prefetch-temporal`** is the single biggest knob — it overlaps next-layer expert `pread`s with current-layer GPU compute. Always on for SSD-bound MoE.
- **DSA indexer overhead is structural.** GLM has 79 layers (vs Kimi 61) plus a per-layer sparse-attention indexer that K2.5 doesn't have. Decode is part SSD-bound, part dense-compute-bound — ~3.9 tok/s on M5 Max 128 GB is the current realistic ceiling without quant-kernel work for IQ1_M/IQ2_XXS `mul_mat_id`.
- If you hit hangs or memory pressure (the DSV2/Kimi GPU-bank path is shared with GLM), fall back with:
  ```bash
  LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1 ./build/bin/llama-cli ...
  ```

## Verify the sidecar

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model /Users/anemll/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --sidecar /Users/anemll/Models/flash/qwen35
```

Gemma4 verification uses the same command shape:

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model /Users/anemll/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --sidecar /Users/anemll/Models/gemma4/packed_experts
```

## Inspect a model or subset

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --layers 1-2 \
  --families routed \
  --include-shared
```

## Extract only selected layers

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model /Users/anemll/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --out-dir /Users/anemll/Models/flash/kimi-layer1 \
  --layers 1 \
  --families all \
  --include-shared
```

## Estimate persistent-bank cost and coverage

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /tmp/kimi-k25-trace.jsonl \
  --banks 4 --banks 8 --banks 16 --banks 32 --banks 64 \
  --byte-budget-gib 8 --byte-budget-gib 16 --byte-budget-gib 24 --byte-budget-gib 32
```

Live terminal dashboard:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /Users/anemll/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  --banks 4 --banks 16 --banks 64 \
  --byte-budget-gib 64 --byte-budget-gib 72 --byte-budget-gib 96 \
  --watch 20
```

Optional dashboard export:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar /Users/anemll/Models/flash/Kimi-K2.5-sidecar \
  --trace /tmp/kimi-k25-trace.jsonl \
  --banks 4 --banks 16 --banks 64 \
  --byte-budget-gib 64 --byte-budget-gib 72 --byte-budget-gib 96 \
  --svg-out /Users/anemll/Models/flash/logs/kimi-k25-cache-dashboard.svg
```

Long Kimi trace run without the normal `llama-cli` chat-loop exit:

```bash
mkdir -p /Users/anemll/Models/flash/logs

nohup ./build/bin/llama-cli \
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
./build/bin/llama-cli \
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
./build/bin/llama-cli \
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

## Run Gemma4-26B-A4B with the sidecar

Resident-bank smoke test:

```bash
./build/bin/llama-cli \
  --color off --simple-io \
  -m /Users/anemll/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --moe-mode resident-bank \
  --moe-sidecar /Users/anemll/Models/gemma4/packed_experts \
  -cnv -st -fit on \
  -ub 1 -b 1 -ngl 0 -c 4096 --seed 0 --temp 0 \
  -p "Make a poem about Apple Neural Engine in 4 lines." \
  -n 64
```

Streamed slot-bank smoke test:

```bash
./build/bin/llama-cli \
  --color off --simple-io \
  -m /Users/anemll/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/gemma4/packed_experts \
  --moe-slot-bank 16 \
  -cnv -st -fit on \
  -ub 1 -b 1 -ngl 0 -c 4096 --seed 0 --temp 0 \
  -p "Make a poem about Apple Neural Engine in 4 lines." \
  -n 64
```

Gemma4-specific notes:

- `gemma-4-26B-A4B-it` is instruction tuned. For quality comparisons, prefer normal chat mode (`-cnv -st`) over `--moe-trace-harness`, which uses raw completion.
- Gemma4 uses native `n_expert_used = 8`. Leave `--moe-topk` unset for fidelity unless you are deliberately testing reduced top-k routing.
- On smaller-memory devices, Gemma4 is more sensitive to slot-bank size than Qwen3.5-35B because each selected expert payload is larger. A slot bank of `8` or `16` is a better starting point than desktop-style larger banks.

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
