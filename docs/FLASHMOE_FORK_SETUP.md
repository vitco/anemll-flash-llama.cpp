# Flash-MoE Fork Setup

This directory is a clean standalone fork candidate derived from:

- upstream source checkout: `/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp`
- Flash-MoE patch source: `/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/llama.cpp`

The goal is to keep only the minimum `llama.cpp`-specific Flash-MoE delta:

- routed expert sidecar loading
- streamed slot-bank runtime
- oracle replay and temporal prefetch plumbing
- Flash-MoE trace support
- Flash-MoE sidecar and cache-estimator tooling

## Files intentionally changed from upstream

- `README.md`
- `common/arg.cpp`
- `common/common.cpp`
- `common/common.h`
- `common/speculative.cpp`
- `include/llama.h`
- `src/llama-context.cpp`
- `src/llama-context.h`
- `src/llama-cparams.h`
- `src/llama-graph.cpp`
- `src/llama-graph.h`
- `src/llama-model-loader.cpp`
- `src/llama-model-loader.h`
- `src/llama-model.cpp`
- `src/llama-model.h`
- `src/llama.cpp`
- `tools/cli/cli.cpp`
- `tools/llama-bench/llama-bench.cpp`
- `tools/flashmoe-sidecar/`
- `docs/moe-bank-modeling-workflow.md`
- `FLASHMOE_VENDOR_COMMIT`

## Local publish workflow

From this directory:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp
git status
git add .
git commit -m "Flash-MoE fork of llama.cpp"
```

Then create an empty GitHub repo named `anemll-flash-llama.cpp`, and connect it:

```bash
git remote rename origin local-source
git remote add upstream git@github.com:ggml-org/llama.cpp.git
git remote add origin git@github.com:anemll/anemll-flash-llama.cpp.git
git push -u origin main
```

Recommended ongoing workflow:

- keep `upstream/main` as the rebase target
- keep Flash-MoE changes small and documented
- avoid storing generated sidecars, traces, or build outputs in the fork repo
