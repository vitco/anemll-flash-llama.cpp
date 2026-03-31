#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${LLAMA_DIR}/.." && pwd)"
PYTHON_DEFAULT="${PROJECT_ROOT}/.env/bin/python3"

PYTHON_BIN="${PYTHON:-python3}"
if [[ -x "${PYTHON_DEFAULT}" ]]; then
  PYTHON_BIN="${PYTHON_DEFAULT}"
fi

MODEL="${MODEL:-/Users/anemll/Models/Kimi/Kimi-K2-Instruct-UD-TQ1_0-00001-of-00005.gguf}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_DIR}/build-flashmoe}"
RESULTS_DIR="${RESULTS_DIR:-${LLAMA_DIR}/flashmoe-results/kimi-k2}"
FLASH_ROOT="${FLASH_ROOT:-/Users/anemll/Models/flash}"
SIDECAR_DIR="${SIDECAR_DIR:-${FLASH_ROOT}/kimi-k2}"
PROMPT="${PROMPT:-Explain why stable expert banks matter for streamed MoE inference.}"
TOKENS="${TOKENS:-32}"
ENABLE_SIDECAR="${ENABLE_SIDECAR:-1}"
INCLUDE_SHARED="${INCLUDE_SHARED:-1}"
LAYERS="${LAYERS:-}"
FAMILIES="${FAMILIES:-}"

mkdir -p "${RESULTS_DIR}"

cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON

cmake --build "${BUILD_DIR}" --target llama-cli llama-bench -j

"${BUILD_DIR}/bin/llama-bench" \
  -m "${MODEL}" \
  -r 1 \
  --no-warmup \
  -p 64 \
  -n 32 \
  -ngl 999 \
  -o md > "${RESULTS_DIR}/stock-bench.md"

"${BUILD_DIR}/bin/llama-cli" \
  -m "${MODEL}" \
  --seed 123 \
  --temp 0 \
  -ngl 999 \
  -n "${TOKENS}" \
  -p "${PROMPT}" > "${RESULTS_DIR}/stock-smoke.txt"

if [[ "${ENABLE_SIDECAR}" == "1" ]]; then
  EXTRACT_ARGS=(--model "${MODEL}" --out-dir "${SIDECAR_DIR}" --force)
  VERIFY_ARGS=(--model "${MODEL}" --sidecar "${SIDECAR_DIR}")
  if [[ "${INCLUDE_SHARED}" == "1" ]]; then
    EXTRACT_ARGS+=(--include-shared)
  fi
  if [[ -n "${LAYERS}" ]]; then
    EXTRACT_ARGS+=(--layers "${LAYERS}")
    VERIFY_ARGS+=(--layers "${LAYERS}")
  fi
  if [[ -n "${FAMILIES}" ]]; then
    EXTRACT_ARGS+=(--families "${FAMILIES}")
    VERIFY_ARGS+=(--families "${FAMILIES}")
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/flashmoe_sidecar.py" extract "${EXTRACT_ARGS[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/flashmoe_sidecar.py" verify "${VERIFY_ARGS[@]}"

  "${BUILD_DIR}/bin/llama-bench" \
    -m "${MODEL}" \
    --moe-mode resident-bank \
    --moe-sidecar "${SIDECAR_DIR}" \
    --moe-verify-sidecar \
    -r 1 \
    --no-warmup \
    -p 64 \
    -n 32 \
    -ngl 999 \
    -o md > "${RESULTS_DIR}/resident-bank-bench.md"

  "${BUILD_DIR}/bin/llama-cli" \
    -m "${MODEL}" \
    --moe-mode resident-bank \
    --moe-sidecar "${SIDECAR_DIR}" \
    --moe-verify-sidecar \
    --seed 123 \
    --temp 0 \
    -ngl 999 \
    -n "${TOKENS}" \
    -p "${PROMPT}" > "${RESULTS_DIR}/resident-bank-smoke.txt"
fi

echo "results written to ${RESULTS_DIR}"
