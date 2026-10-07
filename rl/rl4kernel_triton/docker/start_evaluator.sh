#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

gpu="${TRITON_RL_GPU:-0}"
# Both checks and the service use the same physical index before worker masking.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
python -m triton_rl.preflight evaluator --gpu "$gpu"
exec python -m triton_rl.evaluation.server \
    --reference-root "${TRITON_RL_REFERENCE_ROOT:-/references}" \
    --gpu "$gpu" \
    --host "${TRITON_RL_HOST:-0.0.0.0}" \
    --port "${TRITON_RL_PORT:-8080}" \
    --timeout-seconds "${TRITON_RL_TIMEOUT_SECONDS:-1500}" \
    --seed "${TRITON_RL_SEED:-42}" \
    "$@"
