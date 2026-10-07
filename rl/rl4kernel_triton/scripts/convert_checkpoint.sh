#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: convert_checkpoint.sh INPUT_SFT_CHECKPOINT OUTPUT_MEGATRON_CHECKPOINT" >&2
    exit 2
fi
: "${SLIME_ROOT:=/opt/slime}"
if [[ ! -d "$1" ]]; then
    echo "The input checkpoint directory does not exist." >&2
    exit 2
fi
if [[ -e "$2" ]]; then
    echo "The output checkpoint path already exists." >&2
    exit 2
fi
source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"
exec python "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$1" \
    --save "$2"
