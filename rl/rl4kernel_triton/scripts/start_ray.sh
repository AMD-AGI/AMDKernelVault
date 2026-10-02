#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: start_ray.sh head|worker --head-ip IP [options]

Options:
  --node-ip IP        Set this node's address. The head defaults to --head-ip.
  --num-gpus N        Advertise N GPUs on this node. Default: 8.
  --port N           Set the Ray head port. Default: 6379.
  --dashboard-port N Set the Ray dashboard port. Default: 8265.
  --block            Keep this process active after Ray starts.
  -h, --help         Show this help.

Start this script inside each trainer container.
The script does not stop an existing Ray cluster.
EOF
}

fail() { printf 'Error: %s\n' "$1" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "The $1 option requires a value."; }
is_uint() { [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]; }

if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then
    usage
    exit 0
fi
role="$1"
shift
[[ "$role" == head || "$role" == worker ]] || fail "Select head or worker."
head_ip=""
node_ip=""
num_gpus=8
port=6379
dashboard_port=8265
block=0
while (($#)); do
    case "$1" in
        --head-ip) need_value "$@"; head_ip="$2"; shift 2 ;;
        --node-ip) need_value "$@"; node_ip="$2"; shift 2 ;;
        --num-gpus) need_value "$@"; num_gpus="$2"; shift 2 ;;
        --port) need_value "$@"; port="$2"; shift 2 ;;
        --dashboard-port) need_value "$@"; dashboard_port="$2"; shift 2 ;;
        --block) block=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "The script received an unknown option." ;;
    esac
done
[[ -n "$head_ip" ]] || fail "Supply --head-ip."
[[ "$head_ip" != -* && "$head_ip" != *[[:space:]]* ]] || fail "Supply a valid head address."
[[ -z "$node_ip" || ( "$node_ip" != -* && "$node_ip" != *[[:space:]]* ) ]] || fail "Supply a valid node address."
is_uint "$num_gpus" && ((num_gpus > 0)) || fail "The GPU count must be a positive integer."
is_uint "$port" && ((port > 0 && port <= 65535)) || fail "The Ray port must be from 1 to 65535."
is_uint "$dashboard_port" && ((dashboard_port > 0 && dashboard_port <= 65535)) || fail "The dashboard port must be from 1 to 65535."
command -v ray >/dev/null || fail "Install Ray in the current environment."

export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
args=(start --num-gpus "$num_gpus" --disable-usage-stats)
if [[ "$role" == head ]]; then
    node_ip="${node_ip:-$head_ip}"
    args+=(--head --node-ip-address "$node_ip" --port "$port"
        --dashboard-host 0.0.0.0 --dashboard-port "$dashboard_port")
else
    args+=(--address "${head_ip}:${port}")
    [[ -z "$node_ip" ]] || args+=(--node-ip-address "$node_ip")
fi
((block == 0)) || args+=(--block)
exec ray "${args[@]}"
