#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_container.sh trainer|evaluator --image IMAGE [options] [-- COMMAND...]

Options:
  --name NAME           Set the container name. Default: triton-rl-ROLE.
  --models PATH         Mount a host directory at /models, read-only.
  --data PATH           Mount a host directory at /data, read-only.
  --output PATH         Mount a host directory at /output, writable.
  --cache PATH          Mount a host directory at /cache, writable.
  --references PATH     Mount evaluator references at /references, read-only.
  --gpu N               Select one allocated evaluator GPU. Default: 0.
  --visible-devices IDS Set the trainer HIP_VISIBLE_DEVICES list.
  --port N              Set the evaluator service port. Default: 8080.
  --timeout-seconds N   Set the evaluator timeout. Default: 1500.
  --seed N              Set the evaluator seed. Default: 42.
  --env NAME[=VALUE]    Supply an additional container environment variable.
  --detach              Start the container in the background.
  -h, --help            Show this help.

Supply existing absolute host paths for mounts.
The evaluator requires --references and a GPU allocated exclusively to that worker.
The evaluator worker sees its selected GPU as device 0.
The script uses host networking and does not remove existing containers.
EOF
}

fail() { printf 'Error: %s\n' "$1" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "The $1 option requires a value."; }
is_uint() { [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]; }
mount_dir() {
    local source_path="$1" target_path="$2" access="$3"
    [[ "$source_path" == /* && -d "$source_path" ]] || fail "Each mount requires an existing absolute directory."
    [[ "$source_path" != *:* && "$source_path" != *,* ]] || fail "A mount path must not contain a colon or comma."
    mounts+=(--mount "type=bind,src=${source_path},dst=${target_path}${access}")
}

if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then
    usage
    exit 0
fi
role="$1"
shift
[[ "$role" == trainer || "$role" == evaluator ]] || fail "Select trainer or evaluator."
image=""
name="triton-rl-${role}"
references=""
gpu=0
visible_devices="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
port=8080
timeout_seconds=1500
seed=42
detach=0
mounts=()
environment=()
command_args=()
while (($#)); do
    case "$1" in
        --image) need_value "$@"; image="$2"; shift 2 ;;
        --name) need_value "$@"; name="$2"; shift 2 ;;
        --models) need_value "$@"; mount_dir "$2" /models ,readonly; shift 2 ;;
        --data) need_value "$@"; mount_dir "$2" /data ,readonly; shift 2 ;;
        --output) need_value "$@"; mount_dir "$2" /output ""; shift 2 ;;
        --cache) need_value "$@"; mount_dir "$2" /cache ""; environment+=(-e HF_HOME=/cache/huggingface); shift 2 ;;
        --references) need_value "$@"; references="$2"; mount_dir "$2" /references ,readonly; shift 2 ;;
        --gpu) need_value "$@"; gpu="$2"; shift 2 ;;
        --visible-devices) need_value "$@"; visible_devices="$2"; shift 2 ;;
        --port) need_value "$@"; port="$2"; shift 2 ;;
        --timeout-seconds) need_value "$@"; timeout_seconds="$2"; shift 2 ;;
        --seed) need_value "$@"; seed="$2"; shift 2 ;;
        --env)
            need_value "$@"
            [[ "$2" =~ ^[A-Za-z_][A-Za-z0-9_]*(=.*)?$ ]] || fail "Supply a valid environment variable name."
            requested_env_name="${2%%=*}"
            if [[ "$role" == evaluator ]]; then
                case "$requested_env_name" in
                    HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL)
                        fail "Select the evaluator GPU with --gpu instead of an environment mask." ;;
                esac
            fi
            environment+=(-e "$2")
            shift 2
            ;;
        --detach) detach=1; shift ;;
        --) shift; command_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "The script received an unknown option." ;;
    esac
done
[[ -n "$image" && "$image" != -* ]] || fail "Supply --image with a container image reference."
[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "Supply a valid container name."
is_uint "$gpu" || fail "The GPU index must be a nonnegative integer."
is_uint "$port" && ((port > 0 && port <= 65535)) || fail "The service port must be from 1 to 65535."
is_uint "$timeout_seconds" && ((timeout_seconds > 0)) || fail "The timeout must be a positive integer."
is_uint "$seed" && ((${#seed} <= 10 && seed <= 4294967295)) || fail "The seed must be from 0 to 4294967295."
[[ "$visible_devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "Supply a comma-separated GPU index list."
command -v docker >/dev/null || fail "Install Docker on the host."
[[ -e /dev/kfd && -d /dev/dri ]] || fail "The host must expose /dev/kfd and /dev/dri."

args=(run --rm --name "$name" --network host --device /dev/kfd --device /dev/dri
    --shm-size 128g --ulimit memlock=-1 --ulimit stack=67108864)
for group in video render; do
    group_entry="$(getent group "$group" || true)"
    if [[ -n "$group_entry" ]]; then
        IFS=: read -r _ _ group_id _ <<< "$group_entry"
        args+=(--group-add "$group_id")
    fi
done
if ((detach)); then
    args+=(--detach)
elif [[ -t 0 && -t 1 ]]; then
    args+=(-it)
fi

if [[ "$role" == evaluator ]]; then
    [[ -n "$references" ]] || fail "Supply --references for the evaluator."
    environment+=(-e "TRITON_RL_GPU=${gpu}"
        -e TRITON_RL_DEDICATED_CONTAINER=1 -e "TRITON_RL_PORT=${port}"
        -e "TRITON_RL_TIMEOUT_SECONDS=${timeout_seconds}" -e "TRITON_RL_SEED=${seed}")
    ((${#command_args[@]})) || command_args=(/opt/triton-rl/docker/start_evaluator.sh)
else
    # Preserve the archived trainer's ROCm process and NUMA settings.
    args+=(--cap-add SYS_PTRACE --security-opt seccomp=unconfined)
    environment+=(-e "HIP_VISIBLE_DEVICES=${visible_devices}" -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1)
    if ((${#command_args[@]} == 0)); then
        if ((detach)); then command_args=(sleep infinity); else command_args=(bash); fi
    fi
fi
exec docker "${args[@]}" "${mounts[@]}" "${environment[@]}" "$image" "${command_args[@]}"
