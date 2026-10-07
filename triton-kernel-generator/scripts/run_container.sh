#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_container.sh --image IMAGE --inputs DIRECTORY --output DIRECTORY [--api-key-env NAME] -- GENERATOR_ARGUMENTS...

Mount external source, references, cases, and manifests below /inputs, read-only.
Mount the output directory at /output, writable.
Select the allocated GPU with the generator's --gpu argument.
The optional API key passes through a named environment variable.
Use a disposable GPU host and restricted endpoint credentials.
EOF
}
fail() { printf 'Error: %s\n' "$1" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "The $1 option requires a value."; }

image=""
input_dir=""
output_dir=""
key_name=""
while (($#)); do
    case "$1" in
        --image) need_value "$@"; image="$2"; shift 2 ;;
        --inputs) need_value "$@"; input_dir="$2"; shift 2 ;;
        --output) need_value "$@"; output_dir="$2"; shift 2 ;;
        --api-key-env) need_value "$@"; key_name="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "The container launcher received an unknown option." ;;
    esac
done
[[ -n "$image" && "$image" != -* ]] || fail "Supply --image."
for directory in "$input_dir" "$output_dir"; do
    [[ "$directory" == /* && -d "$directory" ]] || fail "Mount paths must be existing absolute directories."
    [[ "$directory" != *:* && "$directory" != *,* ]] || fail "Mount paths must not contain colons or commas."
done
[[ -e /dev/kfd && -d /dev/dri ]] || fail "The host must expose the AMD GPU devices."
command -v docker >/dev/null || fail "Install Docker before using this launcher."
image_id="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)" || fail "Build or pull the requested image before starting this launcher."
args=(run --rm --user "$(id -u):$(id -g)" --network host --device /dev/kfd --device /dev/dri --shm-size 16g
    -e "TRITON_GEN_CONTAINER_REFERENCE=${image}"
    -e "TRITON_GEN_CONTAINER_IMAGE_ID=${image_id}"
    --mount "type=bind,src=${input_dir},dst=/inputs,readonly"
    --mount "type=bind,src=${output_dir},dst=/output")
for group in video render; do
    entry="$(getent group "$group" || true)"
    if [[ -n "$entry" ]]; then
        IFS=: read -r _ _ group_id _ <<< "$entry"
        args+=(--group-add "$group_id")
    fi
done
if [[ -n "$key_name" ]]; then
    [[ "$key_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail "Supply an environment variable name, not a credential value."
    args+=(-e "$key_name")
    generator_key_args=(--api-key-env "$key_name")
else
    generator_key_args=()
fi
exec docker "${args[@]}" "$image" "${generator_key_args[@]}" "$@"
