# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from THUDM/slime train.py under the Apache-2.0 license.
# Source revision: 5f781608ba28738fc73f44fa12efef1cdb408ee2.
# Source: https://github.com/THUDM/slime/blob/5f781608ba28738fc73f44fa12efef1cdb408ee2/train.py
"""Run Slime's synchronous loop with a final checkpoint and curriculum resume."""

from __future__ import annotations

from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Any


def _runtime() -> Any:
    """Import the GPU runtime only when training starts."""
    import ray
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
    from slime.ray.placement_group import (
        create_actor_group,
        create_placement_groups,
        create_rollout_manager,
    )
    from slime.utils.wandb_utils import init_wandb_primary

    return SimpleNamespace(
        ray=ray,
        weights_tag=GPU_MEMORY_TYPE_WEIGHTS,
        kv_cache_tag=GPU_MEMORY_TYPE_KV_CACHE,
        create_actor_group=create_actor_group,
        create_placement_groups=create_placement_groups,
        create_rollout_manager=create_rollout_manager,
        init_wandb_primary=init_wandb_primary,
    )


def add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--new-curriculum-phase",
        action="store_true",
        help="Start the new prompt dataset without restoring the previous dataset state.",
    )
    return parser


def _validate_budget(args: Any) -> None:
    if args.num_rollout is not None and (type(args.num_rollout) is not int or args.num_rollout < 1):
        raise ValueError("The absolute num_rollout stop index must be a positive integer.")
    if args.start_rollout_id is not None and (
        type(args.start_rollout_id) is not int or args.start_rollout_id < 0
    ):
        raise ValueError("The start_rollout_id must be a nonnegative integer.")
    if (
        args.num_rollout is not None
        and args.start_rollout_id is not None
        and args.start_rollout_id >= args.num_rollout
    ):
        raise ValueError(
            f"No training remains: start_rollout_id={args.start_rollout_id} "
            f"must be less than num_rollout={args.num_rollout}."
        )


def _interval_due(rollout_id: int, interval: int | None, per_epoch: int | None) -> bool:
    return interval is not None and (
        (rollout_id + 1) % interval == 0
        or (per_epoch is not None and (rollout_id + 1) % per_epoch == 0)
    )


def train(args: Any) -> None:
    """Wait for each update before the next rollout, as the pinned Slime driver does."""
    _validate_budget(args)
    if not getattr(args, "save", None):
        raise ValueError("The synchronous driver requires --save for the final checkpoint.")
    for name in ("save_interval", "eval_interval"):
        value = getattr(args, name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"The {name} must be a positive integer or None.")

    runtime = _runtime()
    ray = runtime.ray
    pgs = runtime.create_placement_groups(args)
    wandb_run_id = runtime.init_wandb_primary(args)
    actor_model = runtime.create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)
    rollout_manager = runtime.create_rollout_manager(
        args, pgs["rollout"], wandb_run_id=wandb_run_id
    )

    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(
            rollout_manager.controller.get_num_rollout_per_epoch.remote()
        )
        if type(num_rollout_per_epoch) is not int or num_rollout_per_epoch < 1:
            raise ValueError("The dataset must provide a positive integer rollout count per epoch.")
        if type(args.num_epoch) is not int or args.num_epoch < 1:
            raise ValueError("The num_epoch must be a positive integer.")
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        _validate_budget(args)

    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    if not start_rollout_ids or any(
        type(index) is not int or index < 0 for index in start_rollout_ids
    ):
        raise ValueError("Actor ranks must return nonnegative integer rollout indices.")
    if len(set(start_rollout_ids)) != 1:
        raise ValueError("Actor ranks returned different rollout indices.")
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]
    _validate_budget(args)

    if args.rollout_global_dataset and not getattr(args, "new_curriculum_phase", False):
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))

    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))
    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[runtime.weights_tag]))
    ray.get(actor_model.async_update_weights())
    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[runtime.kv_cache_tag]))

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.async_eval(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id))
        if args.offload:
            ray.get(rollout_manager.async_offload())
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Save while the completed actor update remains resident on the GPUs.
        final_update = rollout_id + 1 == args.num_rollout
        if final_update or _interval_due(rollout_id, args.save_interval, num_rollout_per_epoch):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.controller.save.remote(rollout_id))

        if args.offload:
            ray.get(actor_model.async_offload())
            ray.get(rollout_manager.async_onload(tags=[runtime.weights_tag]))
        ray.get(actor_model.async_update_weights())
        if args.offload:
            ray.get(rollout_manager.async_onload(tags=[runtime.kv_cache_tag]))

        if _interval_due(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.async_eval(rollout_id))


def main() -> None:
    from slime.utils.arguments import parse_args

    train(parse_args(add_custom_arguments=add_custom_arguments))


if __name__ == "__main__":
    main()
