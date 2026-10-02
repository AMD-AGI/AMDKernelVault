# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the synchronous driver with deferred CPU substitutes for Ray calls."""

import argparse
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from triton_rl import driver


class FakeReference:
    def __init__(self, event, result=None, error=None):
        self.event = event
        self.result = result
        self.error = error


class FakeRay:
    def __init__(self, events):
        self.events = events

    def get(self, reference):
        if isinstance(reference, list):
            return [self.get(item) for item in reference]
        if not isinstance(reference, FakeReference):
            raise AssertionError("The driver must resolve a deferred Ray reference.")
        if reference.event is not None:
            self.events.append(reference.event)
        if reference.error is not None:
            raise reference.error
        return reference.result


class FakeRuntime:
    weights_tag = "weights"
    kv_cache_tag = "kv_cache"

    def __init__(self):
        self.events = []
        self.ray = FakeRay(self.events)
        self.starts = [0, 0]
        self.per_epoch = 2
        self.fail_train_id = None
        self.init_args = None
        self.controller = SimpleNamespace(
            load=SimpleNamespace(remote=lambda index: self.ref("dataset_load", index)),
            save=SimpleNamespace(remote=lambda index: self.ref("dataset_save", index)),
            get_num_rollout_per_epoch=SimpleNamespace(
                remote=lambda: self.ref("epoch_count", result=self.per_epoch)
            ),
        )
        self.manager = SimpleNamespace(
            controller=self.controller,
            async_generate=lambda index: self.ref("generate", index, result=f"data:{index}"),
            async_eval=lambda index: self.ref("evaluate", index),
            async_offload=lambda: [self.ref("rollout_offload")],
            async_onload=lambda *, tags: [self.ref("rollout_onload", tuple(tags))],
        )
        self.actor = SimpleNamespace(
            async_init=self.init_actor,
            async_init_weight_update_connections=lambda manager: [self.ref("connect")],
            async_train=self.train_actor,
            async_save_model=lambda index: [self.ref("model_save", index)],
            async_update_weights=lambda: [self.ref("weight_sync")],
            async_offload=lambda: [self.ref("actor_offload")],
        )

    def ref(self, *event, result=None, error=None):
        return FakeReference(event, result, error)

    def init_actor(self, args, *, role, with_ref):
        self.init_args = args
        return [self.ref("actor_init", role, with_ref, result=self.starts[0])] + [
            FakeReference(None, result=value) for value in self.starts[1:]
        ]

    def train_actor(self, index, data):
        error = RuntimeError("The actor update failed.") if index == self.fail_train_id else None
        return [self.ref("train", index, data, error=error)]

    def create_placement_groups(self, args):
        self.events.append(("placements",))
        return {"actor": "actor-placement", "rollout": "rollout-placement"}

    def init_wandb_primary(self, args):
        self.events.append(("wandb",))
        return "run-id"

    def create_actor_group(self, args, placement, *, wandb_run_id):
        self.events.append(("create_actor", placement, wandb_run_id))
        return self.actor

    def create_rollout_manager(self, args, placement, *, wandb_run_id):
        self.events.append(("create_rollout", placement, wandb_run_id))
        return self.manager


def arguments(**overrides):
    values = dict(
        num_rollout=3,
        num_epoch=None,
        start_rollout_id=None,
        kl_coef=0.0,
        use_kl_loss=True,
        rollout_global_dataset=True,
        offload=False,
        save="/synthetic/checkpoints",
        save_interval=2,
        eval_interval=None,
        new_curriculum_phase=False,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        patcher = patch.object(driver, "_runtime", return_value=self.runtime)
        self.runtime_loader = patcher.start()
        self.addCleanup(patcher.stop)

    def selected(self, *names):
        return [event for event in self.runtime.events if event[0] in names]

    def test_on_policy_order_and_final_save_between_periodic_boundaries(self):
        driver.train(arguments(num_rollout=3, save_interval=2))
        self.assertEqual(
            self.selected("weight_sync", "generate", "train", "model_save", "dataset_save"),
            [
                ("weight_sync",),
                ("generate", 0),
                ("train", 0, "data:0"),
                ("weight_sync",),
                ("generate", 1),
                ("train", 1, "data:1"),
                ("model_save", 1),
                ("dataset_save", 1),
                ("weight_sync",),
                ("generate", 2),
                ("train", 2, "data:2"),
                ("model_save", 2),
                ("dataset_save", 2),
                ("weight_sync",),
            ],
        )

    def test_final_periodic_boundary_saves_once(self):
        driver.train(arguments(num_rollout=4, save_interval=2))
        self.assertEqual(self.selected("model_save"), [("model_save", 1), ("model_save", 3)])
        self.assertEqual(self.selected("dataset_save"), [("dataset_save", 1), ("dataset_save", 3)])

    def test_final_save_runs_without_a_periodic_save_interval(self):
        driver.train(arguments(num_rollout=2, save_interval=None))
        self.assertEqual(self.selected("model_save"), [("model_save", 1)])
        self.assertEqual(self.selected("dataset_save"), [("dataset_save", 1)])

    def test_amd_offload_and_evaluation_order(self):
        driver.train(arguments(num_rollout=1, offload=True, eval_interval=1))
        self.assertEqual(
            self.selected(
                "connect",
                "rollout_onload",
                "weight_sync",
                "evaluate",
                "generate",
                "rollout_offload",
                "train",
                "model_save",
                "dataset_save",
                "actor_offload",
            ),
            [
                ("connect",),
                ("rollout_onload", ("weights",)),
                ("weight_sync",),
                ("rollout_onload", ("kv_cache",)),
                ("evaluate", 0),
                ("generate", 0),
                ("rollout_offload",),
                ("train", 0, "data:0"),
                ("model_save", 0),
                ("dataset_save", 0),
                ("actor_offload",),
                ("rollout_onload", ("weights",)),
                ("weight_sync",),
                ("rollout_onload", ("kv_cache",)),
                ("evaluate", 0),
            ],
        )

    def test_normal_resume_restores_prompt_state_and_keeps_absolute_indices(self):
        self.runtime.starts = [8, 8]
        args = arguments(num_rollout=10, save_interval=None)
        driver.train(args)
        self.assertEqual(args.start_rollout_id, 8)
        self.assertEqual(self.selected("dataset_load"), [("dataset_load", 7)])
        self.assertEqual(self.selected("generate"), [("generate", 8), ("generate", 9)])
        self.assertEqual(self.selected("model_save"), [("model_save", 9)])
        self.assertEqual(self.selected("actor_init"), [("actor_init", "actor", True)])

    def test_new_phase_skips_only_prompt_restore(self):
        self.runtime.starts = [8, 8]
        args = arguments(num_rollout=10, save_interval=None, new_curriculum_phase=True)
        driver.train(args)
        self.assertEqual(args.start_rollout_id, 8)
        self.assertEqual(self.selected("dataset_load"), [])
        self.assertEqual(self.selected("generate"), [("generate", 8), ("generate", 9)])
        self.assertEqual(self.selected("dataset_save"), [("dataset_save", 9)])
        self.assertIs(self.runtime.init_args, args)
        self.assertFalse(args.no_load_optim)
        self.assertFalse(args.no_load_rng)
        self.assertFalse(args.finetune)
        self.assertEqual(self.selected("actor_init"), [("actor_init", "actor", True)])

    def test_failed_update_does_not_save_or_sync_partial_parameters(self):
        self.runtime.fail_train_id = 1
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            driver.train(arguments(num_rollout=2, save_interval=1))
        self.assertEqual(self.selected("model_save"), [("model_save", 0)])
        self.assertEqual(self.selected("dataset_save"), [("dataset_save", 0)])
        self.assertEqual(len(self.selected("weight_sync")), 2)

    def test_known_empty_budget_fails_before_runtime_allocation(self):
        for start in (3, 4):
            with self.subTest(start=start):
                with self.assertRaisesRegex(ValueError, "No training remains"):
                    driver.train(arguments(start_rollout_id=start, num_rollout=3))
        self.runtime_loader.assert_not_called()

    def test_resume_without_remaining_steps_fails_before_generation(self):
        self.runtime.starts = [3, 3]
        with self.assertRaisesRegex(ValueError, "No training remains"):
            driver.train(arguments(num_rollout=3))
        self.assertEqual(self.selected("dataset_load", "weight_sync", "generate", "model_save"), [])

    def test_actor_ranks_must_agree_on_the_resume_index(self):
        self.runtime.starts = [1, 2]
        with self.assertRaisesRegex(ValueError, "different rollout indices"):
            driver.train(arguments())
        self.assertEqual(self.selected("generate", "model_save"), [])

    def test_epoch_budget_keeps_upstream_epoch_boundaries(self):
        args = arguments(num_rollout=None, num_epoch=2, save_interval=99, eval_interval=99)
        driver.train(args)
        self.assertEqual(args.num_rollout, 4)
        self.assertEqual(self.selected("epoch_count"), [("epoch_count",)])
        self.assertEqual(self.selected("model_save"), [("model_save", 1), ("model_save", 3)])
        self.assertEqual(
            self.selected("evaluate"), [("evaluate", 0), ("evaluate", 1), ("evaluate", 3)]
        )

    def test_disabled_global_dataset_skips_controller_state_calls(self):
        driver.train(arguments(num_rollout=1, rollout_global_dataset=False))
        self.assertEqual(self.selected("dataset_load", "dataset_save"), [])
        self.assertEqual(self.selected("model_save"), [("model_save", 0)])

    def test_driver_requires_a_checkpoint_destination(self):
        with self.assertRaisesRegex(ValueError, "requires --save"):
            driver.train(arguments(save=None))
        self.runtime_loader.assert_not_called()

    def test_invalid_indices_and_intervals_fail_before_training(self):
        for override in (
            {"num_rollout": 0},
            {"num_rollout": True},
            {"start_rollout_id": -1},
            {"start_rollout_id": True},
            {"save_interval": 0},
            {"eval_interval": 0},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    driver.train(arguments(**override))
        self.runtime_loader.assert_not_called()


class DriverArgumentsTests(unittest.TestCase):
    def test_curriculum_argument_defaults_false_and_can_be_enabled(self):
        parser = argparse.ArgumentParser()
        self.assertIs(driver.add_custom_arguments(parser), parser)
        self.assertFalse(parser.parse_args([]).new_curriculum_phase)
        self.assertTrue(parser.parse_args(["--new-curriculum-phase"]).new_curriculum_phase)

    def test_main_uses_the_supported_slime_argument_extension(self):
        args = arguments()
        argument_module = ModuleType("slime.utils.arguments")
        argument_module.parse_args = Mock(return_value=args)
        modules = {
            "slime": ModuleType("slime"),
            "slime.utils": ModuleType("slime.utils"),
            "slime.utils.arguments": argument_module,
        }
        with patch.dict(sys.modules, modules), patch.object(driver, "train") as train:
            driver.main()
        argument_module.parse_args.assert_called_once_with(
            add_custom_arguments=driver.add_custom_arguments
        )
        train.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
