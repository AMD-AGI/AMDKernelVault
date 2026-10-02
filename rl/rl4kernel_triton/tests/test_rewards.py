# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the reward hierarchy and the paper's unnormalized return."""

import math
import unittest
from types import SimpleNamespace

from triton_rl.contracts import EvaluationResult
from triton_rl.rewards import discounted_return, reward_func, turn_reward


class RewardTests(unittest.TestCase):
    def test_compilation_and_correctness_gate_reward(self):
        self.assertEqual(turn_reward(EvaluationResult(False, False, speedup=8)), 0.0)
        self.assertEqual(turn_reward(EvaluationResult(True, False, speedup=8)), 0.4)
        self.assertEqual(turn_reward(EvaluationResult(True, True)), 1.4)

    def test_paper_speed_reward_points_and_cap(self):
        for speedup, expected in [(1, 1.4), (2, 1.775), (4, 2.9), (16, 2.9), (1e300, 2.9)]:
            with self.subTest(speedup=speedup):
                self.assertAlmostEqual(
                    turn_reward(EvaluationResult(True, True, speedup=speedup)), expected
                )

    def test_slowdowns_and_invalid_values_receive_no_speed_bonus(self):
        for speedup in [0.5, 0, -2, math.nan, math.inf, -math.inf, None, "2", True, 10**400]:
            with self.subTest(speedup=speedup):
                self.assertEqual(turn_reward(EvaluationResult(True, True, speedup=speedup)), 1.4)

    def test_latency_is_never_interpreted_as_speedup(self):
        result = EvaluationResult(True, True, latency_ms=16.0, baseline_latency_ms=32.0)
        self.assertEqual(turn_reward(result), 1.4)

    def test_return_is_an_unnormalized_sum(self):
        self.assertAlmostEqual(discounted_return([0.4, 1.775, 2.9]), 0.4 + 0.6 * 1.775 + 0.36 * 2.9)
        self.assertAlmostEqual(discounted_return([2.9] * 3), 5.684)
        self.assertAlmostEqual(discounted_return([1.4] * 3), 2.744)
        self.assertEqual(discounted_return([]), 0.0)
        self.assertEqual(discounted_return([0.4]), 0.4)

    def test_return_rejects_invalid_numbers(self):
        for discount in [True, "0.6", math.nan, math.inf, -1, 1.1]:
            with self.subTest(discount=discount):
                with self.assertRaises(ValueError):
                    discounted_return([1.0], discount)
        for reward in [True, "1", math.nan, math.inf]:
            with self.subTest(reward=reward):
                with self.assertRaises(ValueError):
                    discounted_return([reward])


class RewardHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_hook_returns_stored_trajectory_reward(self):
        sample = SimpleNamespace(metadata={"triton_rl": {"trajectory_return": 2.509}}, reward=99.0)
        self.assertEqual(await reward_func(None, sample), 2.509)

    async def test_hook_rejects_missing_or_invalid_reward(self):
        for metadata in [{}, {"triton_rl": {}}, {"triton_rl": {"trajectory_return": math.nan}}]:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    await reward_func(None, SimpleNamespace(metadata=metadata))


if __name__ == "__main__":
    unittest.main()
