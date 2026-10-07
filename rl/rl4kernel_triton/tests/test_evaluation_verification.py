# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check saved-output comparison and complete timing coverage on the CPU."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from triton_rl.evaluation.verification import (
    CandidateOutputError,
    CandidateTimingError,
    ReferenceError,
    aggregate_timings,
    compare_outputs,
)

try:
    import torch
except ImportError:
    torch = None


class ImportTests(unittest.TestCase):
    def test_module_import_does_not_import_torch(self):
        command = (
            "import sys; import triton_rl.evaluation.verification; "
            "assert 'torch' not in sys.modules"
        )
        completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipIf(torch is None, "Output tests require the optional PyTorch dependency.")
class OutputTests(unittest.TestCase):
    def saved(self, output, flag=True):
        return {"_CALL_SUCCESS_": torch.tensor(flag), "result": output}

    def compare(self, expected, actual, **tolerances):
        return compare_outputs(
            self.saved(expected),
            self.saved(actual),
            atol=tolerances.get("atol", 0),
            rtol=tolerances.get("rtol", 0),
        )

    def test_nested_outputs_match_without_mutating_saved_dictionaries(self):
        reference = self.saved(
            {
                "tensor": torch.tensor([1.0, 2.0]),
                "nested": [3, (4.0, True, "label", None)],
            }
        )
        candidate = self.saved(
            {
                "nested": [3, (4.0, True, "label", None)],
                "tensor": torch.tensor([1.0, 2.0]),
            }
        )
        matched, feedback = compare_outputs(reference, candidate, atol=0, rtol=0)
        self.assertTrue(matched)
        self.assertIn("6 output leaves", feedback)
        self.assertIn("_CALL_SUCCESS_", reference)
        self.assertIn("_CALL_SUCCESS_", candidate)

    def test_reference_controls_relative_tolerance(self):
        expected, actual = torch.tensor(100.0), torch.tensor(111.0)
        matched, feedback = self.compare(expected, actual, rtol=0.1)
        self.assertFalse(matched)
        self.assertIn("outputs['result']", feedback)
        self.assertTrue(self.compare(expected, actual, rtol=0.12)[0])

    def test_comparison_keeps_caller_absolute_tolerance(self):
        expected, actual = torch.tensor(1.0), torch.tensor(1.005)
        self.assertFalse(self.compare(expected, actual, atol=0.0001)[0])
        self.assertTrue(self.compare(expected, actual, atol=0.006)[0])

    def test_nan_does_not_equal_nan(self):
        for value in [float("nan"), torch.tensor(float("nan"))]:
            with self.subTest(value=value):
                self.assertFalse(self.compare(value, value)[0])

    def test_structure_tensor_properties_and_scalar_types_must_match(self):
        pairs = [
            ({"a": 1}, {"b": 1}),
            ([1], (1,)),
            ([1], [1, 2]),
            (torch.ones(2), torch.ones(1, 2)),
            (torch.tensor([1], dtype=torch.int32), torch.tensor([1], dtype=torch.int64)),
            (torch.tensor(1), 1),
            (1, torch.tensor(1)),
            (True, 1),
            (1, 1.0),
            ("same", "different"),
            (None, "None"),
        ]
        for expected, actual in pairs:
            with self.subTest(expected=expected, actual=actual):
                self.assertFalse(self.compare(expected, actual)[0])

    def test_scalar_failure_rejects_matching_tensors(self):
        self.assertFalse(self.compare([torch.ones(2), 1], [torch.ones(2), 2])[0])

    def test_non_tensor_scalars_require_exact_equality_despite_tolerance(self):
        self.assertFalse(self.compare(1.0, 1.005, atol=1, rtol=1)[0])
        self.assertFalse(self.compare(1 + 0j, 1 + 0.005j, atol=1, rtol=1)[0])

    def test_scalar_boolean_and_binary_numeric_flags_are_valid(self):
        for flag in [True, 1, 1.0, [True], [[1]]]:
            with self.subTest(flag=flag):
                reference = self.saved(1, flag)
                candidate = self.saved(1, flag)
                self.assertTrue(compare_outputs(reference, candidate, atol=0, rtol=0)[0])

    def test_invalid_reference_flag_raises_infrastructure_error(self):
        flags = [
            None,
            True,
            torch.tensor(False),
            torch.tensor(0),
            torch.tensor(2),
            torch.tensor(float("nan")),
            torch.tensor(float("inf")),
            torch.tensor([True, True]),
            torch.tensor(1 + 0j),
        ]
        for flag in flags:
            with self.subTest(flag=flag):
                reference = {"_CALL_SUCCESS_": flag, "result": 1}
                with self.assertRaises(ReferenceError):
                    compare_outputs(reference, self.saved(1), atol=0, rtol=0)
        with self.assertRaises(ReferenceError):
            compare_outputs({"result": 1}, self.saved(1), atol=0, rtol=0)

    def test_invalid_candidate_flag_has_no_compilation_evidence(self):
        flags = [
            None,
            True,
            torch.tensor(False),
            torch.tensor(0.0),
            torch.tensor(-1),
            torch.tensor(float("nan")),
            torch.tensor([1, 1]),
        ]
        for flag in flags:
            with self.subTest(flag=flag):
                with self.assertRaises(CandidateOutputError) as raised:
                    compare_outputs(
                        self.saved(1), {"_CALL_SUCCESS_": flag, "result": 1}, atol=0, rtol=0
                    )
                self.assertFalse(raised.exception.compiled)
                self.assertEqual(raised.exception.stage, "output_flag")

    def test_empty_output_trees_fail_after_removing_flags(self):
        for output in [
            {},
            [],
            (),
            {"nested": [(), {}]},
            torch.empty(0),
            {"empty": torch.empty(0, 2)},
        ]:
            with self.subTest(output=output):
                with self.assertRaises(ReferenceError):
                    compare_outputs(self.saved(output), self.saved(1), atol=0, rtol=0)
                with self.assertRaises(CandidateOutputError) as raised:
                    compare_outputs(self.saved(1), self.saved(output), atol=0, rtol=0)
                self.assertTrue(raised.exception.compiled)
                self.assertEqual(raised.exception.stage, "output_tree")
        with self.assertRaises(ReferenceError):
            compare_outputs({"_CALL_SUCCESS_": torch.tensor(True)}, self.saved(1), atol=0, rtol=0)

    def test_scalar_and_nonempty_tensor_supply_leaves(self):
        self.assertTrue(self.compare(None, None)[0])
        self.assertTrue(
            self.compare([torch.empty(0), torch.ones(1)], [torch.empty(0), torch.ones(1)])[0]
        )

    def test_cycles_and_unsupported_reference_values_are_infrastructure_errors(self):
        cycle = []
        cycle.append(cycle)
        for output in [cycle, {"set": {1}}]:
            with self.subTest(output_type=type(output).__name__):
                with self.assertRaises(ReferenceError):
                    compare_outputs(self.saved(output), self.saved(1), atol=0, rtol=0)
                with self.assertRaises(CandidateOutputError) as raised:
                    compare_outputs(self.saved(1), self.saved(output), atol=0, rtol=0)
                self.assertTrue(raised.exception.compiled)

    def test_reference_validation_precedes_candidate_failure(self):
        with self.assertRaises(ReferenceError):
            compare_outputs(self.saved({}), self.saved(1, False), atol=0, rtol=0)

    def test_tolerances_must_be_finite_nonnegative_numbers(self):
        for value in [True, "0.1", None, -1, math.nan, math.inf]:
            for argument in ["atol", "rtol"]:
                with self.subTest(value=value, argument=argument):
                    kwargs = {"atol": 0, "rtol": 0, argument: value}
                    with self.assertRaises(ValueError):
                        compare_outputs(self.saved(1), self.saved(1), **kwargs)


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.reference = Path(self.temporary.name) / "reference"
        self.candidate = Path(self.temporary.name) / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()

    def save(self, folder, records, name="operator.json"):
        (folder / name).write_text(json.dumps(records), encoding="utf-8")

    def records(self, *times):
        return [{"params": {"case": index}, "ms": value} for index, value in enumerate(times)]

    def test_task_speedup_uses_equal_weight_for_operator_ratios(self):
        self.save(self.reference, self.records(2, 8), "first.json")
        self.save(self.candidate, self.records(1, 2), "first.json")
        self.save(self.reference, self.records(6), "second.json")
        self.save(self.candidate, self.records(5), "second.json")
        summary = aggregate_timings(self.reference, self.candidate)
        self.assertAlmostEqual(summary.speedup, (10 / 3 + 6 / 5) / 2)
        self.assertNotEqual(summary.speedup, summary.baseline_latency_ms / summary.latency_ms)
        self.assertEqual(summary.latency_ms, 8)
        self.assertEqual(summary.baseline_latency_ms, 16)
        self.assertEqual(summary.diagnostics["case_count"], 3)
        self.assertEqual(summary.diagnostics["operator_count"], 2)
        self.assertEqual(summary.diagnostics["aggregation"], "arithmetic_mean(operator_speedups)")
        self.assertEqual(
            summary.diagnostics["operator_aggregation"], "sum(reference_ms) / sum(candidate_ms)"
        )
        first = summary.diagnostics["operators"]["first.json"]
        self.assertEqual(first["case_count"], 2)
        self.assertAlmostEqual(first["speedup"], 10 / 3)
        self.assertEqual(first["rounded_speedup"], 3.3333)
        self.assertEqual(first["latency_ms"], 3)
        self.assertEqual(
            first["cases"],
            [
                {"params": {"case": 0}, "reference_ms": 2, "candidate_ms": 1},
                {"params": {"case": 1}, "reference_ms": 8, "candidate_ms": 2},
            ],
        )

    def test_display_rounding_does_not_change_task_speedup(self):
        self.save(self.reference, self.records(1), "normal.json")
        self.save(self.candidate, self.records(3), "normal.json")
        self.save(self.reference, self.records(1), "slow.json")
        self.save(self.candidate, self.records(100000), "slow.json")
        summary = aggregate_timings(self.reference, self.candidate)
        self.assertEqual(summary.diagnostics["operators"]["slow.json"]["rounded_speedup"], 0)
        self.assertEqual(summary.diagnostics["operators"]["slow.json"]["speedup"], 0.00001)
        self.assertAlmostEqual(summary.speedup, (1 / 3 + 0.00001) / 2)
        self.assertNotEqual(summary.speedup, (0.3333 + 0) / 2)

    def test_actual_timing_settings_survive_aggregation(self):
        config = {"warm_up": 25, "repetition": 100, "quantiles": [0.5, 0.8, 0.2]}
        expected = [{"params": {}, "ms": 2, "median_ms": 2.000001, "timing_config": config}]
        actual = [{"params": {}, "ms": 1, "median_ms": 1.000001, "timing_config": config}]
        self.save(self.reference, expected)
        self.save(self.candidate, actual)
        summary = aggregate_timings(self.reference, self.candidate)
        case = summary.diagnostics["operators"]["operator.json"]["cases"][0]
        self.assertEqual(case["reference_timing"]["timing_config"], config)
        self.assertEqual(case["candidate_timing"]["median_ms"], 1.000001)
        actual[0]["timing_config"] = {**config, "repetition": 200}
        self.save(self.candidate, actual)
        with self.assertRaises(CandidateTimingError):
            aggregate_timings(self.reference, self.candidate)

    def test_rows_and_parameter_key_order_do_not_change_matches(self):
        expected = [
            {
                "params": {"shape": [2, 3], "dtype": "float32", "nested": {"a": 1, "b": False}},
                "ms": 2,
            },
            {"params": {"shape": [3, 4], "dtype": "float32"}, "ms": 8},
        ]
        actual = [
            {"params": {"dtype": "float32", "shape": [3, 4]}, "ms": 2},
            {
                "params": {"nested": {"b": False, "a": 1}, "dtype": "float32", "shape": [2, 3]},
                "ms": 1,
            },
        ]
        self.save(self.reference, expected)
        self.save(self.candidate, actual)
        summary = aggregate_timings(self.reference, self.candidate)
        self.assertAlmostEqual(summary.speedup, 10 / 3)
        cases = summary.diagnostics["operators"]["operator.json"]["cases"]
        self.assertEqual(
            [case["params"] for case in cases], [record["params"] for record in expected]
        )
        self.assertEqual([case["candidate_ms"] for case in cases], [1, 2])

    def test_severe_regressions_and_small_positive_times_remain_included(self):
        self.save(self.reference, self.records(1e-8))
        self.save(self.candidate, self.records(2e-7))
        summary = aggregate_timings(self.reference, self.candidate)
        self.assertAlmostEqual(summary.speedup, 0.05)
        self.assertEqual(summary.latency_ms, 2e-7)

    def test_missing_or_extra_operator_rejects_complete_aggregate(self):
        self.save(self.reference, self.records(2), "shared.json")
        self.save(self.candidate, self.records(1), "shared.json")
        for folder in [self.reference, self.candidate]:
            with self.subTest(folder=folder.name):
                self.save(folder, self.records(1), "unmatched.json")
                with self.assertRaisesRegex(CandidateTimingError, "unmatched.json"):
                    aggregate_timings(self.reference, self.candidate)
                (folder / "unmatched.json").unlink()

    def test_missing_extra_or_changed_case_rejects_complete_aggregate(self):
        self.save(self.reference, self.records(2, 8))
        for actual in [
            self.records(1),
            self.records(1, 2, 3),
            [{"params": {"case": 0}, "ms": 1}, {"params": {"case": 99}, "ms": 2}],
        ]:
            with self.subTest(actual=actual):
                self.save(self.candidate, actual)
                with self.assertRaisesRegex(CandidateTimingError, "parameters differ"):
                    aggregate_timings(self.reference, self.candidate)

    def test_duplicate_params_reject_both_inputs_in_either_order(self):
        for reference in [True, False]:
            folder = self.reference if reference else self.candidate
            error = ReferenceError if reference else CandidateTimingError
            for times in [(1, 2), (2, 1)]:
                with self.subTest(reference=reference, times=times):
                    self.save(self.reference, self.records(2))
                    self.save(self.candidate, self.records(1))
                    self.save(folder, [{"params": {"n": 1}, "ms": value} for value in times])
                    with self.assertRaisesRegex(error, "duplicates"):
                        aggregate_timings(self.reference, self.candidate)

    def test_error_records_reject_both_inputs(self):
        for reference in [True, False]:
            with self.subTest(reference=reference):
                self.save(self.reference, self.records(2, 8))
                self.save(self.candidate, self.records(1, 2))
                folder = self.reference if reference else self.candidate
                self.save(folder, [self.records(1)[0], {"params": {"case": 1}, "error": "failed"}])
                error = ReferenceError if reference else CandidateTimingError
                with self.assertRaisesRegex(error, "error record"):
                    aggregate_timings(self.reference, self.candidate)

    def test_nonpositive_nonfinite_and_nonnumeric_times_reject_both_inputs(self):
        values = [True, False, 0, -1, None, "1", math.nan, math.inf, -math.inf, 10**400]
        for reference in [True, False]:
            folder = self.reference if reference else self.candidate
            error = ReferenceError if reference else CandidateTimingError
            for value in values:
                with self.subTest(reference=reference, value=value):
                    self.save(self.reference, self.records(2))
                    self.save(self.candidate, self.records(1))
                    self.save(folder, self.records(value))
                    with self.assertRaises(error):
                        aggregate_timings(self.reference, self.candidate)

    def test_invalid_records_reject_both_inputs(self):
        records = [
            [],
            {},
            [None],
            [{}],
            [{"params": [], "ms": 1}],
            [{"params": {}}],
            [{"params": {"n": math.nan}, "ms": 1}],
        ]
        for reference in [True, False]:
            folder = self.reference if reference else self.candidate
            error = ReferenceError if reference else CandidateTimingError
            for record in records:
                with self.subTest(reference=reference, record=record):
                    self.save(self.reference, self.records(2))
                    self.save(self.candidate, self.records(1))
                    self.save(folder, record)
                    with self.assertRaises(error):
                        aggregate_timings(self.reference, self.candidate)

    def test_invalid_json_and_duplicate_object_keys_reject_both_inputs(self):
        for reference in [True, False]:
            folder = self.reference if reference else self.candidate
            error = ReferenceError if reference else CandidateTimingError
            for text in ["not JSON", '[{"params":{"n":1,"n":2},"ms":1}]']:
                with self.subTest(reference=reference, text=text):
                    self.save(self.reference, self.records(2))
                    self.save(self.candidate, self.records(1))
                    (folder / "operator.json").write_text(text, encoding="utf-8")
                    with self.assertRaises(error):
                        aggregate_timings(self.reference, self.candidate)

    def test_legacy_aggregate_ms_field_is_not_a_timing_input(self):
        self.save(self.reference, self.records(2))
        self.save(self.candidate, self.records(1))
        self.save(self.candidate, {"operator.json": {"ms": 100}}, "all_perf_results.json")
        with self.assertRaisesRegex(CandidateTimingError, "per-case"):
            aggregate_timings(self.reference, self.candidate)

    def test_empty_and_missing_folders_are_explicit_errors(self):
        with self.assertRaises(ReferenceError):
            aggregate_timings(self.reference, self.candidate)
        self.save(self.reference, self.records(2))
        with self.assertRaises(CandidateTimingError):
            aggregate_timings(self.reference, self.candidate)
        with self.assertRaises(ReferenceError):
            aggregate_timings(self.reference / "missing", self.candidate)
        with self.assertRaises(CandidateTimingError):
            aggregate_timings(self.reference, self.candidate / "missing")

    def test_reference_failure_precedes_candidate_failure(self):
        self.save(self.reference, self.records(-1))
        with self.assertRaises(ReferenceError):
            aggregate_timings(self.reference, self.candidate)

    def test_summed_latency_must_remain_finite(self):
        for reference in [True, False]:
            with self.subTest(reference=reference):
                self.save(self.reference, self.records(1, 1))
                self.save(self.candidate, self.records(1, 1))
                self.save(
                    self.reference if reference else self.candidate, self.records(1e308, 1e308)
                )
                error = ReferenceError if reference else CandidateTimingError
                with self.assertRaisesRegex(error, "summed latency"):
                    aggregate_timings(self.reference, self.candidate)


if __name__ == "__main__":
    unittest.main()
