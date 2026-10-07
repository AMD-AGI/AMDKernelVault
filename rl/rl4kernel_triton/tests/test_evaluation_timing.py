# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check timing protocol behavior with a fake timer and no GPU execution."""

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from triton_rl.evaluation import timing


class TimingTests(unittest.TestCase):
    def setUp(self):
        timing.PYTEST_BENCHMARK_RESULTS.clear()
        self.timer = Mock(return_value=[1.234567, 1.456789, 1.012345])
        triton = types.ModuleType("triton")
        testing = types.ModuleType("triton.testing")
        testing.do_bench = self.timer
        triton.testing = testing
        self.modules = patch.dict(sys.modules, {"triton": triton, "triton.testing": testing})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(timing.PYTEST_BENCHMARK_RESULTS.clear)

    def test_default_protocol_budgets_rounding_and_diagnostics(self):
        op = Mock()
        params = {"shape": [16, 32], "dtype": "float16"}
        gbps = Mock(return_value=1234.567)
        tflops = Mock(return_value=34.567)
        result = timing.PytestBenchmarker(op, "test_op").run_benchmark(params, gbps, tflops)

        self.timer.assert_called_once_with(
            op, warmup=25, rep=100, quantiles=[0.5, 0.8, 0.2], return_mode="median"
        )
        self.assertEqual(result["params"], params)
        self.assertEqual(result["ms"], 1.2346)
        self.assertEqual(result["min_ms"], 1.4568)
        self.assertEqual(result["max_ms"], 1.0123)
        self.assertEqual(result["GB/s"], 1234.57)
        self.assertEqual(result["TFLOPS"], 34.57)
        self.assertEqual(result["median_ms"], 1.234567)
        self.assertEqual(result["p80_ms"], 1.456789)
        self.assertEqual(result["p20_ms"], 1.012345)
        self.assertEqual(result["timing_config"]["budget_unit"], "ms")
        self.assertEqual(result["timing_config"]["timer"], "triton.testing.do_bench")
        gbps.assert_called_once_with(params, 1.234567)
        self.assertIs(gbps.call_args.args[0], params)
        tflops.assert_called_once_with(params, 1.234567)
        self.assertEqual(timing.PYTEST_BENCHMARK_RESULTS["test_op"], [result])

    def test_collector_protects_records_from_later_caller_changes(self):
        params = {"shape": [16, 32]}
        result = timing.PytestBenchmarker(lambda: None, "op").run_benchmark(params)
        params["shape"][0] = 64
        result["params"]["shape"][1] = 128
        collected = timing.PYTEST_BENCHMARK_RESULTS["op"][0]
        self.assertEqual(collected["params"], {"shape": [16, 32]})

    def test_custom_budgets_and_reordered_quantiles(self):
        config = timing.do_bench_config(0, 200, [0.2, 0.5, 0.8], "mean")
        self.timer.return_value = [1.0, 2.0, 3.0]
        benchmarker = timing.PytestBenchmarker(lambda: None, "op", config)
        result = benchmarker.run_benchmark({})
        self.assertEqual((result["ms"], result["min_ms"], result["max_ms"]), (1.0, 2.0, 3.0))
        self.assertEqual((result["median_ms"], result["p80_ms"], result["p20_ms"]), (2.0, 3.0, 1.0))
        self.assertEqual(self.timer.call_args.kwargs["rep"], 200)
        self.assertEqual(result["timing_config"]["return_mode"], "mean")

    def test_configs_do_not_share_the_quantile_list(self):
        first = timing.do_bench_config()
        second = timing.do_bench_config()
        first.quantiles.reverse()
        self.assertEqual(second.quantiles, [0.5, 0.8, 0.2])
        self.assertEqual(timing.do_bench_config(quantiles=None).quantiles, second.quantiles)

    def test_other_quantiles_preserve_positions_without_false_percentile_fields(self):
        config = timing.do_bench_config(quantiles=[0.1, 0.9, 0.5])
        self.timer.return_value = [1.0, 3.0, 2.0]
        calculator = Mock(return_value=4.0)
        result = timing.PytestBenchmarker(lambda: None, "op", config).run_benchmark({}, calculator)
        self.assertEqual((result["ms"], result["min_ms"], result["max_ms"]), (1.0, 3.0, 2.0))
        self.assertEqual(result["median_ms"], 2.0)
        self.assertNotIn("p80_ms", result)
        self.assertNotIn("p20_ms", result)
        calculator.assert_called_once_with({}, 1.0)

    def test_invalid_configuration_fails_before_measurement(self):
        invalid = (
            {"warm_up": -1},
            {"warm_up": True},
            {"warm_up": math.inf},
            {"repetition": 0},
            {"repetition": math.nan},
            {"repetition": "100"},
            {"quantiles": [0.5]},
            {"quantiles": [0.5, math.nan, 0.2]},
            {"quantiles": [0.5, 1.1, 0.1]},
            {"quantiles": "0.5"},
            {"return_mode": "unknown"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                timing.do_bench_config(**kwargs)
        self.timer.assert_not_called()

    def test_mutated_configuration_produces_an_error_record(self):
        benchmarker = timing.PytestBenchmarker(lambda: None, "op")
        benchmarker.config.repetition = 0
        result = benchmarker.run_benchmark({"case": 1})
        self.assertIn("error", result)
        self.assertNotIn("ms", result)
        self.timer.assert_not_called()

    def test_invalid_timer_values_produce_explicit_error_records(self):
        invalid = (
            [0, 2, 1],
            [-1, 2, 1],
            [math.nan, 2, 1],
            [math.inf, 2, 1],
            [True, 2, 1],
            ["1", 2, 1],
            [1, 2],
            [1, 2, 3],
            [0.00001, 0.00002, 0.000001],
            1.0,
        )
        benchmarker = timing.PytestBenchmarker(lambda: None, "op")
        for index, values in enumerate(invalid):
            with self.subTest(values=values):
                self.timer.return_value = values
                result = benchmarker.run_benchmark({"case": index})
                self.assertEqual(result["params"], {"case": index})
                self.assertIn("error", result)
                self.assertNotIn("ms", result)
        self.assertEqual(len(timing.PYTEST_BENCHMARK_RESULTS["op"]), len(invalid))

    def test_timer_exception_stays_in_saved_results(self):
        self.timer.side_effect = RuntimeError("The timer failed.")
        result = timing.PytestBenchmarker(lambda: None, "op").run_benchmark({"case": 1})
        self.assertEqual(
            result, {"params": {"case": 1}, "error": "RuntimeError: The timer failed."}
        )
        with tempfile.TemporaryDirectory() as directory:
            timing.save_all_benchmark_results(directory)
            saved = json.loads((Path(directory) / "op.json").read_text())
        self.assertEqual(saved, [result])

    def test_missing_and_failed_calculators_use_na(self):
        benchmarker = timing.PytestBenchmarker(lambda: None, "op")
        for calculator in (
            None,
            Mock(side_effect=RuntimeError("failed")),
            Mock(return_value=math.nan),
            Mock(return_value=True),
            Mock(return_value=-1),
            Mock(return_value="N/A"),
        ):
            with self.subTest(calculator=calculator):
                result = benchmarker.run_benchmark({}, calculator, calculator)
                self.assertEqual(result["GB/s"], "N/A")
                self.assertEqual(result["TFLOPS"], "N/A")
                self.assertIn("ms", result)

    def test_save_writes_operator_lists_and_clears_records(self):
        timing.add_benchmark_result("op_a", {"params": {"case": 1}, "ms": 1})
        timing.add_benchmark_result("op_b", {"params": {"case": 2}, "ms": 2})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "results"
            timing.save_all_benchmark_results(output)
            self.assertEqual(json.loads((output / "op_a.json").read_text())[0]["ms"], 1)
            self.assertEqual(json.loads((output / "op_b.json").read_text())[0]["ms"], 2)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()), ["op_a.json", "op_b.json"]
            )
        self.assertEqual(timing.PYTEST_BENCHMARK_RESULTS, {})

    def test_torch_dtype_and_device_parameters_are_strings_in_json(self):
        class Dtype:
            def __str__(self):
                return "torch.float16"

        class Device:
            def __str__(self):
                return "cuda:0"

        torch = types.ModuleType("torch")
        torch.dtype = Dtype
        torch.device = Device
        timing.add_benchmark_result("op", {"params": {"dtype": Dtype(), "device": Device()}})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": torch}):
            timing.save_all_benchmark_results(directory)
            saved = json.loads((Path(directory) / "op.json").read_text())
        self.assertEqual(saved[0]["params"], {"dtype": "torch.float16", "device": "cuda:0"})

    def test_save_error_raises_and_preserves_collector(self):
        timing.add_benchmark_result("op", {"params": {}, "ms": 1})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "op.json"
            output.write_text("previous data")
            with patch.object(timing.os, "replace", side_effect=OSError("The write failed.")):
                with self.assertRaisesRegex(OSError, "The write failed"):
                    timing.save_all_benchmark_results(directory)
            self.assertEqual(output.read_text(), "previous data")
            self.assertEqual(list(Path(directory).iterdir()), [output])
        self.assertEqual(len(timing.PYTEST_BENCHMARK_RESULTS["op"]), 1)

    def test_serialization_failure_preserves_all_records_and_existing_files(self):
        timing.add_benchmark_result("op_a", {"params": {}, "ms": 1})
        timing.add_benchmark_result("op_b", {"params": {}, "ms": math.nan})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "op_a.json"
            output.write_text("previous data")
            with self.assertRaises(ValueError):
                timing.save_all_benchmark_results(directory)
            self.assertEqual(output.read_text(), "previous data")
        self.assertEqual(len(timing.PYTEST_BENCHMARK_RESULTS), 2)

    def test_invalid_operator_names_cannot_escape_output_directory(self):
        for name in ("", ".", "..", "../op", "/tmp/op", "a/b", "a\\b", "op\x00", "op\n"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                timing.add_benchmark_result(name, {})
        self.assertEqual(timing.PYTEST_BENCHMARK_RESULTS, {})

    def test_invalid_output_paths_preserve_collector(self):
        timing.add_benchmark_result("op", {"params": {}, "ms": 1})
        with tempfile.TemporaryDirectory() as directory:
            regular_file = Path(directory) / "file"
            regular_file.write_text("file")
            link = Path(directory) / "link"
            link.symlink_to(Path(directory), target_is_directory=True)
            for output in (
                "",
                " ",
                "bad\x00path",
                regular_file,
                link,
                Path(directory) / ".." / "escape",
            ):
                with self.subTest(output=output), self.assertRaises((ValueError, OSError)):
                    timing.save_all_benchmark_results(output)
        self.assertEqual(len(timing.PYTEST_BENCHMARK_RESULTS["op"]), 1)

    def test_atomic_save_replaces_destination_symlink_without_following_it(self):
        timing.add_benchmark_result("op", {"params": {}, "ms": 1})
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "outside.json"
            original.write_text("original")
            output = Path(directory) / "results"
            output.mkdir()
            (output / "op.json").symlink_to(original)
            timing.save_all_benchmark_results(output)
            self.assertEqual(original.read_text(), "original")
            self.assertFalse((output / "op.json").is_symlink())


if __name__ == "__main__":
    unittest.main()
