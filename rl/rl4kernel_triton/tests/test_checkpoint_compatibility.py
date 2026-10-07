# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the narrow checkpoint metadata patch without importing Megatron."""

from __future__ import annotations

import ast
import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_megatron_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_compatibility_patch", SCRIPT)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

FIXTURE = '''"""Synthetic checkpoint reader. No Megatron implementation is included."""
def unrelated(iteration, release):
    assert iteration > 0 or release

def read_metadata(tracker_filename):
    iteration = 0
    release = False
    with open(tracker_filename) as stream:
        marker = stream.read().strip()
    try:
        iteration = int(marker)
    except ValueError:
        release = marker == "release"
        if not release:
            raise ValueError("Unknown checkpoint marker.")
    assert iteration > 0 or release, "Invalid checkpoint metadata."
    return iteration, release
'''


def reader_from_source(source: str):
    """Execute only the reader, with distributed execution disabled."""
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "read_metadata"
    )
    namespace = {
        "open_file": open,
        "sys": sys,
        "print_rank_0": lambda *unused: None,
        "torch": SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "checkpoint_reader", "exec"),
        namespace,
    )
    return namespace["read_metadata"]


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_patch_changes_only_the_reader_assertion(self):
        result, changed = PATCHER.patched_source(FIXTURE)
        expected = FIXTURE.replace(
            'assert iteration > 0 or release, "Invalid checkpoint metadata."',
            'assert iteration >= 0 or release, "Invalid checkpoint metadata."',
        )
        self.assertTrue(changed)
        self.assertEqual(result, expected)
        self.assertIn("def unrelated(iteration, release):\n    assert iteration > 0", result)

    def test_patched_reader_accepts_zero_positive_and_release_but_rejects_negative(self):
        result, _ = PATCHER.patched_source(FIXTURE)
        reader = reader_from_source(result)
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory) / "latest_checkpointed_iteration.txt"
            for text, expected in (
                ("0", (0, False)),
                ("1", (1, False)),
                ("42", (42, False)),
                ("release", (0, True)),
            ):
                with self.subTest(text=text):
                    tracker.write_text(text)
                    self.assertEqual(reader(str(tracker)), expected)
            for text in ("-1", "-42"):
                with self.subTest(text=text):
                    tracker.write_text(text)
                    with self.assertRaises(AssertionError):
                        reader(str(tracker))
            tracker.write_text("unknown")
            with self.assertRaises(ValueError):
                reader(str(tracker))

    def test_already_patched_source_is_idempotent(self):
        result, _ = PATCHER.patched_source(FIXTURE)
        self.assertEqual(PATCHER.patched_source(result), (result, False))

    def test_unknown_function_signature_or_assertion_fails(self):
        cases = [
            FIXTURE.replace("def read_metadata(", "def another_reader("),
            FIXTURE.replace("def read_metadata(tracker_filename):", "def read_metadata(path):"),
            FIXTURE.replace(
                "def read_metadata(tracker_filename):", "async def read_metadata(tracker_filename):"
            ),
            FIXTURE.replace(
                "def read_metadata(tracker_filename):",
                "def read_metadata(tracker_filename, **kwargs):",
            ),
            FIXTURE.replace("iteration > 0 or release,", "iteration > 1 or release,"),
            FIXTURE.replace("iteration > 0 or release,", "iteration > False or release,"),
            FIXTURE.replace("iteration > 0 or release,", "iteration > 0 and release,"),
            FIXTURE.replace("iteration > 0 or release,", "iteration < 0 or release,"),
            FIXTURE.replace("iteration > 0 or release,", "release or iteration > 0,"),
            FIXTURE.replace("iteration > 0 or release,", "(iteration\n        > 0) or release,"),
            FIXTURE + "\ndef read_metadata(tracker_filename):\n    return 0\n",
            FIXTURE.replace(
                "    return iteration, release",
                "    assert iteration > 0 or release\n    return iteration, release",
            ),
            "def read_metadata(:\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaises(PATCHER.CompatibilityPatchError):
                    PATCHER.patched_source(source)

    def test_nested_assertion_does_not_match(self):
        source = "def read_metadata(tracker_filename):\n    def helper():\n        assert iteration > 0 or release\n"
        with self.assertRaises(PATCHER.CompatibilityPatchError):
            PATCHER.patched_source(source)

    def test_atomic_patch_preserves_bytes_and_permissions_and_does_not_repeat_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpointing.py"
            original = ("# Encoding example: café\n" + FIXTURE).replace("\n", "\r\n").encode()
            path.write_bytes(original)
            path.chmod(0o640)
            with patch.object(PATCHER.os, "replace", wraps=os.replace) as replace:
                self.assertTrue(PATCHER.patch_checkpoint_file(path))
                self.assertFalse(PATCHER.patch_checkpoint_file(path))
                self.assertEqual(replace.call_count, 1)
            expected = original.replace(
                b'assert iteration > 0 or release, "Invalid checkpoint metadata."',
                b'assert iteration >= 0 or release, "Invalid checkpoint metadata."',
            )
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)

    def test_unknown_source_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpointing.py"
            path.write_text("def another_function():\n    return 0\n")
            original = path.read_bytes()
            with self.assertRaises(PATCHER.CompatibilityPatchError):
                PATCHER.patch_checkpoint_file(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_file_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.py"
            original.write_text(FIXTURE)
            link = Path(directory) / "checkpointing.py"
            link.symlink_to(original)
            with self.assertRaises(PATCHER.CompatibilityPatchError):
                PATCHER.patch_checkpoint_file(link)
            self.assertEqual(original.read_text(), FIXTURE)

    def test_cli_uses_environment_root_and_rejects_unknown_source(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "megatron" / "training" / "checkpointing.py"
            target.parent.mkdir(parents=True)
            target.write_text(FIXTURE)
            env = dict(os.environ, MEGATRON_LM_PATH=directory)
            first = subprocess.run(
                [sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True
            )
            second = subprocess.run(
                [sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("Patched:", first.stdout)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("Already compatible:", second.stdout)
            target.write_text("def unknown():\n    pass\n")
            failed = subprocess.run(
                [sys.executable, str(SCRIPT), "--megatron-root", directory],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("Cannot patch Megatron checkpoint metadata", failed.stderr)
            self.assertEqual(target.read_text(), "def unknown():\n    pass\n")

    @unittest.skipUnless(
        os.environ.get("TRITON_RL_TEST_MEGATRON_CHECKPOINT_SOURCE"),
        "Set TRITON_RL_TEST_MEGATRON_CHECKPOINT_SOURCE to test an external Megatron source file.",
    )
    def test_external_megatron_reader(self):
        source = Path(os.environ["TRITON_RL_TEST_MEGATRON_CHECKPOINT_SOURCE"]).read_text()
        result, _ = PATCHER.patched_source(source)
        reader = reader_from_source(result)
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory) / "latest_checkpointed_iteration.txt"
            for text, expected in (("0", (0, False)), ("1", (1, False)), ("release", (0, True))):
                with self.subTest(text=text):
                    tracker.write_text(text)
                    self.assertEqual(reader(str(tracker)), expected)
            tracker.write_text("-1")
            with self.assertRaises(AssertionError):
                reader(str(tracker))


if __name__ == "__main__":
    unittest.main()
