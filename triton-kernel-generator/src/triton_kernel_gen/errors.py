# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Errors at the boundary between trusted tasks and candidate execution."""


class ReferenceValidationError(ValueError):
    """The trusted task cannot establish a valid test boundary."""

    def __init__(self, message: str, *, failure_stage: str = "reference") -> None:
        super().__init__(message)
        self.failure_stage = failure_stage
