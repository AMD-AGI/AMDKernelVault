# HIPKernelGen correspondence with AMDKernelVault

The three HIP kits implement the main workflow in
[AMDKernelVault Section 3.1](https://arxiv.org/html/2609.12471#S3.SS1).
They do not reconstruct the complete historical validation of the released corpus.

| Paper component | Current implementation |
| --- | --- |
| PyTorch reference standardization | `torch_modu2func_kit` constructs a functional reference and checks its outputs and injected-function path. |
| PyTorch-to-HIP synthesis | `torch2hip_kit` supplies the module and paired reference to the generator. |
| AMD compilation | HIP functions use `torch.utils.cpp_extension` with a ROCm PyTorch build and HIP toolchain. |
| Numerical verification | Tensor checks use `rtol=1e-4`, `atol=1e-4`, and `equal_nan=True`. Output shapes must match exactly. |
| Failure feedback | Later generation prompts include previous candidates and verifier feedback. |
| Performance selection | The kits measure execution and retain the best correct candidate within the attempt budget. |
| HIP-to-HIP optimization | `py_hip_kernel2kernel_kit` validates a baseline, modifies a selected function, and compares candidates. |
| Retention | A correct candidate does not need speedup greater than one. |

The checks require an actual call through the injected function.
Input copies preserve ordinary tensor layouts, strides, and storage offsets.
Compilation, runtime, correctness, and timing outcomes retain their completed phase information.
Artifact reuse requires matching source and output fingerprints.
Missing outputs trigger work again, and repeated runs retain matching success records.

## Comparison limits

Each current HIP verifier uses one argument set from the original `get_inputs()`.
The paper describes broader source-derived and augmented test coverage.
These entry points therefore do not prove that expanded coverage for each generated kernel.
The separate Triton generator accepts an explicit fixed-case provider and records its actual configurations.

The HIP kits use direct retry feedback rather than a separately named reflector request.
This supplies the feedback loop, but it does not establish the paper's exact historical prompts or agent implementation.
The paper does not prescribe the kits' exact attempt counts, seeds, or HIP timing repetitions.

The HIP timing paths measure their Python wrapper boundaries.
The refinement path includes input copies in that boundary.
Those measurements do not establish isolated kernel-only latency for arbitrary wrappers.
The configurable defaults remain implementation settings.
Do not interpret the Triton paper's 25/100 ms timing budgets as these HIP repetition counts.

Native HIP compilation and GPU execution require validation on the target machine.
The code checks do not prove historical model use, corpus counts, or reproduced benchmark results.
The kit READMEs describe additional runtime and timeout limits.

## Public provider interface

The examples preserve the paper's GPT-5 model name.
The public OpenAI adapter uses `max_completion_tokens` and model-default sampling for original GPT-5 identifiers.
The [official Chat API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
defines this completion budget as including visible and reasoning tokens.
Provider availability and deployed model identity remain separate from the historical collection claim.
