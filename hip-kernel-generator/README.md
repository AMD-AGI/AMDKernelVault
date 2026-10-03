[![License](https://img.shields.io/github/license/AMD-AGI/hip-kernel-generator.svg?style=flat)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/AMD-AGI/hip-kernel-generator.svg?style=flat)](https://github.com/AMD-AGI/hip-kernel-generator/graphs/contributors)

# hip-kernel-generator

> LLM-assisted pipelines for converting PyTorch modules to functions or HIP and optimizing HIP kernels.

`hip-kernel-generator` is a monorepo of three production-oriented, LLM-assisted
pipelines for generating and optimizing GPU-facing implementations across
PyTorch and HIP.

The repository currently covers three workflows:

1. PyTorch module -> PyTorch functional implementation
2. PyTorch module -> HIP kernel implementation
3. HIP kernel -> optimized HIP kernel

Each workflow is packaged as its own installable tool with a CLI, prompt
assets, retry logic, artifact persistence, JSON record generation, and unit
tests.

## Repository Layout

```text
hip-kernel-generator/
├── torch_modu2func_kit/         # module -> functional
├── torch2hip_kit/               # module -> HIP
└── py_hip_kernel2kernel_kit/    # HIP -> optimized HIP
```

Detailed package docs:

- [`torch_modu2func_kit/README.md`](torch_modu2func_kit/README.md)
- [`torch2hip_kit/README.md`](torch2hip_kit/README.md)
- [`py_hip_kernel2kernel_kit/README.md`](py_hip_kernel2kernel_kit/README.md)

## What Each Kit Does

### `torch_modu2func_kit`

Converts a tree of PyTorch module files into verified functional equivalents.

- Input: PyTorch module tree such as `kernelbench_torch_modu`
- Output: functional tree such as `kernelbench_torch_func`
- Verification: imports both versions and compares outputs with
  `get_init_inputs()`, `get_inputs()`, and `Model.forward()`
- CLI: `torch-modu2func`

See [`torch_modu2func_kit/README.md`](torch_modu2func_kit/README.md).

### `torch2hip_kit`

Generates HIP kernels from PyTorch module files and validates them against a
paired functional PyTorch tree.

- Input: module tree plus paired functional tree
- Output: selected HIP files
- Verification: compiles HIP, calls the paired functional wrapper with the HIP
  `forward`, checks correctness, and benchmarks latency
- CLI: `torch2hip`

See [`torch2hip_kit/README.md`](torch2hip_kit/README.md).

### `py_hip_kernel2kernel_kit`

Optimizes an existing baseline HIP tree by rewriting one selected GPU function
per file and validating the result against paired PyTorch references.

- Input: baseline HIP tree plus paired module and functional trees
- Output: selected optimized HIP files
- Verification: validates baseline HIP first, then benchmarks optimized
  candidates against both baseline HIP and original PyTorch
- CLI: `py-hip-kernel2kernel`

See [`py_hip_kernel2kernel_kit/README.md`](py_hip_kernel2kernel_kit/README.md).

## Recommended Workflow

If you are starting from PyTorch module implementations, the intended flow is:

1. Use `torch-modu2func` to build a verified functional reference tree.
2. Use `torch2hip` to generate initial HIP implementations.
3. Use `py-hip-kernel2kernel` to optimize those HIP implementations further.

This gives you a clean functional reference for correctness checking before you
move into HIP generation and kernel-level optimization.

## Shared Design Traits

Across the three kits, the repository follows the same general pattern:

- recursive input discovery that preserves relative paths
- attempt-based generation with bounded retries
- prompt history injection from prior failed attempts
- persisted prompt and candidate artifacts under `.artifacts`
- structured JSON output for all, successful, and failed samples
- self-contained prompt assets or override hooks
- focused unit tests around prompting, pipeline orchestration, and verification

## Requirements

General requirements:

- Python 3.10+
- PyTorch
- one configured LLM provider

HIP-related workflows additionally require:

- a PyTorch build with ROCm/HIP support
- a working ROCm/HIP toolchain available to
  `torch.utils.cpp_extension`
- at least one visible CUDA/HIP-capable device for end-to-end verification

## Installation

There is no single root package for the whole repository. Install the specific
kit you want to run:

```bash
pip install -e ./torch_modu2func_kit[dev]
pip install -e ./torch2hip_kit[dev]
pip install -e ./py_hip_kernel2kernel_kit[dev]
```

You can install only the subpackage you need, or all three if you use the full
pipeline.

## Quick Start

Set `OPENAI_API_KEY` in the environment before using the OpenAI examples.


### 1. PyTorch module -> functional

```bash
torch-modu2func \
  --input-dir ../kernelbench_torch_modu \
  --output-dir ../kernelbench_torch_func \
  --artifacts-dir .artifacts \
  --provider openai \
  --model-id gpt-5
```

### 2. PyTorch module -> HIP

```bash
torch2hip \
  --module-dir ../kernelbench_torch_modu \
  --functional-dir ../kernelbench_torch_func \
  --output-dir ./output_hip \
  --artifacts-dir .artifacts \
  --provider openai \
  --model-id gpt-5
```

### 3. HIP -> optimized HIP

```bash
py-hip-kernel2kernel \
  --baseline-hip-dir ../output_l1_hip \
  --module-dir ../kernelbench_torch_modu_l1 \
  --functional-dir ../kernelbench_torch_func_l1 \
  --num-workers 1 \
  --target-function-mode auto \
  --output-dir ./optimized_output_l1_hip \
  --artifacts-dir .artifacts \
  --provider openai \
  --model-id gpt-5
```

For full argument descriptions, environment-variable fallbacks, and operational
details, see the README inside each subpackage.

## Outputs and Artifacts

All three pipelines preserve the relative layout of the dataset they process.

Typical outputs include:

- generated or optimized source files under the configured output directory
- `.artifacts/prompts/...` for exact prompts sent to the model
- `.artifacts/candidates/...` or
  `.artifacts/function_candidates/...` for per-attempt generations
- `.artifacts/build/...` for optional per-attempt build directories
- JSON summary records for all, successful, and failed samples

This makes it easier to inspect failures, compare retries, and debug prompt or
verification issues.

## Testing

Each subpackage contains its own test suite. Run tests from the package
directory:

```bash
cd torch_modu2func_kit && python -m pytest
cd torch2hip_kit && python -m pytest
cd py_hip_kernel2kernel_kit && python -m pytest
```

## Path Conventions

Repository documentation and serialized artifact paths use POSIX-style `/`
separators for consistency across platforms. Runtime filesystem operations still
use the native path handling provided by Python's `pathlib`.

## Contributing [Required for public repos]

We welcome contributions. Until a dedicated `CONTRIBUTING.md` is added at the
repository root, start with the README in the target subpackage, follow the
existing test patterns, and submit focused pull requests with clear validation.

For bugs and feature requests, open a [GitHub Issue](https://github.com/AMD-AGI/hip-kernel-generator/issues).

## Security [Required for public repos]

To report a security vulnerability, **do not open a public GitHub issue**.
Until a dedicated `SECURITY.md` is added at the repository root, please contact
the repository owners listed in [`.github/CODEOWNERS`](.github/CODEOWNERS)
directly.

## Contact

For questions, issues, or contributions, please reach out to the maintainers:

- `@AMD-AGI/AI-Algorithm` - repository owning team
- `@liuji` - repository maintainer

See [`.github/CODEOWNERS`](.github/CODEOWNERS) for the ownership list.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## AMDKernelVault correspondence

See [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for the implemented HIP workflow and its validation limits.
The separate [Triton generator](../triton-kernel-generator/README.md) adds fixed-case Triton construction and AMD verification.
