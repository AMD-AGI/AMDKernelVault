# py_hip_kernel2kernel_kit

`py_hip_kernel2kernel_kit` is a sibling of `torch2hip_kit`.
It optimizes existing baseline HIP files instead of generating a new HIP file
from scratch.

The pipeline is designed for datasets such as:

- PyTorch module tree: `kernelbench_torch_modu_l1`
- PyTorch functional tree: `kernelbench_torch_func_l1`
- Baseline HIP tree: `output_l1_hip`

For each baseline `.hip` file, the kit:

1. Finds the paired PyTorch module and functional files by relative path.
2. Extracts all `__global__` and `__device__` functions from the HIP source.
3. Selects one optimization target per file, preferring the longest
   `__global__` function and otherwise falling back to the longest `__device__`
   function. You can also force device-only extraction with
   `--target-function-mode device`.
4. Validates the baseline HIP file against the paired PyTorch module.
5. Prompts an LLM to optimize only the selected GPU function.
6. Replaces the original function body with the optimized body and recompiles.
7. Validates correctness against the original PyTorch module with the same
   `get_init_inputs()` and `get_inputs()` contract used by `torch2hip_kit`.
8. Benchmarks the optimized candidate against both the baseline HIP file and the
   original PyTorch module.
9. Repeats for up to `5` attempts, feeding the previous candidate code,
   compiler/runtime errors, correctness mismatches, and performance numbers into
   the next prompt.
10. Saves the best validated optimized HIP file together with JSON records and
    per-attempt artifacts.

## Highlights

- A CLI, configuration, JSON records, and saved artifacts.
- Default single-thread execution with optional multi-threaded production via
  worker threads.
- Configurable target selection: `auto`, `global`, or device-only.
- Self-contained prompt assets with optional override hooks.
- In-process HIP compilation through `torch.utils.cpp_extension.load`.
- Timeout controls for Python loading, HIP compilation,
  correctness execution, and benchmarking.
- Records for Python exceptions in each sample.
- Verification that separates:
  - baseline HIP validation
  - candidate compilation
  - correctness comparison
  - speedup vs baseline HIP
  - speedup vs original PyTorch module
- Function-level optimization rather than whole-file rewriting.
- Test coverage for parser, prompting, and pipeline record selection.

## Output Layout

Assuming `--artifacts-dir .artifacts`, the pipeline writes:

- `<output-dir>/.../*.hip`: selected optimized HIP files
- `.artifacts/prompts/...`: the exact prompt used for each attempt
- `.artifacts/function_candidates/...`: extracted model code for the target GPU
  function
- `.artifacts/candidates/...`: patched full HIP files used for compilation and
  validation
- `.artifacts/build/...`: optional build directories when
  `--keep-build-dirs` is enabled
- `.artifacts/optimization_records.json`: all records
- `.artifacts/successful_optimizations.json`: successful samples only
- `.artifacts/successful_optimizations/.../*.json`: one JSON per successful HIP
  sample, mirroring the HIP relative path with a `.json` suffix
- `.artifacts/failed_optimizations.json`: failed samples only

Each attempt record stores:

- prompt path
- function candidate path
- patched HIP path
- extracted optimized GPU function
- compile result
- correctness result
- speedup vs baseline HIP
- speedup vs PyTorch
- module, baseline, and candidate latency
- captured feedback or traceback

Each conversion record also stores:

- baseline HIP path
- PyTorch module path
- PyTorch functional path
- selected baseline GPU function
- selected optimized GPU function
- fingerprints for the sources, configuration, and selected output

## Quick Start

Install locally:

```bash
pip install -e .[dev]
```

Run the optimizer:

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
  --model-id gpt-4o \
  --api-key YOUR_API_KEY
```

Resume a previous run from the same artifact directory:

```bash
py-hip-kernel2kernel \
  --baseline-hip-dir ../output_l1_hip \
  --module-dir ../kernelbench_torch_modu_l1 \
  --functional-dir ../kernelbench_torch_func_l1 \
  --output-dir ./optimized_output_l1_hip \
  --artifacts-dir .artifacts \
  --resume
```

Environment variables are also supported:

```bash
export PY_HIP_KERNEL2KERNEL_API_KEY=YOUR_API_KEY
py-hip-kernel2kernel --baseline-hip-dir ../output_l1_hip --module-dir ../kernelbench_torch_modu_l1 --functional-dir ../kernelbench_torch_func_l1 --output-dir ./optimized_output_l1_hip
```

Compatibility fallbacks are accepted too:

- `HIP2HIP_API_KEY`
- `TORCH2HIP_API_KEY`
- `TORCH_MODU2FUNC_API_KEY`

The `openai` and `standard-openai` names use the public OpenAI API.
The `claude` and `standard-claude` names use the public Anthropic API.
The `gemini` provider uses Google's public OpenAI-compatible endpoint.
Select a model that the chosen provider supports.
Standard provider environment variables also supply credentials: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`.

Original GPT-5 models use `max_completion_tokens` and sampling defaults from the provider.
This rule covers `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, and their 2025 date snapshots.
The CLI temperature setting applies to models that support it.

## Important Arguments

- `--baseline-hip-dir`: root directory of baseline HIP samples
- `--module-dir`: root directory of paired PyTorch module samples
- `--functional-dir`: root directory of paired PyTorch functional samples
- `--num-workers`: number of worker threads for production; default `1`. Each
  worker processes one baseline HIP optimization task at a time
- `--target-function-mode`: `auto`, `global`, or `device`
- `--output-dir`: root directory for selected optimized HIP outputs
- `--artifacts-dir`: prompt, candidate, build, and record output directory
- `--resume`: retained for existing commands. Every run reuses matching verified outputs and retries missing outputs or failures.
- `--provider`: one of `openai`, `standard-openai`, `claude`,
  `standard-claude`, `gemini`
- `--model-id`: provider-specific model identifier
- `--api-key`: explicit API key override
- `--max-attempts`: maximum optimized attempts per sample, default `5`
- `--rtol`: relative tolerance for output comparison, default `1e-4`
- `--atol`: absolute tolerance for output comparison, default `1e-4`
- `--seed`: deterministic verification seed, default `1234`
- `--temperature`: model sampling temperature
- `--max-tokens`: maximum tokens per generation call
- `--python-load-timeout-seconds`: timeout for importing Python samples and
  constructing callable objects; `0` disables it
- `--hip-compile-timeout-seconds`: timeout for HIP extension compilation; `0`
  disables it
- `--execution-timeout-seconds`: timeout for one correctness execution of a
  PyTorch or HIP path; `0` disables it
- `--benchmark-timeout-seconds`: timeout for one benchmark phase; `0` disables
  it
- `--perf-warmup`: warmup iterations before timing, default `25`
- `--perf-iterations`: measured timing iterations, default `200`
- `--offload-arch`: optional ROCm arch such as `gfx90a`
- `--keep-build-dirs`: preserve build directories for debugging
- `--overwrite`: rewrite existing optimized HIP outputs; when combined with
  `--resume`, this also forces successful samples to run again and refresh
  their records

Normal reruns preserve a success record only when its sources, configuration, and selected output match the saved fingerprints.
Missing outputs run again.
Stale outputs and outputs without fingerprints require `--overwrite` before the pipeline replaces them.
This rule also applies to records from older releases.
The pipeline writes each JSON file atomically.
It preserves unchanged success shards.

## Supported GPU Signatures

The parser and extractor are designed to cover at least these GPU-function
signature shapes when they are defined with bodies:

```cpp
__global__ void k();
__global__ __launch_bounds__(256) void k();
__global__ __launch_bounds__(256,4) void k();

__device__ __forceinline__ int f();
__device__ __noinline__ int f();

__host__ __device__ constexpr int f();

template<typename T>
__global__ void k(T*);

template<int N>
__global__ __launch_bounds__(N) void k();

__global__
void k(float* __restrict__);
```

Function declarations without bodies are ignored during optimization, but the
same signature forms are supported for real function definitions.

## Runtime Requirements

To run real end-to-end optimization, you need:

- a PyTorch build with HIP/ROCm support
- a working ROCm/HIP toolchain available to `torch.utils.cpp_extension`
- the required Python dependencies from `pyproject.toml`
- access to at least one configured LLM provider

The included unit tests use CPU fixtures and mocked HIP compilation.
They do not establish GPU correctness or performance.

## Verification and timing limits

The pipeline calls the original `get_inputs()` once per source file with one fixed seed.
Every candidate uses that same argument set.
The pipeline does not generate additional shape, stride, dtype, or boundary cases.
It requires the functional wrapper to call the injected HIP function during correctness checks.
Tensor outputs must have exactly matching shapes.
Tensor comparisons use `rtol=1e-4`, `atol=1e-4`, and `equal_nan=True` by default.
Independent input copies preserve tensor strides and storage offsets.

The pipeline feeds compiler, correctness, and timing feedback directly into the next generation prompt.
It does not call a separate reflector model.
The pipeline saves the fastest correct candidate among its attempts, even when every candidate is slower than the baseline.
CUDA events measure repeated callable execution, including input copies and the functional wrapper.
These measurements are not isolated kernel timings.
Concurrent workers can affect measurements through shared GPU use and global random seeds.

## Failure Handling

The pipeline handles Python failures as follows:

- parser, compile, execution, and benchmark failures are captured into the
  per-sample JSON record
- timeout failures are reported with the exact phase that timed out
- unexpected exceptions at the file level are converted into failed records so
  the remaining dataset can continue to run
- multi-threaded production is supported with one baseline HIP sample per
  worker thread; each worker lazily creates and reuses its own model client

Compilation and execution run in the current process.
A timeout reports a failure but does not guarantee that native work stops.
A worker thread can continue after its timeout.
Fatal native errors can terminate the process.
