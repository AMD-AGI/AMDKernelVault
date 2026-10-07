# torch2hip_kit

`torch2hip_kit` is a production-oriented pipeline for converting PyTorch module
implementations into HIP kernels with Python bindings, validating correctness
against paired functional PyTorch code, and selecting the fastest validated
kernel across multiple LLM attempts.

It is designed as the HIP counterpart to `torch_modu2func_kit`:

- It walks a module input tree such as `kernelbench_torch_modu`.
- It pairs each module sample with the matching functional sample under
  `kernelbench_torch_func`.
- It asks an LLM to generate a HIP source file with a Python-callable
  `forward` binding.
- It compiles the HIP code through `torch.utils.cpp_extension.load`.
- It checks correctness by calling `get_init_inputs()`, `get_inputs()`, and
  `Model.forward()` on the original PyTorch module and the functional wrapper
  backed by the generated HIP kernel.
- It benchmarks validated candidates and keeps the best-performing correct HIP
  file.

## Design Requirements Summary

The intended design goals for this project are:

1. Build a production-oriented `pytorch-module-to-hip` pipeline in
   `torch2hip_kit`.
2. Recursively traverse an input PyTorch module tree such as
   `kernelbench_torch_modu`.
3. Use an LLM plus prompt assets and few-shot examples to generate HIP kernels
   with a Python-callable binding.
4. Compile or wrap the HIP implementation into a callable Python interface.
5. Use the paired PyTorch functional tree such as `kernelbench_torch_func` to
   invoke the HIP-backed path.
6. Validate correctness by calling the original module's `get_init_inputs()`,
   `get_inputs()`, and `Model.forward()` with the same inputs on both paths.
7. Benchmark correctness-passing HIP kernels against the original PyTorch
   module.
8. Allow at most `5` attempts per sample and keep the best-performing correct
   HIP result.
9. Persist prompt artifacts, candidate artifacts, and production-style JSON
   records.
10. Keep the LLM calling interface aligned with `torch_modu2func_kit`.
11. Keep prompt assets self-contained inside `torch2hip_kit` by default.

## Current Status Against Requirements

After re-checking the implementation, the project currently stands at:

- Fully implemented:
  recursive discovery, module-to-functional pairing, LLM-driven HIP generation,
  correctness verification from the module-side input contract, performance
  benchmarking, best-of-attempt selection, JSON/artifact persistence, and
  self-contained prompt assets.
- Public-interface compatible with `torch_modu2func_kit`:
  `create_model_client(provider, model_id, api_key)`, provider names, and
  `client.generate(messages, **kwargs)` are aligned; legacy
  `TORCH_MODU2FUNC_API_KEY` is also accepted.
- Partially implemented at the wording level:
  the current build path uses `torch.utils.cpp_extension.load` to JIT-compile
  the HIP extension and expose `forward`; it does not use the literal
  `torch.compile(...)` API.
- Partially implemented at the "production-level" operations level:
  the codebase has CLI/config/tests/artifacts and retry logic, but it does not
  yet include stronger operational features such as resumable scheduling,
  distributed execution control, or richer logging/metrics infrastructure.
- Important contract assumption:
  the paired functional file must expose `module_fn`, `Model`, `get_inputs`,
  and `get_init_inputs`; correctness inputs are always sourced from the
  original module file.
  `Model.forward` must accept an explicit `fn` keyword parameter.
  The wrapper must call the supplied function during the correctness check.

## Known Gaps And Boundaries

- HIP compilation depends on a working PyTorch + ROCm/HIP environment.
- Verification currently fails fast when no CUDA/HIP-capable device is visible.
- Best-candidate selection is based on the highest recorded speedup among
  correctness-passing attempts; equal-speed ties currently keep the earlier
  success.
- The selected candidate can be slower than PyTorch. The pipeline applies no
  minimum speedup threshold.
- The package is production-oriented in structure, but still a kit rather than
  a fully operationalized service.

## Validation Scope

The verifier calls the original `get_inputs()` once with the configured seed.
It uses that argument set for correctness and timing.
The kit does not create additional shape, stride, dtype, or boundary cases.
Use separate reference tasks when you need those checks.

Tensor comparison requires equal output shapes.
It uses `torch.allclose` with default `rtol=1e-4`, `atol=1e-4`, and `equal_nan=True`.
The checker applies no fixed tolerance rule for a dtype or operator.
Input copies preserve tensor strides and storage offsets.
Device transfers preserve these layouts for ordinary strided tensors.
Sparse and quantized tensors use PyTorch's normal device transfer.

The verifier rejects a wrapper that ignores the supplied HIP function.
This check proves invocation on the sampled input.
It does not prove equivalence for all inputs or establish complete dependence on the function's result.
Runtime errors preserve the successful compile flag.
Timing errors preserve successful compile and correctness flags.
The pipeline selects candidates only when correctness and timing complete successfully.

The timing loop measures repeated calls with CUDA/HIP events.
It includes the functional wrapper's GPU operations.
The loop reuses its inputs and models, so timing assumes stable repeated calls.
The kit does not isolate each candidate in a separate process or impose a runtime timeout.

Each retry receives prior code and verifier feedback.
The kit uses one generation call per attempt and does not call a separate reflector model.
The released code therefore demonstrates a feedback loop, not the full historical corpus protocol.

## Highlights

- Mirrors the production-style pipeline structure of `torch_modu2func_kit`.
- Preserves the input directory structure when writing output HIP files.
- Retries each sample up to a configurable attempt limit, default `5`.
- Injects prior attempt history into the next prompt, including candidate code,
  failure feedback, and best known speedup.
- Validates correctness against the original module using the same sample's
  `get_init_inputs()` and `get_inputs()` contract.
- Benchmarks every correct candidate and keeps the fastest validated result.
- Writes structured JSON records for all samples and persists per-attempt
  prompts and candidate HIP files for debugging.
- Keeps the LLM factory interface compatible with `torch_modu2func_kit`.
- Ships with self-contained default prompt assets, including embedded few-shot
  HIP examples, so the package can run without depending on sibling prompt files.

## LLM Interface Compatibility

The external LLM interface is intentionally kept aligned with
`torch_modu2func_kit`.

You can still use:

```python
from torch2hip_kit.model_factory import create_model_client

client = create_model_client(provider, model_id, api_key)
response = client.generate(messages, temperature=0.0, max_tokens=12000)
```

Supported provider names are the same:

- `openai`
- `standard-openai`
- `claude`
- `standard-claude`
- `gemini`

`openai` and `standard-openai` use the public OpenAI API.
`claude` and `standard-claude` use the public Anthropic API.
`gemini` uses Google's public OpenAI-compatible endpoint.
The legacy provider names remain available as aliases.

Original GPT-5 models use `max_completion_tokens` and the provider's default sampling.
This includes `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, and their dated 2025 snapshots.
The completion budget includes reasoning tokens, as the [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) specifies.
The adapter preserves the caller's exact model identifier.
CLI `--temperature` applies only to models that support it.
Other model families and Gemini retain their existing request parameters.

The main compatibility note is API key resolution:

- `TORCH2HIP_API_KEY` is supported.
- `TORCH_MODU2FUNC_API_KEY` is also accepted for compatibility with the
  existing `torch_modu2func_kit` workflow.
- The provider's standard environment variable is also accepted.
  These variables are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

So compared with the earlier draft, the LLM call shape is now consistent:

- same `create_model_client(provider, model_id, api_key)` signature
- same provider names
- same `client.generate(messages, **kwargs)` calling pattern
- compatible environment-variable fallback

## How It Works

For each `.py` sample under `--module-dir`:

1. Read the original module file.
2. Find the paired functional file at the same relative path under
   `--functional-dir`.
3. Build a prompt from:
   - the system instruction
   - HIP few-shot examples
   - the original module source
   - the paired functional source
   - prior attempt history, if any, including previous HIP candidates, verifier
     feedback, traceback text, and any available speedup or latency data
4. Call the configured LLM provider.
5. Extract a HIP candidate from the model response and save it under
   `.artifacts/candidates/...`.
6. Compile the candidate as a PyTorch extension.
7. Instantiate the original module `Model` and the paired functional `Model`,
   feed both with inputs from the original module file, and compare outputs with
   `torch.allclose`.
8. If correctness passes, benchmark both paths and record the resulting speedup.
9. After all attempts are exhausted, keep the fastest correct HIP candidate and
   write it to `--output-dir`.

This differs from `torch_modu2func_kit` in one intentional way: the HIP
pipeline does not stop on the first correct result. It keeps searching across
attempts and selects the best validated performer.

## Directory Assumptions

The default workflow assumes two aligned trees:

- `kernelbench_torch_modu/...`: original PyTorch module implementations
- `kernelbench_torch_func/...`: paired functional implementations that can call
  a supplied `fn`

For a sample such as:

- `kernelbench_torch_modu/level_1/1_Square_matrix_multiplication_.py`

the pipeline expects the paired file at:

- `kernelbench_torch_func/level_1/1_Square_matrix_multiplication_.py`

and writes the selected HIP file to:

- `<output-dir>/level_1/1_Square_matrix_multiplication_.hip`

## Output Layout

Assuming `--artifacts-dir .artifacts`, the pipeline writes:

- `<output-dir>/.../*.hip`: selected best HIP files
- `.artifacts/prompts/...`: the exact prompt used for each attempt
- `.artifacts/candidates/...`: raw generated HIP candidates for each attempt
- `.artifacts/build/...`: optional per-attempt build directories when
  `--keep-build-dirs` is enabled
- `.artifacts/conversion_records.json`: all records
- `.artifacts/successful_conversions.json`: successful samples only
- `.artifacts/failed_conversions.json`: failed samples only

Each attempt record stores:

- prompt path
- candidate path
- status
- compile result
- correctness result
- speedup, if available
- module latency and HIP latency, if available
- captured feedback or traceback

Normal reruns retain matching success records and their selected outputs.
Reuse checks source contents, output contents, generation settings, validation settings, client class, and model identifier.
The records store fingerprints for these checks.
Missing outputs trigger generation again.
Existing stale or untracked outputs require `--overwrite` before replacement.
The pipeline preserves pending records if a rerun stops before those samples finish.

These checks do not record the complete GPU and compiler environment.
Keep that environment separately when you need to reproduce a measured latency.

## Prompt Assets

By default, `torch2hip_kit` is self-contained:

- the default system instruction lives in `torch2hip_kit/prompt_assets.py`
- the default few-shot prompt also lives in `torch2hip_kit/prompt_assets.py`
- the embedded few-shot section includes the vendored HIP reference cases used
  to guide interface shape, launcher structure, and `PYBIND11_MODULE` exposure

You can override them with:

- `--instruction-file`
- `--few-shot-file`

The files are expected to define:

- `hip_generation_req`
- `few_shot_code_instructions`

This means the default installation no longer requires runtime reads from the
external `torch2hip` directory just to assemble the prompt.

## Quick Start

Install locally:

```bash
pip install -e .[dev]
```

Run the pipeline:

```bash
torch2hip \
  --module-dir ../kernelbench_torch_modu \
  --functional-dir ../kernelbench_torch_func \
  --output-dir ./output_hip \
  --artifacts-dir .artifacts \
  --provider openai \
  --model-id gpt-4o \
  --api-key YOUR_API_KEY
```

Or use environment variables:

```bash
export TORCH2HIP_API_KEY=YOUR_API_KEY
torch2hip --module-dir ../kernelbench_torch_modu --functional-dir ../kernelbench_torch_func --output-dir ./output_hip
```

Compatibility with the existing module-to-function workflow is also supported:

```bash
export TORCH_MODU2FUNC_API_KEY=YOUR_API_KEY
torch2hip --module-dir ../kernelbench_torch_modu --functional-dir ../kernelbench_torch_func --output-dir ./output_hip
```

## Important Arguments

- `--module-dir`: root directory of PyTorch module samples
- `--functional-dir`: root directory of paired functional samples
- `--output-dir`: root directory for selected HIP outputs
- `--artifacts-dir`: prompt, candidate, build, and record output directory
- `--provider`: one of `openai`, `standard-openai`, `claude`,
  `standard-claude`, `gemini`
- `--model-id`: provider-specific model identifier
- `--api-key`: explicit API key override
- `--max-attempts`: maximum HIP attempts per sample, default `5`
- `--rtol`: relative tolerance for output comparison, default `1e-4`
- `--atol`: absolute tolerance for output comparison, default `1e-4`
- `--seed`: deterministic verification seed, default `1234`
- `--temperature`: model sampling temperature
- `--max-tokens`: maximum tokens per generation call
- `--perf-warmup`: warmup iterations before timing, default `25`
- `--perf-iterations`: measured timing iterations, default `200`
- `--keep-build-dirs`: preserve build directories for debugging
- `--overwrite`: rewrite existing selected HIP outputs

The pipeline also supports prompt history truncation through:

- `PipelineConfig.history_code_char_limit`
- `PipelineConfig.history_feedback_char_limit`

## Runtime Requirements

To run real end-to-end HIP generation, you need:

- a PyTorch build with HIP/ROCm support
- a working ROCm/HIP toolchain available to `torch.utils.cpp_extension`
- the required Python dependencies from `pyproject.toml`
- access to at least one configured LLM provider

The included unit tests do not require a working HIP compiler because they mock
the verification stage.

## Development

Run unit tests:

```bash
python -m pytest
```
