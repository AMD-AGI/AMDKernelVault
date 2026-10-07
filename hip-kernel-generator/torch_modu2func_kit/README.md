# torch_modu2func_kit

The toolkit converts PyTorch modules into functional Python files through an LLM provider.
It accepts a generated file only after the checks below pass.
Successful files retain the original directory structure and file names.

## How It Works

For each `.py` sample under the input tree:

1. Read the original module file.
2. Build a prompt from the conversion rules, examples, and original source code.
3. Call the selected provider through the `models/` interface.
4. Import the original and generated modules.
5. Compare the original output with the generated output under the checks below.
6. Add the failed candidate and failure details to the next prompt if a check fails.
7. Stop after the first accepted candidate or after `--max-attempts` attempts.

## Verification Scope

Verification uses one original `get_init_inputs()` result and one original `get_inputs()` result per candidate.
Each model receives independent clones of those values.
Generated input functions must exist, but verification does not call them.

The verifier uses the configured seed before model construction and each forward call.
It sets each model to evaluation mode and disables gradient recording during forward calls.
The generated `Model.forward` must accept the optional keyword parameter `fn`.
The verifier compares the original output with both the default call and a tracked `fn=module_fn` injection.
The injected call uses a fresh model with the same initialization seed.
The verifier rejects a candidate that does not call the injected function.

Output containers and tensor shapes must match.
Tensor comparisons use `torch.allclose` with default `rtol=1e-4`, `atol=1e-4`, and `equal_nan=True`.
The CLI can change `rtol` and `atol`.

Clones retain each tensor's strides and storage offset.
Clones do not preserve shared storage between separate tensor entries.
Each clone copies the tensor's underlying storage, which can exceed its visible slice.

These checks cover one sampled case.
They do not prove equivalence for other shapes, dtypes, devices, random seeds, or code paths.
Tracking confirms that the sampled call invokes `fn`.
It does not prove that every output depends only on `fn`.

## Output Layout

With `--artifacts-dir .artifacts`, the pipeline writes:

- `kernelbench_torch_func/...`: accepted functional files
- `.artifacts/prompts/...`: the prompt for each attempt
- `.artifacts/candidates/...`: the generated Python candidate for each attempt
- `.artifacts/conversion_records.json`: all saved records
- `.artifacts/successful_conversions.json`: current successes only
- `.artifacts/failed_conversions.json`: current failures only

Each attempt record stores paths, status, hashes, verification settings, and any failure details.
Successful records also store source and output SHA256 hashes.

## Resume Behavior

A default rerun preserves a previous success and its attempts when its provenance matches.
The source path, output path, file hashes, seed, `rtol`, and `atol` must match.
A missing output triggers a new conversion.
New attempts use new artifact numbers and retain earlier attempt records.
`attempts_used` counts recorded attempts across runs.
Retry prompts include only attempts from the current run.

An existing output without matching provenance receives `status="skipped"` and a `skip_reason`.
This includes legacy outputs, changed files, and changed verification settings.
The pipeline excludes these records from the success file.
Use `--overwrite` to generate and verify these outputs again.
An invalidated record remains unverified if the original bytes or settings return.
Provenance covers file bytes and verification settings, but not the verifier version or external dependencies.

The pipeline retains saved records during partial reruns.
Each JSON file uses an atomic replacement, but the three files do not form one atomic transaction.
The returned summary covers files discovered during the current run.
Retained successes count as `success` in that summary.

## Quick Start

Install the package from this directory:

```bash
pip install -e '.[dev]'
```

Run the pipeline with a public provider:

```bash
torch-modu2func \
  --input-dir ../kernelbench_torch_modu \
  --output-dir ../kernelbench_torch_func \
  --artifacts-dir .artifacts \
  --provider openai \
  --model-id gpt-5 \
  --api-key YOUR_API_KEY
```

The existing key variable also works:

```bash
export TORCH_MODU2FUNC_API_KEY=YOUR_API_KEY
torch-modu2func --input-dir ../kernelbench_torch_modu --output-dir ../kernelbench_torch_func
```

## Providers

Provider names retain their existing spelling.
The `openai` and `claude` names select the same public adapters as their `standard-` aliases.
The toolkit does not supply private gateways or deployment names.
`--api-key` takes priority over `TORCH_MODU2FUNC_API_KEY`.
The package key takes priority over the provider's standard key.
When `--model-id` is absent, the provider selects the default model below.

| Provider | Public default model | Standard key variable |
| --- | --- | --- |
| `openai`, `standard-openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `claude`, `standard-claude` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.5-pro` | `GEMINI_API_KEY` |

Gemini uses Google's public OpenAI-compatible endpoint at `https://generativelanguage.googleapis.com/v1beta/openai/`.
See the [Google endpoint documentation](https://ai.google.dev/gemini-api/docs/openai).
Choose a model available through your provider account with `--model-id`.

For original `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, and their 2025 date snapshots, the adapter sends `max_completion_tokens`.
It omits `temperature`, so these models use the provider's default sampling.
For these models, `--max-tokens` includes visible output tokens and reasoning tokens.
The `--temperature` option applies only to models that support it.

## Important Arguments

- `--max-attempts`: maximum generation attempts per sample in the current run, default `5`
- `--rtol`: relative tolerance, default `1e-4`
- `--atol`: absolute tolerance, default `1e-4`
- `--seed`: verification seed, default `1234`
- `--temperature`: model sampling temperature
- `--max-tokens`: maximum tokens per generation call
- `--overwrite`: generate and verify a replacement for an existing output

`PipelineConfig.history_code_char_limit` and `PipelineConfig.history_feedback_char_limit` limit each history block in retry prompts.

## Development

Run the unit tests:

```bash
python -m pytest
```
