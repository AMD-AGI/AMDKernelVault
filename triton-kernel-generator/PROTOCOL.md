# External task and verification protocol

## Task manifests

The CLI accepts one task or a manifest containing several tasks.
Manifest paths resolve relative to the manifest file.
Absolute paths are also accepted.

```json
{
  "schema_version": 1,
  "tasks": [
    {
      "task_id": "task_001",
      "module": "modules/task_001.py",
      "functional": "functional/task_001.py",
      "case_provider": "cases/task_001.py",
      "dependencies": ["helpers.py"],
      "provenance": {"source": "caller-supplied", "license": "record the applicable upstream terms"}
    }
  ]
}
```

Task IDs must be unique and start with a letter or digit.
They can also contain dots, underscores, and hyphens.
Each code path must identify an existing Python file.
The original module and functional reference must be separate files.

Optional `seed_kernel` supplies raw source context.
It can require porting and does not run during reference preparation.
Optional `baseline_kernel` supplies a baseline that must pass AMD validation and timing.
The generator compares candidate outputs against PyTorch even when it uses a Triton timing baseline.

Declare local helper files and package initializers in `dependencies`.
These files receive source snapshots and hash checks.
Regular-package relative imports require each enclosing `__init__.py` to appear in that list.
Declared sources execute from current bytes rather than timestamp-based bytecode caches.
Unlisted dependencies remain outside the declared-source provenance boundary.

## Cases

A case provider exports:

```python
def get_cases(task_id: str, module, *, kind: str):
    ...
```

`module` is the original, trusted PyTorch module.
`kind` is either `correctness` or `performance`.
Return or yield `triton_kernel_gen.Case` objects.
Both suites must be nonempty, and case IDs must be unique within each suite.

| Field | Meaning |
| --- | --- |
| `case_id` | Stable configuration identifier |
| `args`, `kwargs` | Arguments for `Model.forward` |
| `init_args`, `init_kwargs` | Arguments for constructing `Model` |
| `atol`, `rtol` | Optional task-specific tolerance overrides |
| `metadata` | Caller-supplied configuration description |

The default argument containers are tuples and dictionaries.
Use approximately ten correctness configurations and three performance configurations to follow the paper's stated coverage.
Select meaningful shapes, strides, dtypes, and boundary cases for the operator.
Different random values alone do not establish different shape or stride coverage.

`--seed` controls verification inputs and replay.
It does not set the inference service's sampling seed.
Use an external request-options file if the service supports a separate inference seed.

The verifier copies each yielded case before requesting the next one.
It preserves supported tensor strides, offsets, aliases, and repeated tensor references.
Supported values include plain tensors, exact `nn.Parameter` objects, ordinary scalars, and nested lists, tuples, or dictionaries.
Unsupported tensor layouts, subclasses, and named tensors fail explicitly.
The serialized tensor tree has a one-GiB size limit.

## Standardized reference interface

The original and functional files export `Model` classes.
The functional file also exports `module_fn`.
Its forward method selects the supplied `fn`, binds inputs and state, and returns that call directly.
Tensor computation belongs inside `module_fn`, including transformations of input tensors.

A conceptual forwarding signature is:

```python
def forward(self, x, fn=module_fn):
    return fn(x, self.weight, self.bias)
```

This interface is a release choice because the paper does not prescribe callable names.
The verifier rejects wrappers that bypass `fn`, transform its returned output, or move workload computation outside the function.
Arguments and model parameters, buffers, and ordinary attributes must remain unchanged during verification.

The trusted reference stage compares original, default functional, injected functional, and replayed function outputs.
It records native output-device information before transferring comparison data to the CPU.
Shape, dtype, device, structure, and value checks remain separate.
Tensor tolerances use the reference as expected and set `equal_nan=False`.

## Candidate execution

The candidate exports a callable `module_fn` with the required argument interface.
It is self-contained and uses Triton kernels for device computation.
Return one complete Python code fence or one complete `<answer>` region from the generation endpoint.
Thought text, ambiguous code blocks, and incomplete wrappers do not become executable candidates.

Candidate workers receive captured function arguments, not the reference module or expected outputs in their request.
The trusted controller loads expected results before launching a candidate worker.
Separate validation calls observe Triton compilation and kernel launches.
Those observations do not prove that every output causally depends on a particular observed launch.

Every configuration receives JIT warmup and an execution check.
The record separates compiler, runtime, numerical, timing, reference, and infrastructure failures.
An early runtime failure can leave later compilation configurations untested.
The verifier does not report those untested configurations as compiled.

Timeouts terminate the worker process group.
Worker environments omit model-service credentials and unrelated environment variables.
The source, case, expected-output, and candidate hashes remain checked around verification boundaries.
These controls provide fault containment and provenance checks.
They do not provide an operating-system sandbox for hostile Python code.

## Timing and retention

Correctness and performance use separate external configurations.
The default timer uses Triton 3.3.0 `do_bench` with `warmup=25`, `rep=100`, and `quantiles=[0.5]`.
These budgets use milliseconds.
The timer supplies GPU events, iteration estimation, cache flushing, and the median calculation.
No additional observation filtering occurs in this package.

Compilation, autotuning warmup, and input cloning precede timing.
The measured callback contains no validation launch probes.
It includes the ordinary `module_fn` call and any allocations inside that function.
Separate seeded checks validate outputs and input state after timing.

Each performance case records PyTorch latency, candidate latency, and the selected baseline.
If a validated Triton baseline is supplied, its measured latency becomes the baseline for speedup.
Per-case speedup is `baseline_ms / candidate_ms`.
The package does not apply a universal speedup threshold.

The default stops after the first correct, measured candidate.
Additional requested variants share the explicit construction budget.
Distinct source hashes identify retained variants.
The package does not claim that every retained variant improves performance.

## Run records

Final statuses are `success`, `partial_success`, `exhausted`, `reference_failed`, `provider_error`, `infrastructure_error`, or `input_modified`.
The record stores source snapshots, hashes, configuration, provenance, model responses, reasoning, prompts, reflection, and verification evidence.
It also records actual generation, reflection, verification, and retention counts.

Repeated runs create new record directories and reverify candidates.
They preserve earlier valid outputs and records.
Generated outputs and their metadata remain outside the source release.
