# Execution service protocol

The service accepts generated code and identifies an external reference file.
It runs the candidate in a subprocess inside a dedicated GPU container.
The training process never executes generated kernels.

## Requests and results

`POST /evaluate` accepts a JSON object with these fields:

| Field | Meaning |
| --- | --- |
| `protocol_version` | Integer `1` |
| `code` | One extracted Python/Triton module |
| `filename` | Relative path below the external reference directory |
| `atol` | Task-specific absolute tolerance |
| `rtol` | Task-specific relative tolerance |

`GET /health` reports service availability.
The client uses `TRITON_RL_SANDBOX_URLS` for its endpoint list.
Endpoints share capacity across clients in one rollout event loop.
Each worker handles one evaluation at a time on its assigned GPU.

Results use the shared `EvaluationResult` contract:

| Field | Meaning |
| --- | --- |
| `compiled` | The supplied tests report successful candidate invocation |
| `correct` | All required correctness checks pass |
| `speedup` | Dimensionless task speedup, when timing succeeds |
| `latency_ms` | Sum of matched candidate case latencies |
| `baseline_latency_ms` | Sum of matched reference case latencies |
| `feedback` | Compiler, runtime, correctness, or performance feedback |
| `error_type` | Candidate failure category, when applicable |
| `diagnostics` | Test counts, timing settings, case values, and operator ratios |

The compilation signal follows the external test harness's `_CALL_SUCCESS_` convention.
It does not independently prove that a candidate uses Triton or passes every possible input.
Use trusted tests that exercise the candidate's intended kernel.

Transport failures, invalid responses, and broken reference tests raise errors.
They do not become zero-reward training examples.
Candidate timing failures retain established compilation and correctness results.
They provide no performance bonus.

## External reference contract

The release includes no reference kernels or dataset preparation scripts.
Provide each reference as a Python file with one separator containing exactly 146 hash characters.
The prefix contains the reference implementation.
The suffix contains the trusted pytest tests for both implementations.

Correctness tests save an output dictionary at this path:

```python
__file__.replace(".", "_") + ".pt"
```

The dictionary must contain `_CALL_SUCCESS_` as a one-value PyTorch tensor.
It must also contain nonempty output evidence.
Supported outputs include tensors, scalar values, dictionaries, lists, and tuples.
The service checks reference success, pytest exit status, keys, shapes, and values.
Tensor comparison uses the reference as the expected value.
It retains the supplied tolerances and uses `equal_nan=False`.
Non-tensor outputs require exact type and value equality.

The absolute work directory must contain no periods because of the archived output filename convention.
The default `/tmp` work directory satisfies this condition.

The harness seeds Python, NumPy, and PyTorch before test definitions and before each test.
Reference and candidate runs use the same seed.
Tests that create inputs between candidate calls must use independent generators or supplied shared inputs.
A candidate can otherwise advance a shared random generator inside a test.

## Performance contract

The service invokes `test_performance`, then `test_save_performance_results`.
Tests write operator records under `PERF_OUTPUT_DIR`.
The timing helper supports these import paths:

```text
triton_rl.evaluation.timing
performance_utils_pytest
tb_eval.perf.ROCm.performance_utils_pytest
geak_eval.perf.ROCm.performance_utils_pytest
```

These aliases expose an independent compatibility implementation.
They do not install or redistribute either external evaluator package.

The default helper uses `triton.testing.do_bench` with a 25 ms warmup and a 100 ms measurement budget.
The requested percentiles are p50, p80, and p20, in that order.
External tests can supply different timing settings.
The service records the actual settings and requires reference/candidate agreement.

The archived `ms` field denotes the first requested percentile.
With default settings, it is the median latency.
The archived `min_ms` and `max_ms` names correspond to p80 and p20.
Explicit percentile fields preserve their actual meaning and unrounded values.

Matched records produce these ratios:

```text
operator_speedup = sum(reference_case_ms) / sum(candidate_case_ms)
task_speedup = arithmetic_mean(operator_speedup)
```

The calculation uses positive, finite per-case `ms` latency values.
The timing helper retains the archived four-decimal precision for that field.
The task average uses unrounded operator ratios.
The worker retains individual cases, explicit percentiles, and four-decimal display ratios in diagnostics.
For multiple operators, task speedup can differ from the ratio of the two total latency fields.
Missing cases, mismatched parameters, and failed timing records cannot create a speed bonus.

## Runtime boundary

Run the service only in a dedicated Docker container on an allocated evaluator GPU.
The CLI requires `TRITON_RL_DEDICATED_CONTAINER=1` and Docker's container marker.
Keep reference mounts read-only and restrict endpoint access to the training network.
This subprocess isolation is not a general security boundary for hostile, unrestricted code.

Each request receives a unique work directory and process group.
Timeouts and disconnected clients terminate that process group.
Worker-owned JSON files carry results, independently of candidate stdout.

The service requires ROCm PyTorch and Triton 3.3.0.
PyTorch uses its `torch.cuda` API for ROCm devices as well.
The worker controls `ROCR_VISIBLE_DEVICES` for its assigned physical GPU.
Avoid applying a second device mask to evaluator containers.
