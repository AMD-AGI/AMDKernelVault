# Correspondence with the paper

The specification comes from
[AMDKernelVault](https://arxiv.org/html/2609.12471), Section 3.1, Figure 1, Section 5, and the dataset appendices.

| Reported element | Release implementation |
| --- | --- |
| Stage 1 creates a function-style PyTorch reference. | The existing module-conversion kit can produce the reference. This package revalidates it on the external case suites. |
| Stage 2 uses generator, evaluator, and reflector roles. | Generation and reflection use separate model requests. The evaluator performs AMD compilation, correctness, and timing checks. |
| PyTorch reference grounds correctness. | The original and functional implementations must agree before candidate generation. |
| Diverse Triton generation models | Caller-selected service endpoints support the three model families named in the paper. Both requested and returned identifiers are recorded. |
| Triton 3.3.0 on AMD hardware | Workers require that Triton version, ROCm PyTorch, and a matching `gfx942` or `gfx950` device. |
| Approximately ten correctness and three performance configurations | The caller supplies fixed suites. The record preserves their actual counts and tensor metadata. |
| Task-specific tolerance or fallback | Each case can override tolerance. Defaults are `rtol=1e-4`, `atol=1e-3`, and `equal_nan=False`. |
| Shape-, stride-, and dtype-aware validation | The safe tensor transport preserves supported storage/view relationships and records actual inputs. |
| Reflection after failure | Later generation uses candidate feedback and a separate reflection plan. |
| Latency-guided refinement | Additional requested variants receive measured latency feedback. Correctness remains mandatory. |
| Retention depends on AMD execution | Retention requires successful compilation, execution, correctness, and measurement. It does not require speedup above one. |

The prose describes standardized references as generation inputs.
Figure 1 also supplies the original module to the generator.
This release supplies both, while keeping the standardized function as the execution interface.

## Explicit implementation choices

The paper does not prescribe `module_fn`, a wrapper signature, or this case-provider API.
Those interfaces make the release executable and auditable.
The wrapper must forward arguments and state without doing tensor computation outside `module_fn`.
Mutable tasks and unsupported tensor subclasses fail explicitly.

The caller chooses attempt limits and retained-variant counts.
The paper's later three-turn RL and three-iteration benchmark settings do not define those limits.
Sampling temperature, token budget, random seed, and timeout are release settings.
Their configured values appear in run records.
The container launcher also records its selected image reference and local image ID.
Native runs can leave container identity empty.

The generator never changes accepted references, case membership, or tolerances in response to candidate failures.
Declared helper files receive source snapshots and fingerprints.
Undeclared external dependencies remain outside that source-identity claim.

## Measurement scope

The timer uses Triton 3.3.0 `do_bench` with a 25 ms warmup and a 100 ms measurement budget.
It requests the median GPU-event latency.
The Triton timer supplies its cache-flush and iteration-estimation behavior.
Launch probes run during separate validation calls and remain disabled during timing.
Initial compilation, autotuning warmup, and input cloning remain outside the measured callable.
Allocations inside `module_fn` remain inside it.

Observed Triton launches do not prove complete causal dependence of the returned result on those launches.
Use trusted, sufficiently varied reference cases and review retained artifacts.
The release does not claim a general defense against adversarial generated code.

The archive contains related generator implementations with different configurations.
This release implements the documented procedure rather than claiming one recovered exact historical revision.
It does not establish the paper's full corpus counts, historical model use, or benchmark scores.
