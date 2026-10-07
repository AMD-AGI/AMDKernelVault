# Multi-turn Triton RL on AMD GPUs

This package implements the Triton training procedure in
[AMDKernelVault](https://arxiv.org/html/2609.12471), Sections 4.3 and Appendices E and G.
One policy generates code, receives execution feedback, and produces a revision.
Slime supplies distributed GRPO training and SGLang generation.
A separate ROCm service checks correctness and measures performance.

The release contains code, configuration, tests, and container recipes.
Checkpoints, prepared prompts, reference kernels, and dataset-generation code remain external.

## Training procedure

The default [configuration](src/triton_rl/paper.json) follows the paper:

| Setting | Value |
| --- | --- |
| Policy | Qwen3-8B, initialized from the external SFT checkpoint |
| Training hardware | 32 MI325X GPUs |
| Prompt batch | 64 |
| Samples per prompt | 8 |
| Maximum turns | 3 |
| Turn discount | 0.6 |
| Sequence budget | 16,384 tokens, including prompt and feedback |
| Learning rate | 1e-6 |
| KL loss coefficient | 0.01 |
| PPO clipping bounds | 0.2 / 0.28 |
| Reward normalization | Group mean subtraction, without standard-deviation division |
| Reported epoch target | 2.7, subject to the externally supplied curriculum schedule |

Each turn receives:

```text
r = 0.4 * compiled + 1.0 * correct + performance_bonus
performance_bonus = min(0.375 * log2(speedup)^2, 1.5)
G = r1 + 0.6 * r2 + 0.36 * r3
```

The performance bonus requires correctness and a finite speedup of at least one.
The turn reward ranges from zero to 2.9.
The trajectory return uses the unnormalized discounted sum.
The next turn can improve performance after correctness succeeds.

The adapter preserves sampled token IDs and their conditioning context.
Generated reflection and code receive a loss mask of one.
Execution feedback receives a loss mask of zero.
The synchronous driver updates the serving weights before collecting another training batch.

## 1. Build the containers

Run these commands from this directory:

```bash
docker build -f docker/Dockerfile.trainer -t amdkernelvault-triton-trainer:local .
docker build -f docker/Dockerfile.evaluator -t amdkernelvault-triton-evaluator:local .
```

The trainer recipe uses this archived runtime tag, pinned by its current manifest digest:

```text
rlsys/slime:ubuntu22.04_rocm6.3.4_sglang0.5.0rc0_megatron_ray2.47.1_apex_torch-memory-saver0.0.8
sha256:80fe6e12ba7969e3169db8361e9cb89d67c714cf24288eaf1cda104211c241f5
```

The recipe preserves its patched AMD Megatron, Apex, and vLLM allocator libraries.
It pins Slime to `5f781608ba28738fc73f44fa12efef1cdb408ee2`.
The evaluator uses a separate environment with `torch==2.7.0+rocm6.3` and `pytorch-triton-rocm==3.3.0`.
This separation preserves the paper's Triton compiler version without replacing trainer libraries.

These Dockerfiles define the release environments.
The archive does not establish an immutable image digest for the original experiment.
The containers require validation on the target GPU cluster before a training run.

## 2. Start the execution workers

Provide external reference files that follow the [execution protocol](EXECUTION_PROTOCOL.md).
Allocate a dedicated GPU for each worker.
Expose worker endpoints only on the trusted training network.

```bash
bash scripts/run_container.sh evaluator \
  --image amdkernelvault-triton-evaluator:local \
  --references /absolute/path/to/reference_files \
  --gpu 0 --port 8080 --detach
```

For additional workers, supply distinct GPU indices, ports, and container names.
Each worker serializes candidate execution on its assigned GPU.
The worker measures reference and candidate latency on that same GPU.
Training uses the separate 32-GPU allocation.

## 3. Start the trainer containers and Ray

Start the same trainer image on four nodes with eight MI325X GPUs per node.
Use the same container paths for shared files on every node.

```bash
bash scripts/run_container.sh trainer \
  --image amdkernelvault-triton-trainer:local \
  --models /absolute/path/to/checkpoints \
  --data /absolute/path/to/prepared_prompts \
  --output /absolute/path/to/output \
  --cache /absolute/path/to/cache
```

Inside each container, check its runtime:

```bash
python -m triton_rl.preflight trainer
```

On the head node, start Ray with its allocated network address:

Set `HEAD_IP` to the head node's address inside each trainer container.

```bash
bash scripts/start_ray.sh head --head-ip "$HEAD_IP"
```

On each other node, connect its Ray worker:

Set `NODE_IP` to that worker's address.

```bash
bash scripts/start_ray.sh worker --head-ip "$HEAD_IP" --node-ip "$NODE_IP"
```

The launcher colocates training and generation across the 32 GPUs.
Colocation, tensor parallelism of two, and seed 42 are release setup choices.
They are not additional experimental claims from the paper.

## 4. Convert the external SFT checkpoint

Run the conversion once inside a trainer container:

```bash
bash scripts/convert_checkpoint.sh /models/qwen3-8b-sft /output/qwen3-8b-sft-megatron
```

This converts weights with Slime's pinned conversion tool.
It does not prepare or modify training data.
Keep this initial SFT checkpoint as the reference policy throughout the curriculum.

## 5. Inspect and submit the training job

Prepared prompt files must supply `prompt` and `reward_model` fields.
The prompt contains chat messages for the checkpoint's chat template.
The label contains `ground_truth.filename`, relative to the execution worker's reference directory.
Optional `ground_truth.atol` and `ground_truth.rtol` specify task tolerances.
Otherwise, the worker uses `atol=1e-3`, `rtol=1e-4`, and `equal_nan=False`.
Responses must use one complete Python code fence or one complete `<answer>` region.

Set the execution endpoints and the exact rollout budget for the selected curriculum phase:

Set `PHASE_ROLLOUT_STEPS` to the number of updates allocated to that phase.

```bash
export TRITON_RL_SANDBOX_URLS="http://executor-a:8080,http://executor-b:8080"
export RAY_JOB_ADDRESS="http://${HEAD_IP}:8265"

bash scripts/run_training.sh \
  --hf-checkpoint /models/qwen3-8b-sft \
  --reference-checkpoint /output/qwen3-8b-sft-megatron \
  --train-data /data/phase1.parquet \
  --output /output/phase1 \
  --rollout-steps "$PHASE_ROLLOUT_STEPS" \
  --dry-run
```

Inspect the printed command and budget.
Remove `--dry-run` to submit the job.
The driver saves the final successful update even when its index does not match the periodic save interval.

For a resumed run, add `--resume /output/phase1`.
The requested rollout count then means additional updates after the saved checkpoint.
For the next curriculum phase, also add `--new-curriculum-phase` and supply its new prompt file and output directory.
This preserves model and optimizer state while starting the new prompt phase at its beginning.

The paper's curriculum uses L1, then mixed L1/L2/L3, then harder L2/L3 variants.
Its rounded phase counts do not define the exact rollout allocation.
Supply the retained phase membership and budgets from the external experiment configuration.
For one fixed phase, `--effective-prompts N` can replace `--rollout-steps`.
It rounds the 2.7-epoch target up to a complete 64-prompt batch and reports the resulting effective epochs.
Do not interpret that convenience calculation as the paper's exact three-phase schedule.

## Development and validation

Install the package and CPU test dependencies:

```bash
python -m pip install -e '.[test,dev]'
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
ruff check src tests scripts
```

CPU tests cover rewards, exact token history, feedback masks, execution errors, and the reference-file protocol.
They do not establish GPU performance or reproduce the reported benchmark scores.
Slime training-time reward evaluation is separate from the paper's GEAK OptimAgent-v2 benchmark protocol.

See [runtime settings](RUNTIME.md) for implementation defaults and dependency details.
See [attribution](NOTICE.md) for upstream code sources.
