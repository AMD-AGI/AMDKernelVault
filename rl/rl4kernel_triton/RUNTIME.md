# Runtime and launch settings

The Dockerfiles pin release environments and preserve AMD-specific dependencies.
Their image digests identify the current registry objects used by these recipes.
They do not establish the exact immutable images used for the reported experiments.

## Trainer

The trainer uses the archived ROCm 6.3.4 Slime image referenced in the main guide.
It retains the image's patched SGLang, Megatron-LM, Apex, and vLLM allocator libraries.
The image metadata records SGLang commit `8ecf6b9d2480c3f600826c7d8fef6a16ed603c3f`.
It records Megatron commit `48406695c4efcf1026a7ed70bb390793918dd97b` with AMD patches.

The recipe installs Slime commit `5f781608ba28738fc73f44fa12efef1cdb408ee2` without replacing GPU dependencies.
That revision already contains the HIP allocation and offload paths.
Those paths require `vllm.device_allocator.cumem.CuMemAllocator`.
Checkpoint conversion also requires `mbridge.AutoBridge` and `slime_plugins.mbridge`.
The preflight checks these imports and API symbols before training.

The principal container paths are:

| Path | Purpose |
| --- | --- |
| `/opt/slime` | Pinned Slime source |
| `/opt/triton-rl` | This release |
| `/workspace/Megatron-LM` | The inherited patched Megatron tree |
| `/models` | Read-only external SFT checkpoints |
| `/data` | Read-only external prompt files |
| `/output` | Converted checkpoints and training outputs |
| `/cache` | Writable model and tokenizer cache |

All trainer nodes must resolve these shared paths consistently.
The container exposes `/dev/kfd` and `/dev/dri` and uses the host's video/render groups.
It retains the archived 128 GiB shared-memory allocation, unlimited locked memory, and 64 MiB stack limit.
The trainer preserves the runtime's `SYS_PTRACE` and seccomp settings.

The launcher uses these additional implementation defaults:

| Setting | Default |
| --- | --- |
| Placement | Four nodes, eight GPUs each, colocated actor and generation |
| Tensor parallelism | 2 |
| Generation GPUs per engine | 2 |
| Pipeline/context/expert parallelism | 1 |
| Sampling temperature | 1.0 |
| Seed | 42 |
| SGLang static memory fraction | 0.4 |
| Optimizer | Adam, beta1 0.9, beta2 0.98, weight decay 0.1 |
| Learning-rate schedule | Constant |
| Entropy coefficient | 0.0 |
| Dropout | 0.0 |
| Recompute | Full, uniform, one layer |
| Per-GPU token budget | 16,384 |
| Periodic checkpoint interval | 20 rollout updates |

These settings make the release configuration explicit.
The paper does not independently specify all of these implementation values.
The authoritative reported training values appear in `paper.json` and the main guide.

The custom driver follows Slime's synchronous update order.
It also saves the last completed update and supports explicit curriculum transitions.
`--rollout-steps` means additional updates, including when resuming.
The launcher reads the saved iteration and constructs Slime's absolute stop index.
Resume restores the optimizer step count and extends the constant scheduler horizon to that stop index.
Same-phase resume also requires the saved prompt cursor.
The driver rejects a budget that contains no remaining updates.

Slime numbers rollout checkpoints from zero.
The trainer build applies a narrow Megatron compatibility patch that accepts iteration zero.
It preserves negative-iteration rejection and the existing positive and release checkpoint behavior.
The patcher rejects unknown source layouts and changes no model or optimizer state.

## Evaluator

The evaluator base is pinned separately:

```text
rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.7.1
sha256:0190ecdc22aa9984e37906b634c477a02bb7edb3c1ab1340df6c81dfc530fb51
```

The recipe creates an isolated environment without inherited Python packages.
It installs the official `torch==2.7.0+rocm6.3` and `pytorch-triton-rocm==3.3.0` wheel pair.
The PyTorch package metadata requires that Triton version.
This pairing avoids changing the trainer's generation-engine dependencies.

The evaluator also pins `aiohttp==3.12.15`, `pytest==8.4.2`, and `numpy==1.26.4`.
Both build environments use `setuptools==80.9.0` and `wheel==0.45.1`.
The verifier requires no private evaluator package or sandbox image archive.

The default execution timeout is 1,500 seconds.
The client's default request timeout is 1,560 seconds.
Use `TRITON_RL_SANDBOX_TIMEOUT_SECONDS` if worker timeouts require a different client budget.
Reference and candidate timings use the same assigned GPU.
See `EXECUTION_PROTOCOL.md` for the reference-test and timing interfaces.

## Preflight and observability

Run the appropriate check inside each container:

```bash
python -m triton_rl.preflight trainer
python -m triton_rl.preflight evaluator --gpu 0
```

`--no-gpu` skips hardware queries but retains HIP, version, import, and API checks.
Neither mode measures kernel speed or verifies a complete distributed training step.
The trainer checks its source commit before job submission.

Each trajectory stores turn results, rewards, token counts, and its stopping reason in `Sample.metadata["triton_rl"]`.
This records generated tokens separately from feedback tokens.
Execution diagnostics retain test outcomes and matched timing cases.
Slime supplies training loss, KL, gradient, and checkpoint logs.
Enable optional W&B logging with `--wandb-project` and environment-based authentication inside the containers.
The launcher does not insert credentials into the Ray job command.

CPU tests validate contracts and failure handling.
GPU compilation, image builds, distributed startup, and end-to-end training remain separate validation steps.
