[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat)](LICENSE)

# hip-kernel-llm-lab

> A HIP kernel LLM training framework covering continued pretraining, supervised fine-tuning, and reinforcement learning.

`hip-kernel-llm-lab` is a repository for training and evolving LLMs toward HIP
kernel generation, understanding, optimization, and related engineering tasks.
The project is organized around three training stages and the supporting kernel
data-generation/evaluation tooling needed to run them:

1. Continued pretraining (`cpt`)
2. Supervised fine-tuning (`sft`)
3. Reinforcement learning (`rl`)
4. HIP kernel data generation and optimization (`hip-kernel-generator`)
5. Triton kernel generation and verification (`triton-kernel-generator`)

## Repository Layout

```text
hip_kernel_llm_lab/
├── cpt/                    # continued pretraining launchers and configs
├── sft/                    # supervised fine-tuning launchers and configs
├── rl/                     # HIP and Triton reinforcement learning stacks
├── hip-kernel-generator/   # LLM-assisted PyTorch/HIP data generation tools
├── triton-kernel-generator/ # Triton generation with fixed AMD verification cases
├── examples/               # example data, recipes, and usage samples
├── .github/                # repository metadata such as CODEOWNERS
├── LICENSE
└── README.md
```

## Training Scope

### `cpt`

The continued pretraining stage is intended for domain adaptation on HIP
kernel-related corpora, including source code, optimization patterns, compiler
knowledge, runtime usage, and associated engineering context.

### `sft`

The supervised fine-tuning stage is intended for instruction-following behavior
on HIP kernel tasks such as code generation, code transformation, debugging,
optimization suggestions, and reasoning over kernel implementations.

### `rl`

The reinforcement learning stage is intended for further improving model
behavior using task-specific rewards, such as correctness, compilability,
performance-oriented preferences, or other training objectives relevant to HIP
kernel workflows.

### `hip-kernel-generator`

The generator tools build and refine data for HIP kernel training workflows.
They cover:

- PyTorch module to PyTorch functional conversion
- PyTorch module to HIP kernel generation
- HIP kernel to optimized HIP kernel transformation

These tools are useful for producing SFT examples, RL seed data, and artifacts
that can be inspected or replayed during model development.

## Planned Workflow

The intended high-level training flow is:

1. Use continued pretraining to adapt the base model to HIP kernel and
   GPU-system-specific knowledge.
2. Use supervised fine-tuning to teach task-oriented instruction behavior.
3. Use reinforcement learning to optimize for downstream quality signals such as
   correctness, preference alignment, and performance-related objectives.

## Triton Kernel Generation

[`triton-kernel-generator/`](triton-kernel-generator/README.md) implements the
TritonKernelGen construction workflow described in
[AMDKernelVault](https://arxiv.org/html/2609.12471).
It validates external PyTorch references, generates Triton candidates, and uses
AMD correctness and latency feedback for further attempts.
Prepared references and case collections remain external.

The [HIP alignment note](hip-kernel-generator/PAPER_ALIGNMENT.md) explains the
existing HIP workflow, its corrected verification paths, and its coverage limits.

## Dataset Resources

The training datasets are maintained separately from this code repository.

- Hugging Face dataset: [`amd/hip_llm_dataset`](https://huggingface.co/datasets/amd/hip_llm_dataset)
- Local dataset directory convention: `hip_llm_dataset/`

Download the dataset from the repository root:

```bash
export HF_TOKEN=...
huggingface-cli download amd/hip_llm_dataset \
  --repo-type dataset \
  --local-dir hip_llm_dataset
```

The dataset is organized by training stage:

- CPT: `hip_llm_dataset/cpt/`
- SFT: `hip_llm_dataset/sft/`
- RL: `hip_llm_dataset/rl/`

Common dataset names used by this repository include:

- `pretrain_hip`: continued pretraining corpus
- `FIM_hip`: FIM / continuation-style code training data
- `merged_110k_if_reasoning`: SFT messages dataset
- `rl_raw_hard_3k`: raw hard RL JSON data
- `rl_raw_normal_10k`: raw normal RL JSON data
- `rl_mi300x_verl_filtered`: direct-use MI300X veRL parquet data

See the stage-specific README files for exact training commands and data
registration details.

## Quick Start

### Continued Pretraining

```bash
cd cpt
bash run_train_cpt.sh
```

The default CPT YAMLs are under `cpt/train_yamls/`.

### Supervised Fine-Tuning

```bash
cd sft
bash run_train_full_sft.sh
```

The default SFT YAML is `sft/train_yamls/qwen3-14b-full-sft-hip.yaml`.
Update its `dataset`, `model_name_or_path`, and `output_dir` fields before
launching a new run.

### Reinforcement Learning

Start the HIP evaluation server first:

```bash
cd rl/rl4kernel_hip/hip_kernel_evaluation_server
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./setup_server_req_deploy_hip2hip_batch.sh
```

Then launch HIP2HIP single-turn RL training:

```bash
cd rl/rl4kernel_hip
export WANDB_API_KEY=...
bash scripts/train/react_single_turn_v1_hip2hip.sh \
  --train hip_llm_dataset/rl/rl_data_v01_mi300x_verl_filtered.parquet \
  --val hip_llm_dataset/rl/rl_data_v01_mi300x_verl_filtered.parquet \
  --sf-url http://host:8080/run_code
```

### HIP Kernel Data Generation

Install the specific generator package you need:

```bash
cd hip-kernel-generator
pip install -e ./torch_modu2func_kit[dev]
pip install -e ./torch2hip_kit[dev]
pip install -e ./py_hip_kernel2kernel_kit[dev]
```

See `hip-kernel-generator/README.md` for package-specific CLI examples.

### Multi-turn Triton Reinforcement Learning

[`rl/rl4kernel_triton/`](rl/rl4kernel_triton/README.md) provides the multi-turn
Triton training procedure described in
[AMDKernelVault](https://arxiv.org/html/2609.12471).
It includes a Slime adapter, a ROCm execution service, pinned container recipes,
and a guide for the 32-MI325X training configuration.
Prepared prompts, reference tests, and checkpoints remain external.

## Current Status

This repository contains runnable assets for all main stages:

- `cpt/` and `sft/` provide LLaMA-Factory based launchers and training YAMLs.
- `rl/rl4kernel_hip/` provides a veRL-based HIP kernel RL stack, reward code,
  dataset conversion docs, training launchers, and evaluation server tooling.
- `rl/rl4kernel_triton/` provides Slime-based multi-turn Triton RL and an AMD
  execution service, with code-only setup and validation tools.
- `hip-kernel-generator/` provides installable tools for PyTorch-to-HIP and
  HIP-to-HIP generation/optimization workflows.
- Stage-specific READMEs contain the operational details for each workflow.

## Code Ownership

Current repository ownership is defined in [`.github/CODEOWNERS`](.github/CODEOWNERS).

Temporary owners for the repository are:

- `@AMD-AGI/AI-Algorithm`
- `@liuji` `@Majumder Saptarshi` `@zepingli` `@chushi` `@zihao` `@puyuan`

Because the code for each training stage has not been added yet, ownership is
currently maintained at the repository level. More granular ownership can be
introduced after `cpt`, `sft`, and `rl` receive their stage-specific code.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull
request. Keep changes focused on one training stage or generator workflow,
update the relevant README, and include validation commands or run details.

## Security

Do not open public issues for suspected vulnerabilities. Follow the private
reporting process in [SECURITY.md](SECURITY.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
