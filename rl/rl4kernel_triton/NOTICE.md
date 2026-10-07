# Attribution

New release code is copyright 2026 Advanced Micro Devices, Inc.
The code uses the repository's Apache License 2.0.

The synchronous training driver derives from `train.py` in
[THUDM/slime](https://github.com/THUDM/slime/tree/5f781608ba28738fc73f44fa12efef1cdb408ee2).
The Qwen3-8B architecture settings derive from that revision's `scripts/models/qwen3-8B.sh`.
Slime uses Apache License 2.0.
Its source remains an external pinned dependency.

The rollout and execution interfaces follow the AMD Triton RL source and the published AMDKernelVault procedure.
The execution verifier is independently implemented for the external pytest reference-file protocol.
This package does not redistribute GEAK-eval, datasets, model weights, or reference kernels.

Container dependencies retain their own licenses and notices.
The release does not relicense those dependencies.
