# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
FROM rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.7.1@sha256:0190ecdc22aa9984e37906b634c477a02bb7edb3c1ab1340df6c81dfc530fb51

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PATH=/opt/kernelgen/bin:/opt/conda/envs/py_3.10/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/triton-kernelgen/src

# Keep the paper's compiler version in an isolated, compatible wheel environment.
RUN /opt/conda/envs/py_3.10/bin/python -m venv /opt/kernelgen \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/rocm6.3 \
       "torch==2.7.0+rocm6.3" "pytorch-triton-rocm==3.3.0" \
    && python -m pip install --no-cache-dir \
       "numpy==1.26.4" "requests==2.32.5" "setuptools==80.9.0" "wheel==0.45.1"

WORKDIR /opt/triton-kernelgen
COPY . /opt/triton-kernelgen
RUN python -m pip install --no-deps --no-build-isolation -e . \
    && python -c 'import torch, triton; assert torch.version.hip; assert triton.__version__ == "3.3.0"'

ENTRYPOINT ["triton-kernelgen"]
CMD ["--help"]
