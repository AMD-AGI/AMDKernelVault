# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib.util
from pathlib import Path

SYSTEM_INSTRUCTION = """
Given the following PyTorch code, generate an equivalent HIP implementation with the following constraints:

- Output only HIP code and inline comments. No explanations or discussion.
- The HIP implementation must consist of three parts:
  1. HIP kernel
  2. Kernel launcher
  3. Python binding exposing a `forward` function
- Strictly follow the style and structure of the provided examples.
- The exposed `forward` signature must be callable from the paired PyTorch functional code.
- Preserve numerical semantics of the original PyTorch module.
- Prefer implementations that are both correct and performant on ROCm.
""".strip()

# These few-shot cases are vendored directly into `torch2hip_kit` so the package
# has no runtime dependency on sibling prompt assets.
FEW_SHOT_EXAMPLES = '''
### Example 1:

pytorch:

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = torch.relu(x)
        return x

batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]


hip:

#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define TILE_SIZE 16

template <typename scalar_t>
__global__ void linear_subtract_multiply_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float subtract_value,
    const float multiply_value) {

    __shared__ scalar_t weight_shared[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t input_shared[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    scalar_t acc = 0;

    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < batch_size && t * TILE_SIZE + threadIdx.y < in_features) {
            input_shared[threadIdx.x][threadIdx.y] =
                input[row * in_features + t * TILE_SIZE + threadIdx.y];
        } else {
            input_shared[threadIdx.x][threadIdx.y] = 0;
        }

        if (col < out_features && t * TILE_SIZE + threadIdx.x < in_features) {
            weight_shared[threadIdx.y][threadIdx.x] =
                weight[col * in_features + t * TILE_SIZE + threadIdx.x];
        } else {
            weight_shared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc = fma(input_shared[threadIdx.x][k], weight_shared[threadIdx.y][k], acc);
        }

        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        acc += bias[col];
        acc = (acc - subtract_value) * multiply_value;
        acc = acc > 0 ? acc : 0;
        output[row * out_features + col] = acc;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value) {

    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks(
        (batch_size + TILE_SIZE - 1) / TILE_SIZE,
        (out_features + TILE_SIZE - 1) / TILE_SIZE
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "linear_subtract_multiply_relu_kernel", ([&] {
        linear_subtract_multiply_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            subtract_value,
            multiply_value
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Linear transform with subtract, multiply and ReLU forward");
}

### Example 2:

pytorch:

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling,
    another global average pooling, and then calculates the mean.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose.bias = nn.Parameter(
            self.conv_transpose.bias
            + torch.randn(
                self.conv_transpose.bias.shape,
                device=self.conv_transpose.bias.device,
                dtype=self.conv_transpose.bias.dtype,
            ) * 0.02
        )
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.multiplier
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = torch.mean(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]


hip:

#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

__global__ void atomic_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ global_sum,
    int total_elements) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

at::Tensor module_fn(
    at::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    at::Tensor conv_transpose,
    at::Tensor conv_transpose_bias,
    double multiplier
) {
    at::Tensor y = at::conv_transpose2d(
        x,
        conv_transpose,
        conv_transpose_bias,
        {stride, stride},
        {padding, padding},
        {output_padding, output_padding},
        1,
        {1, 1}
    );

    y = y * multiplier;

    int64_t total_elements = y.numel();
    at::Tensor y_flat = y.contiguous().view({total_elements});

    auto options = torch::TensorOptions().device(y.device()).dtype(torch::kFloat);
    at::Tensor global_sum_tensor = torch::zeros({1}, options);

    const int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    gridSize = gridSize > 1024 ? 1024 : gridSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    atomic_reduce_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        y_flat.data_ptr<float>(),
        global_sum_tensor.data_ptr<float>(),
        total_elements
    );

    hipDeviceSynchronize();

    float global_sum = global_sum_tensor.item<float>();
    float mean_val = global_sum / static_cast<float>(total_elements);
    return torch::tensor(mean_val, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module function");
}
'''.strip()

DEFAULT_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION
DEFAULT_FEW_SHOT_EXAMPLES = FEW_SHOT_EXAMPLES


def _load_symbol_from_python_file(file_path: Path, symbol_name: str) -> str | None:
    if not file_path.exists():
        return None

    module_name = f"torch2hip_prompt_assets_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, symbol_name, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_prompt_assets(
    instruction_file: Path | None = None,
    few_shot_file: Path | None = None,
) -> tuple[str, str]:
    system_instruction = (
        _load_symbol_from_python_file(instruction_file, "hip_generation_req")
        if instruction_file is not None
        else None
    )
    few_shot_examples = (
        _load_symbol_from_python_file(few_shot_file, "few_shot_code_instructions")
        if few_shot_file is not None
        else None
    )

    return (
        system_instruction or SYSTEM_INSTRUCTION,
        few_shot_examples or FEW_SHOT_EXAMPLES,
    )
