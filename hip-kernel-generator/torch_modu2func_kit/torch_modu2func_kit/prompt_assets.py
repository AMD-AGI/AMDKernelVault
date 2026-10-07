# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prompt assets used by the conversion pipeline."""

SYSTEM_INSTRUCTION = """
You are a PyTorch functional conversion expert.

Given a PyTorch module file, generate an equivalent standalone Python file with
the following constraints:

- Output only executable Python code and brief inline comments.
- The generated file must keep the original file-level globals and expose:
  1. `module_fn`
  2. `Model`
  3. `get_inputs`
  4. `get_init_inputs`
- Preserve the initialization and forward interfaces used by the original
  module. The generated `Model` must accept the same init inputs and forward
  inputs as the original module.
- Add the optional `fn=module_fn` parameter to `Model.forward`.
- Call `fn` with all computation inputs, parameters, and constants, and return its result.
- Implement the computation with PyTorch functional operators whenever
  possible.
- The generated file must be self-contained and importable.
- Do not omit helper constants or helper functions required by
  `get_inputs` / `get_init_inputs`.
- Return only Python code with no markdown fences or prose.
"""

FEW_SHOT_EXAMPLES = """

Example 1:

pytorch module:

import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    \"\"\"
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = torch.sum(x, dim=1, keepdim=True) 
        return x

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]


pytorch functional:


import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    \"\"\"
    Applies a 3D transposed convolution operation followed by two max pooling layers and a sum operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, max pooling and sum reduction
    \"\"\"
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = F.max_pool3d(x, kernel_size=2)
    x = F.max_pool3d(x, kernel_size=3)
    x = torch.sum(x, dim=1, keepdim=True)
    return x


class Model(nn.Module):
    \"\"\"
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    \"\"\"

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        self.stride = stride
        self.padding = padding

    def forward(self, x, fn=module_fn):
        return fn(
            x, self.stride, self.padding, self.conv_transpose_parameter, self.conv_transpose_bias
        )

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]


Example 2:

pytorch module:

import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    \"\"\"
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        \"\"\"
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        \"\"\"
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x

batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]

pytorch functional:

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    \"\"\"
    Performs matrix multiplication, applies sigmoid, and sums the result.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        weight: Weight tensor of shape (hidden_size, input_size)
        bias: Bias tensor of shape (hidden_size)

    Returns:
        Output tensor of shape (batch_size, 1)
    \"\"\"
    x = F.linear(x, weight, bias)
    x = torch.sigmoid(x)
    x = torch.sum(x, dim=1, keepdim=True)
    return x


class Model(nn.Module):
    \"\"\"
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    \"\"\"

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        gemm = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)

    def forward(self, x, fn=module_fn):
        \"\"\"
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        \"\"\"
        return fn(x, self.weight, self.bias)


batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]


Example 3:

pytorch module:

import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    \"\"\"
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        \"\"\"
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        \"\"\"
        return torch.matmul(A, B)

M = 16384
N = 16

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed


pytorch functional:


import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A, B):
    \"\"\"
    Performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M).

    Args:
        A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
        B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

    Returns:
        torch.Tensor: Output matrix of shape (M, N) or (N, M)
    \"\"\"
    return torch.matmul(A, B)


class Model(nn.Module):
    \"\"\"
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    \"\"\"

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


M = 16384
N = 16


def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
"""
