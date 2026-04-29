"""
Masked linear layer for graph-derived sparse connectivity.

Why this file exists
--------------------
This file isolates the fixed-mask linear transformation used to enforce
graph-defined sparsity in compiled models. The separation makes the
package's core sparse-connection mechanism explicit and reusable without
mixing it into higher-level block or model definitions.

Role in the package
-------------------
This is an internal neural-network implementation module. It defines the
masked linear primitive that backend-specific blocks use to realize
graph-structured connectivity in PyTorch. It should contain sparse
linear-layer behavior, not compilation logic, public API handling, or
model orchestration.
"""

import math

import torch
from torch import nn
from torch.nn import functional as f


class MaskedLinear(nn.Module):
    """
    Linear layer with a fixed binary connection mask.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = True,
    ):
        super().__init__()

        if mask.shape != (out_features, in_features):
            raise ValueError(f"'mask' must have shape ({out_features}, {in_features}).")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", mask.to(dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize weights and bias.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is None:
            return

        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply masked linear transformation.
        """
        masked_weight = self.weight * self.mask
        return f.linear(x, masked_weight, self.bias)
