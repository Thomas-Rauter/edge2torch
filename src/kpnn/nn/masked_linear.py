import math

import torch
from torch import nn
from torch.nn import functional as F


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
            raise ValueError(
                "'mask' must have shape "
                f"({out_features}, {in_features})."
            )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

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
        return F.linear(x, masked_weight, self.bias)
