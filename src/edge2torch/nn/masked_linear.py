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
from typing import cast

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
            raise ValueError(
                f"'mask' must have shape ({out_features}, {in_features})."
            )

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
        mask = cast(torch.Tensor, self.mask)
        masked_weight = self.weight * mask

        return f.linear(x, masked_weight, self.bias)


CONSTRAINT_UNCONSTRAINED = 0
CONSTRAINT_POSITIVE = 1
CONSTRAINT_NEGATIVE = 2
CONSTRAINT_FIXED = 3


def constraint_name_to_code(name: str) -> int:
    """
    Convert a constraint name to its internal integer code.
    """
    normalized_name = name.strip().lower()

    if normalized_name == "unconstrained":
        return CONSTRAINT_UNCONSTRAINED

    if normalized_name == "positive":
        return CONSTRAINT_POSITIVE

    if normalized_name == "negative":
        return CONSTRAINT_NEGATIVE

    if normalized_name == "fixed":
        return CONSTRAINT_FIXED

    raise ValueError(f"Unsupported constraint '{name}'.")


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable inverse of softplus for positive inputs.
    """
    eps = torch.finfo(x.dtype).eps
    x = x.clamp_min(eps)

    return x + torch.log(-torch.expm1(-x))


class ConstrainedMaskedLinear(nn.Module):
    """
    Linear layer with a fixed binary connection mask and optional edge-wise
    weight constraints.

    The layer stores trainable weights as unconstrained latent parameters and
    transforms them during the forward pass to obtain effective weights.

    Supported constraints are:

    - unconstrained: ``w = raw_weight``
    - positive: ``w = softplus(raw_weight)``
    - negative: ``w = -softplus(raw_weight)``
    - fixed: ``w = fixed_weight``

    ``initial_weight`` may contain ``NaN`` values. A finite value initializes
    the corresponding effective edge weight from user-provided metadata. A
    ``NaN`` value means that the edge uses the default layer initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        initial_weight: torch.Tensor | None = None,
        constraint: torch.Tensor | None = None,
        bias: bool = True,
    ):
        super().__init__()

        expected_shape = (out_features, in_features)

        if mask.shape != expected_shape:
            raise ValueError(
                f"'mask' must have shape ({out_features}, {in_features})."
            )

        if (
            initial_weight is not None
            and initial_weight.shape != expected_shape
        ):
            raise ValueError(
                "'initial_weight' must have shape "
                f"({out_features}, {in_features})."
            )

        if constraint is not None and constraint.shape != expected_shape:
            raise ValueError(
                f"'constraint' must have shape ({out_features}, {in_features})."
            )

        self.in_features = in_features
        self.out_features = out_features

        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", mask.to(dtype=torch.float32))

        if initial_weight is None:
            initial_weight = torch.full(
                expected_shape,
                fill_value=float("nan"),
                dtype=torch.float32,
            )

        if constraint is None:
            constraint = torch.full(
                expected_shape,
                fill_value=CONSTRAINT_UNCONSTRAINED,
                dtype=torch.long,
            )

        self.register_buffer(
            "initial_effective_weight",
            initial_weight.to(dtype=torch.float32),
        )
        self.register_buffer("constraint", constraint.to(dtype=torch.long))
        self.register_buffer(
            "fixed_weight",
            torch.zeros(out_features, in_features, dtype=torch.float32),
        )

        self._validate_fixed_weights_have_initial_values()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize latent weights, fixed weights, and bias.

        Finite entries in ``initial_effective_weight`` are used as explicit
        edge initializations. Missing entries use default Kaiming-uniform
        initialization at the effective-weight level.
        """
        initial_effective_weight = cast(
            torch.Tensor,
            self.initial_effective_weight,
        )
        constraint = cast(torch.Tensor, self.constraint)

        default_effective_weight = torch.empty(
            self.out_features,
            self.in_features,
            dtype=self.raw_weight.dtype,
            device=self.raw_weight.device,
        )
        nn.init.kaiming_uniform_(default_effective_weight, a=math.sqrt(5))

        has_initial_weight = torch.isfinite(initial_effective_weight)
        target_effective_weight = torch.where(
            has_initial_weight,
            initial_effective_weight.to(
                dtype=self.raw_weight.dtype,
                device=self.raw_weight.device,
            ),
            default_effective_weight,
        )

        positive = constraint == CONSTRAINT_POSITIVE
        negative = constraint == CONSTRAINT_NEGATIVE
        fixed = constraint == CONSTRAINT_FIXED

        with torch.no_grad():
            self.raw_weight.copy_(target_effective_weight)

            constrained_magnitude = target_effective_weight.abs()
            constrained_raw_weight = _softplus_inverse(constrained_magnitude)

            self.raw_weight[positive] = constrained_raw_weight[positive]
            self.raw_weight[negative] = constrained_raw_weight[negative]

            fixed_weight = cast(torch.Tensor, self.fixed_weight)
            fixed_weight.zero_()
            fixed_weight[fixed] = target_effective_weight[fixed]

        if self.bias is None:
            return

        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    @property
    def effective_weight(self) -> torch.Tensor:
        """
        Return the effective constrained weight matrix.
        """
        constraint = cast(torch.Tensor, self.constraint)
        fixed_weight = cast(torch.Tensor, self.fixed_weight)

        effective_weight = self.raw_weight.clone()

        positive = constraint == CONSTRAINT_POSITIVE
        negative = constraint == CONSTRAINT_NEGATIVE
        fixed = constraint == CONSTRAINT_FIXED

        effective_weight[positive] = f.softplus(self.raw_weight[positive])
        effective_weight[negative] = -f.softplus(self.raw_weight[negative])
        effective_weight[fixed] = fixed_weight[fixed]

        return effective_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply constrained masked linear transformation.
        """
        mask = cast(torch.Tensor, self.mask)
        masked_weight = self.effective_weight * mask

        return f.linear(x, masked_weight, self.bias)

    def _validate_fixed_weights_have_initial_values(self) -> None:
        """
        Validate that fixed edges have explicit initial weights.
        """
        initial_effective_weight = cast(
            torch.Tensor,
            self.initial_effective_weight,
        )
        constraint = cast(torch.Tensor, self.constraint)
        mask = cast(torch.Tensor, self.mask)

        fixed_edges = (constraint == CONSTRAINT_FIXED) & mask.bool()
        fixed_without_initial_weight = fixed_edges & ~torch.isfinite(
            initial_effective_weight
        )

        if fixed_without_initial_weight.any():
            raise ValueError(
                "Edges with constraint 'fixed' require finite "
                "'initial_weight' values."
            )
