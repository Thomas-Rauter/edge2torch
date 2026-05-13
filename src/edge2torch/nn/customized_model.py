"""
PyTorch wrapper for optional post-compilation model customization.

Why this file exists
--------------------
This file isolates the internal PyTorch module used to add optional
downstream architectural components around a compiled edge2torch model. The
separation keeps the public customize_model() API thin and prevents
model-wrapper implementation details from being mixed into API
validation or orchestration code.

Role in the package
-------------------
This is an internal neural-network implementation module. It defines the
runtime wrapper that applies optional activation, dropout, and head
components while preserving access to the wrapped compiled model where
needed. It should contain model-execution behavior, not public API
validation or higher-level customization orchestration.
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn


class CustomizedEdgeModel(nn.Module):
    """
    PyTorch wrapper around a compiled edge2torch model with optional downstream
    architectural components.

    Components are applied to the output of the compiled model. The wrapper
    does not modify the compiled graph structure or insert modules inside
    backend-specific computations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        activation: nn.Module | None = None,
        dropout: float | int | None = None,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.activation = activation
        self.dropout_layer = (
            nn.Dropout(float(dropout)) if dropout is not None else None
        )
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the compiled model and optional downstream
        components.
        """
        x = self.base_model(x)

        if self.activation is not None:
            activation = cast(nn.Module, self.activation)
            x = activation(x)

        if self.dropout_layer is not None:
            dropout_layer = cast(nn.Module, self.dropout_layer)
            x = dropout_layer(x)

        if self.head is not None:
            head = cast(nn.Module, self.head)
            x = head(x)

        return x

    def _edge2torch_get_feedforward_layer_block(
        self,
        layer_name: str,
    ) -> nn.Module:
        """
        Delegate feedforward layer-block access to the wrapped compiled model.
        """
        method_name = "_edge2torch_get_feedforward_layer_block"

        if not hasattr(self.base_model, method_name):
            raise AttributeError(
                f"'{type(self.base_model).__name__}' object has no attribute "
                f"'{method_name}'"
            )

        get_layer_block = cast(
            Callable[[str], nn.Module],
            getattr(self.base_model, method_name),
        )

        return get_layer_block(layer_name)
