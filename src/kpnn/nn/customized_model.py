import torch
from torch import nn


class CustomizedKPNNModel(nn.Module):
    """
    PyTorch wrapper around a compiled KPNN model with optional downstream
    architectural components.
    """

    def __init__(
        self,
        base_model: nn.Module,
        activation: nn.Module | None = None,
        dropout: float | int | None = None,
        head: nn.Module | None = None,
    ):
        super().__init__()

        self.base_model = base_model
        self.activation = activation
        self.dropout = (
            nn.Dropout(float(dropout))
            if dropout is not None
            else None
        )
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the compiled model and optional downstream
        components.
        """
        x = self.base_model(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.head is not None:
            x = self.head(x)

        return x
