from torch import nn

from .customize.input_validation import validate_customize_model_inputs
from .nn.customized_model import CustomizedKPNNModel


def customize_model(
    model: nn.Module,
    activation: nn.Module | None = None,
    dropout: float | int | None = None,
    head: nn.Module | None = None,
) -> nn.Module:
    """
    Add optional downstream architectural components to a compiled KPNN
    model.

    Parameters
    ----------
    model : nn.Module
        Compiled PyTorch model returned by ``compile_graph()``.
    activation : nn.Module | None, default=None
        Optional PyTorch activation module applied after the compiled model.
        This should be an instantiated module such as ``nn.ReLU()``.
    dropout : float | int | None, default=None
        Optional dropout probability applied after the activation. Must
        satisfy ``0 <= dropout < 1``.
    head : nn.Module | None, default=None
        Optional PyTorch module applied after dropout. This should be an
        instantiated module such as ``nn.Linear(...)``.

    Returns
    -------
    nn.Module
        Wrapped PyTorch model with the requested architectural components.

    Raises
    ------
    KPNNError
        If any input is invalid.
    """
    validate_customize_model_inputs(
        model=model,
        activation=activation,
        dropout=dropout,
        head=head,
    )

    return CustomizedKPNNModel(
        base_model=model,
        activation=activation,
        dropout=dropout,
        head=head,
    )
