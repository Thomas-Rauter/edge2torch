from torch import nn

from .customize.input_validation import validate_customize_model_inputs
from .nn.customized_model import CustomizedEdgeModel


def customize_model(
    model: nn.Module,
    activation: nn.Module | None = None,
    dropout: float | int | None = None,
    head: nn.Module | None = None,
) -> nn.Module:
    """
    Wrap a compiled KPNN model with optional downstream PyTorch modules.

    This function is a convenience layer for common post-compilation additions.
    It applies the requested components sequentially to the output of the
    compiled model. It does not modify the compiled graph structure, insert
    modules inside graph-derived layers, or replace ordinary PyTorch
    training/customization.

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
    Edge2TorchError
        If any input is invalid.

    Examples
    --------
    Add an activation function after the compiled model.

    >>> from torch import nn
    >>> from edge2torch.customize_model import customize_model
    >>>
    >>> customized_model = customize_model(
    ...     model=model,
    ...     activation=nn.ReLU(),
    ... )

    Add an activation, dropout, and task-specific prediction head.

    >>> customized_model = customize_model(
    ...     model=model,
    ...     activation=nn.ReLU(),
    ...     dropout=0.2,
    ...     head=nn.Linear(1, 1),
    ... )
    """
    validate_customize_model_inputs(
        model=model,
        activation=activation,
        dropout=dropout,
        head=head,
    )

    return CustomizedEdgeModel(
        base_model=model,
        activation=activation,
        dropout=dropout,
        head=head,
    )
