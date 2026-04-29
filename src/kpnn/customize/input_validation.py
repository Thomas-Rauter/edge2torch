"""
Validation logic for the customize_model() public API.

Why this file exists
--------------------
This file separates strict public input validation from both the
customize_model() API wrapper and the internal model-wrapper
implementation. Keeping validation here makes the accepted customization
contract easier to reason about and avoids duplicating checks in
multiple places.

Role in the package
-------------------
This is an internal validation module for model customization. It
defines what inputs are accepted by customize_model() and raises clear
errors for unsupported or ambiguous usage. It should not contain PyTorch
execution logic, model construction, or public API orchestration.
"""

from torch import nn

from ..utils.errors import KPNNError


def validate_customize_model_inputs(
    model,
    activation,
    dropout,
    head,
):
    """
    Validate the public inputs of ``customize_model()``.

    Parameters
    ----------
    model
        Compiled PyTorch model returned by ``compile_graph()``.
    activation
        Optional PyTorch activation module applied after the compiled model.
    dropout
        Optional dropout probability applied after the activation.
    head
        Optional PyTorch module applied after dropout.

    Raises
    ------
    KPNNError
        If any input is invalid.
    """
    if not isinstance(model, nn.Module):
        raise KPNNError("'model' must be a torch.nn.Module.")

    if not hasattr(model, "forward"):
        raise KPNNError("'model' must define a forward method.")

    if activation is not None and not isinstance(activation, nn.Module):
        raise KPNNError("'activation' must be a torch.nn.Module or None.")

    if activation is model:
        raise KPNNError("'activation' must not be the same object as 'model'.")

    if dropout is not None:
        if isinstance(dropout, bool):
            raise KPNNError("'dropout' must be a float, int, or None.")

        if not isinstance(dropout, (int, float)):
            raise KPNNError("'dropout' must be a float, int, or None.")

        dropout = float(dropout)

        if not 0 <= dropout < 1:
            raise KPNNError("'dropout' must satisfy 0 <= dropout < 1.")

    if head is not None and not isinstance(head, nn.Module):
        raise KPNNError("'head' must be a torch.nn.Module or None.")

    if head is model:
        raise KPNNError("'head' must not be the same object as 'model'.")

    if activation is not None and head is not None and activation is head:
        raise KPNNError("'activation' and 'head' must not be the same object.")
