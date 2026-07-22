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

from typing import Any

import torch
from torch import nn

from ..utils.errors import Edge2TorchError


def validate_customize_model_inputs(
    model: Any,
    activation: Any,
    dropout: Any,
    head: Any,
) -> None:
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
    Edge2TorchError
        If any input is invalid.
    """
    if not isinstance(model, nn.Module):
        raise Edge2TorchError("'model' must be a torch.nn.Module.")

    if not hasattr(model, "forward"):
        raise Edge2TorchError("'model' must define a forward method.")

    if activation is not None and not isinstance(activation, nn.Module):
        raise Edge2TorchError("'activation' must be a torch.nn.Module or None.")

    if activation is model:
        raise Edge2TorchError(
            "'activation' must not be the same object as 'model'."
        )

    if dropout is not None:
        if isinstance(dropout, bool):
            raise Edge2TorchError("'dropout' must be a float, int, or None.")

        if not isinstance(dropout, (int, float)):
            raise Edge2TorchError("'dropout' must be a float, int, or None.")

        dropout = float(dropout)

        if not 0 <= dropout < 1:
            raise Edge2TorchError("'dropout' must satisfy 0 <= dropout < 1.")

    if head is not None and not isinstance(head, nn.Module):
        raise Edge2TorchError("'head' must be a torch.nn.Module or None.")

    if head is model:
        raise Edge2TorchError("'head' must not be the same object as 'model'.")

    if activation is not None and head is not None and activation is head:
        raise Edge2TorchError(
            "'activation' and 'head' must not be the same object."
        )

    _validate_head_input_width(model=model, head=head)


def _validate_head_input_width(
    model: nn.Module,
    head: nn.Module | None,
) -> None:
    """
    Reject heads whose fixed input width does not match model output width.
    """
    if head is None:
        return

    head_in_features = getattr(head, "in_features", None)
    if not isinstance(head_in_features, int) or head_in_features <= 0:
        return

    expected_features = _infer_model_output_features(model)
    if expected_features is None:
        return

    if head_in_features == expected_features:
        return

    raise Edge2TorchError(
        f"'head' expects {head_in_features} input feature(s), but the "
        f"model output has {expected_features}. Use a head whose "
        f"in_features match the model output width."
    )


def _infer_model_output_features(model: nn.Module) -> int | None:
    """
    Infer the trailing feature width of ``model`` via a dry-run forward.
    """
    n_input_features = _infer_model_input_features(model)
    if n_input_features is None:
        return None

    try:
        with torch.no_grad():
            output = model(
                torch.zeros(1, n_input_features, dtype=torch.float32)
            )
    except Exception:
        return None

    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        return None

    return int(output.shape[-1])


def _infer_model_input_features(model: nn.Module) -> int | None:
    """
    Infer the input feature width of a compiled or linear-like model.
    """
    unwrapped = _unwrap_base_model(model)

    in_features = getattr(unwrapped, "in_features", None)
    if isinstance(in_features, int) and in_features > 0:
        return in_features

    blocks = getattr(unwrapped, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        linear = getattr(blocks[0], "linear", None)
        block_in = getattr(linear, "in_features", None)
        if isinstance(block_in, int) and block_in > 0:
            return block_in

    input_indices = getattr(unwrapped, "input_indices", None)
    if input_indices is not None:
        return len(input_indices)

    return None


def _unwrap_base_model(model: nn.Module) -> nn.Module:
    """
    Unwrap nested CustomizedEdgeModel wrappers to the compiled core.
    """
    current = model
    seen: set[int] = set()

    while True:
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)

        base = getattr(current, "base_model", None)
        if base is None or not isinstance(base, nn.Module):
            break
        current = base

    return current
