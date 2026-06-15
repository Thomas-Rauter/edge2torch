"""
Interpretation-site access helpers for compiled KPNN models.

Why this file exists
--------------------
This file centralizes interpretation-site identifier parsing and model lookup
logic shared by compiled models and interpretation modules. Keeping this logic
in one place makes the site-access contract consistent across backends.

Role in the package
-------------------
This is an internal neural-network helper module. It should define site-id
parsing and model-provider lookup utilities, not public API orchestration or
Captum execution logic.
"""

from torch import nn

from ..utils.errors import Edge2TorchError

LIST_INTERPRETATION_SITE_IDS_METHOD = "_edge2torch_list_interpretation_site_ids"
GET_INTERPRETATION_SITE_METHOD = "_edge2torch_get_interpretation_site"


def find_interpretation_site_provider(model: nn.Module) -> nn.Module:
    """
    Find a module that exposes edge2torch interpretation-site access.

    Raw compiled models expose these methods directly. Models returned by
    ``customize_model()`` and many manually wrapped PyTorch models expose them
    through a registered submodule.
    """
    if hasattr(model, GET_INTERPRETATION_SITE_METHOD):
        return model

    for module in model.modules():
        if module is model:
            continue

        if hasattr(module, GET_INTERPRETATION_SITE_METHOD):
            return module

    raise Edge2TorchError(
        "Node-level interpretation requires access to the compiled model's "
        "internal interpretation sites. This is supported for raw models "
        "returned by compile_graph(), models returned by customize_model(), "
        "and wrappers that keep the compiled model as a registered PyTorch "
        "submodule."
    )


def parse_feedforward_site_id(site_id: str) -> int:
    """
    Parse a feedforward interpretation site like ``layer_1``.
    """
    if not site_id.startswith("layer_"):
        raise Edge2TorchError(f"Invalid interpretation site '{site_id}'.")

    try:
        layer_idx = int(site_id.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise Edge2TorchError(
            f"Invalid interpretation site '{site_id}'."
        ) from exc

    if layer_idx == 0:
        raise Edge2TorchError(
            "The input layer 'layer_0' is not an interpretation site."
        )

    return layer_idx


def parse_state_update_site_id(site_id: str) -> int:
    """
    Parse a recurrent or graphnn interpretation site like ``step_1``.
    """
    if not site_id.startswith("step_"):
        raise Edge2TorchError(f"Invalid interpretation site '{site_id}'.")

    try:
        step_number = int(site_id.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise Edge2TorchError(
            f"Invalid interpretation site '{site_id}'."
        ) from exc

    if step_number <= 0:
        raise Edge2TorchError(f"Invalid interpretation site '{site_id}'.")

    return step_number - 1
