"""
State-update step blocks for recurrent and graphnn KPNN models.

Why this file exists
--------------------
This file isolates one interpretable recurrent or graphnn update step. Each
step applies a shared masked linear transformation and re-injects input-node
values so that later interpretation can hook one unique module per step.

Role in the package
-------------------
This is an internal neural-network implementation module. It defines the
runtime building blocks used by compiled recurrent and graphnn models. It
should contain step-block behavior and shared masked-linear construction,
not public API logic, validation, or backend dispatch.
"""

from typing import cast

import pandas as pd
import torch
from torch import nn

from ..utils.errors import Edge2TorchError
from .masked_linear import (
    CONSTRAINT_UNCONSTRAINED,
    ConstrainedMaskedLinear,
    MaskedLinear,
    constraint_name_to_code,
)


class StateUpdateStep(nn.Module):
    """
    One recurrent or graphnn state-update step.

    The step applies a shared masked linear update to the full node-state
    vector and then re-injects the external input-node values. The masked
    linear module is referenced but not registered as a child so that model
    parameters remain owned by a single shared module.
    """

    _linear: ConstrainedMaskedLinear | MaskedLinear

    def __init__(
        self,
        linear: ConstrainedMaskedLinear | MaskedLinear,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_linear", linear)

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        input_indices: list[int],
    ) -> torch.Tensor:
        """
        Apply one masked state update and re-inject input-node values.
        """
        linear = cast(ConstrainedMaskedLinear | MaskedLinear, self._linear)

        state = linear(state)
        state[:, input_indices] = x
        return state


def build_node_state_linear(
    *,
    original_edges: pd.DataFrame,
    node_names: list[str],
    node_index: dict[str, int],
    bias: bool,
) -> ConstrainedMaskedLinear | MaskedLinear:
    """
    Build the shared masked linear layer for recurrent and graphnn models.
    """
    n_nodes = len(node_names)

    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)

    has_initial_weight = "initial_weight" in original_edges.columns
    has_constraint = "constraint" in original_edges.columns
    has_edge_metadata = has_initial_weight or has_constraint

    initial_weight = None
    constraint = None

    if has_initial_weight:
        initial_weight = torch.full(
            (n_nodes, n_nodes),
            fill_value=float("nan"),
            dtype=torch.float32,
        )

    if has_constraint:
        constraint = torch.full(
            (n_nodes, n_nodes),
            fill_value=CONSTRAINT_UNCONSTRAINED,
            dtype=torch.long,
        )

    for row in original_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        if source not in node_index:
            raise Edge2TorchError(
                f"Unknown source node '{source}' in graph edges."
            )

        if target not in node_index:
            raise Edge2TorchError(
                f"Unknown target node '{target}' in graph edges."
            )

        source_idx = node_index[source]
        target_idx = node_index[target]

        mask[target_idx, source_idx] = 1.0

        if has_initial_weight:
            assert initial_weight is not None
            row_initial_weight = getattr(row, "initial_weight")

            if pd.notna(row_initial_weight):
                initial_weight[target_idx, source_idx] = float(
                    row_initial_weight
                )

        if has_constraint:
            assert constraint is not None
            constraint[target_idx, source_idx] = constraint_name_to_code(
                str(row.constraint)
            )

    if has_edge_metadata:
        return ConstrainedMaskedLinear(
            in_features=n_nodes,
            out_features=n_nodes,
            mask=mask,
            initial_weight=initial_weight,
            constraint=constraint,
            bias=bias,
        )

    return MaskedLinear(
        in_features=n_nodes,
        out_features=n_nodes,
        mask=mask,
        bias=bias,
    )


def build_state_update_steps(
    *,
    linear: ConstrainedMaskedLinear | MaskedLinear,
    steps: int,
) -> nn.ModuleList:
    """
    Build one interpretable state-update step module per configured step.
    """
    return nn.ModuleList(StateUpdateStep(linear=linear) for _ in range(steps))
