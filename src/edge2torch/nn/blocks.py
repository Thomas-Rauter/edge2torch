"""
Feedforward block definitions for compiled KPNN models.

Why this file exists
--------------------
This file isolates the PyTorch modules that implement feedforward
layer-to-layer execution for compiled graphs. The separation keeps
feedforward block behavior explicit and prevents backend-specific model
semantics from being mixed into higher-level model orchestration code.

Role in the package
-------------------
This is an internal neural-network implementation module. It defines the
runtime building blocks used by compiled feedforward models, including
the handling of compiler-generated pseudo nodes during execution. It
should contain feedforward block behavior, not public API logic,
validation, or backend dispatch.
"""

from typing import cast

import pandas as pd
import torch
from torch import nn

from ..utils.constants import PSEUDO_NODE_PREFIX
from ..utils.errors import Edge2TorchError
from .masked_linear import (
    CONSTRAINT_UNCONSTRAINED,
    ConstrainedMaskedLinear,
    MaskedLinear,
    constraint_name_to_code,
)


class FeedforwardLayerBlock(nn.Module):
    """
    One adjacent-layer feedforward block.

    All output nodes are computed through a masked linear layer. Pseudo node
    outputs are then overwritten with the copied activations from their unique
    input nodes.
    """

    def __init__(
        self,
        input_node_names: list[str],
        output_node_names: list[str],
        block_edges: pd.DataFrame,
        bias: bool = True,
    ):
        super().__init__()

        self.input_node_names = input_node_names
        self.output_node_names = output_node_names

        input_index = {
            node_name: idx for idx, node_name in enumerate(input_node_names)
        }
        output_index = {
            node_name: idx for idx, node_name in enumerate(output_node_names)
        }

        out_features = len(output_node_names)
        in_features = len(input_node_names)

        mask = torch.zeros(out_features, in_features, dtype=torch.float32)

        has_initial_weight = "initial_weight" in block_edges.columns
        has_constraint = "constraint" in block_edges.columns
        has_edge_metadata = has_initial_weight or has_constraint

        initial_weight = None
        constraint = None

        if has_initial_weight:
            initial_weight = torch.full(
                (out_features, in_features),
                fill_value=float("nan"),
                dtype=torch.float32,
            )

        if has_constraint:
            constraint = torch.full(
                (out_features, in_features),
                fill_value=CONSTRAINT_UNCONSTRAINED,
                dtype=torch.long,
            )

        pseudo_targets_seen = set()
        pseudo_input_indices = []
        pseudo_output_indices = []

        for _, row in block_edges.iterrows():
            source = row["source"]
            target = row["target"]

            source_idx = input_index[source]
            target_idx = output_index[target]

            mask[target_idx, source_idx] = 1.0

            if has_initial_weight:
                assert initial_weight is not None
                row_initial_weight = row["initial_weight"]

                if pd.notna(row_initial_weight):
                    initial_weight[target_idx, source_idx] = float(
                        row_initial_weight
                    )

            if has_constraint:
                assert constraint is not None
                constraint[target_idx, source_idx] = constraint_name_to_code(
                    str(row["constraint"])
                )

            if target.startswith(PSEUDO_NODE_PREFIX):
                if target in pseudo_targets_seen:
                    raise Edge2TorchError(
                        "Pseudo nodes must have exactly one incoming edge."
                    )

                pseudo_targets_seen.add(target)
                pseudo_input_indices.append(source_idx)
                pseudo_output_indices.append(target_idx)

        self.linear: ConstrainedMaskedLinear | MaskedLinear

        if has_edge_metadata:
            self.linear = ConstrainedMaskedLinear(
                in_features=in_features,
                out_features=out_features,
                mask=mask,
                initial_weight=initial_weight,
                constraint=constraint,
                bias=bias,
            )
        else:
            self.linear = MaskedLinear(
                in_features=in_features,
                out_features=out_features,
                mask=mask,
                bias=bias,
            )

        self.register_buffer(
            "pseudo_input_indices",
            torch.tensor(pseudo_input_indices, dtype=torch.long),
        )
        self.register_buffer(
            "pseudo_output_indices",
            torch.tensor(pseudo_output_indices, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute one layer transition and overwrite pseudo node outputs.
        """
        pseudo_input_indices = cast(torch.Tensor, self.pseudo_input_indices)
        pseudo_output_indices = cast(torch.Tensor, self.pseudo_output_indices)

        y = self.linear(x)

        if pseudo_output_indices.numel() > 0:
            y[:, pseudo_output_indices] = x[:, pseudo_input_indices]

        return y
