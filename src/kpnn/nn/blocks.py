import pandas as pd
import torch
from torch import nn

from .masked_linear import MaskedLinear
from ..utils.errors import KPNNError


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
            node_name: idx
            for idx, node_name in enumerate(input_node_names)
        }
        output_index = {
            node_name: idx
            for idx, node_name in enumerate(output_node_names)
        }

        out_features = len(output_node_names)
        in_features = len(input_node_names)

        mask = torch.zeros(out_features, in_features, dtype=torch.float32)

        pseudo_targets_seen = set()
        pseudo_input_indices = []
        pseudo_output_indices = []

        for _, row in block_edges.iterrows():
            source = row["source"]
            target = row["target"]

            source_idx = input_index[source]
            target_idx = output_index[target]

            mask[target_idx, source_idx] = 1.0

            if target.startswith("pseudo__"):
                if target in pseudo_targets_seen:
                    raise KPNNError(
                        "Pseudo nodes must have exactly one incoming edge."
                    )

                pseudo_targets_seen.add(target)
                pseudo_input_indices.append(source_idx)
                pseudo_output_indices.append(target_idx)

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
        y = self.linear(x)

        if self.pseudo_output_indices.numel() > 0:
            y[:, self.pseudo_output_indices] = x[:, self.pseudo_input_indices]

        return y
