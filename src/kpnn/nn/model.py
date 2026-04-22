import pandas as pd
from torch import nn

from .blocks import FeedforwardLayerBlock
from ..utils.errors import KPNNError


class KPNNModel(nn.Module):
    """
    Feedforward KPNN model compiled from an execution plan.

    The model contains only the compiled structural connectivity and pseudo-node
    overwrite logic. No activation functions or other architectural choices are
    imposed here.
    """

    def __init__(
        self,
        execution_plan,
        backend: str = "feedforward",
        bias: bool = True,
    ):
        super().__init__()

        if backend != "feedforward":
            raise KPNNError(
                "KPNNModel currently only supports the 'feedforward' backend."
            )

        self.execution_plan = execution_plan
        self.backend = backend
        self.bias = bias

        self.layer_names = self._sort_layer_names(
            list(execution_plan.node_names_by_layer.keys())
        )

        self.blocks = nn.ModuleList()

        for layer_idx in range(len(self.layer_names) - 1):
            input_layer_name = self.layer_names[layer_idx]
            output_layer_name = self.layer_names[layer_idx + 1]

            input_node_names = execution_plan.node_names_by_layer[
                input_layer_name
            ]
            output_node_names = execution_plan.node_names_by_layer[
                output_layer_name
            ]

            block_edges = self._select_block_edges(
                expanded_edges=execution_plan.expanded_edges,
                input_node_names=input_node_names,
                output_node_names=output_node_names,
            )

            block = FeedforwardLayerBlock(
                input_node_names=input_node_names,
                output_node_names=output_node_names,
                block_edges=block_edges,
                bias=bias,
            )

            self.blocks.append(block)

    def forward(self, x):
        """
        Forward pass through the compiled feedforward KPNN.
        """
        for block in self.blocks:
            x = block(x)

        return x

    def get_layer_block(self, layer_name: str):
        """
        Return the feedforward block that produces the given layer.

        Parameters
        ----------
        layer_name : str
            Layer name like ``"layer_1"`` or ``"layer_2"``.

        Returns
        -------
        nn.Module
            Feedforward block whose output corresponds to ``layer_name``.

        Raises
        ------
        KPNNError
            If the layer name is invalid or refers to the input layer.
        """
        if not isinstance(layer_name, str):
            raise KPNNError(
                "'layer_name' must be a string."
            )

        if not layer_name.startswith("layer_"):
            raise KPNNError(
                f"Invalid layer name '{layer_name}'."
            )

        try:
            layer_idx = int(layer_name.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise KPNNError(
                f"Invalid layer name '{layer_name}'."
            ) from exc

        if layer_idx == 0:
            raise KPNNError(
                "The input layer 'layer_0' does not have a "
                "feedforward block."
            )

        if layer_name not in self.layer_names:
            raise KPNNError(
                f"Unknown layer name '{layer_name}'."
            )

        block_idx = layer_idx - 1

        if block_idx >= len(self.blocks):
            raise KPNNError(
                f"No block exists for layer '{layer_name}'."
            )

        return self.blocks[block_idx]

    @staticmethod
    def _sort_layer_names(layer_names: list[str]) -> list[str]:
        """
        Sort layer names like 'layer_0', 'layer_1', ...
        """
        return sorted(
            layer_names,
            key=lambda name: int(name.split("_")[1]),
        )

    @staticmethod
    def _select_block_edges(
        expanded_edges: pd.DataFrame,
        input_node_names: list[str],
        output_node_names: list[str],
    ) -> pd.DataFrame:
        """
        Select the edges connecting one layer to the next.
        """
        block_edges = expanded_edges[
            expanded_edges["source"].isin(input_node_names)
            & expanded_edges["target"].isin(output_node_names)
        ].copy()

        return block_edges.reset_index(drop=True)
