import pandas as pd
import torch
from torch import nn

from .blocks import FeedforwardLayerBlock
from .masked_linear import MaskedLinear
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


class KPNNRecurrentModel(nn.Module):
    """
    Recurrent KPNN model compiled from a recurrent execution plan.

    The model applies a masked recurrent update over all graph nodes for a
    fixed number of steps. No activation functions or other architectural
    choices are imposed here.

    Input features are injected into nodes with zero in-degree. The model
    returns the activations of nodes with zero out-degree.
    """

    def __init__(
        self,
        execution_plan,
        backend: str = "recurrent",
        steps: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        if backend != "recurrent":
            raise KPNNError(
                "KPNNRecurrentModel currently only supports the "
                "'recurrent' backend."
            )

        if not isinstance(steps, int) or steps <= 0:
            raise KPNNError(
                "'steps' must be a positive integer."
            )

        self.execution_plan = execution_plan
        self.backend = backend
        self.steps = steps
        self.bias = bias

        self.node_names = list(execution_plan.node_names)
        self.input_node_names = list(execution_plan.input_node_names)
        self.output_node_names = list(execution_plan.output_node_names)

        self.node_index = {
            node_name: idx
            for idx, node_name in enumerate(self.node_names)
        }

        self.input_indices = [
            self.node_index[node_name]
            for node_name in self.input_node_names
        ]
        self.output_indices = [
            self.node_index[node_name]
            for node_name in self.output_node_names
        ]

        n_nodes = len(self.node_names)

        mask = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)

        for _, row in execution_plan.original_edges.iterrows():
            source = row["source"]
            target = row["target"]

            source_idx = self.node_index[source]
            target_idx = self.node_index[target]

            mask[target_idx, source_idx] = 1.0

        self.recurrent = MaskedLinear(
            in_features=n_nodes,
            out_features=n_nodes,
            mask=mask,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the recurrent KPNN for a fixed number of update steps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (n_examples, n_input_nodes).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (n_examples, n_output_nodes).
        """
        if x.ndim != 2:
            raise KPNNError(
                "Input tensor must be 2-dimensional."
            )

        expected_n_features = len(self.input_indices)

        if x.shape[1] != expected_n_features:
            raise KPNNError(
                "Input tensor has the wrong number of features. "
                f"Expected {expected_n_features}, got {x.shape[1]}."
            )

        batch_size = x.shape[0]
        n_nodes = len(self.node_names)

        state = torch.zeros(
            batch_size,
            n_nodes,
            dtype=x.dtype,
            device=x.device,
        )

        state[:, self.input_indices] = x

        for _ in range(self.steps):
            state = self.recurrent(state)
            state[:, self.input_indices] = x

        return state[:, self.output_indices]


class KPNNGraphNNModel(nn.Module):
    """
    Graph neural network KPNN model compiled from a graphnn execution plan.

    The model applies a masked message-passing style update over all graph
    nodes for a fixed number of steps. No activation functions or other
    architectural choices are imposed here.

    Input features are injected into nodes with zero in-degree. The model
    returns the activations of nodes with zero out-degree.
    """

    def __init__(
        self,
        execution_plan,
        backend: str = "graphnn",
        steps: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        if backend != "graphnn":
            raise KPNNError(
                "KPNNGraphNNModel currently only supports the "
                "'graphnn' backend."
            )

        if not isinstance(steps, int) or steps <= 0:
            raise KPNNError(
                "'steps' must be a positive integer."
            )

        self.execution_plan = execution_plan
        self.backend = backend
        self.steps = steps
        self.bias = bias

        self.node_names = list(execution_plan.node_names)
        self.input_node_names = list(execution_plan.input_node_names)
        self.output_node_names = list(execution_plan.output_node_names)

        self.node_index = {
            node_name: idx
            for idx, node_name in enumerate(self.node_names)
        }

        self.input_indices = [
            self.node_index[node_name]
            for node_name in self.input_node_names
        ]
        self.output_indices = [
            self.node_index[node_name]
            for node_name in self.output_node_names
        ]

        n_nodes = len(self.node_names)

        mask = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)

        for _, row in execution_plan.original_edges.iterrows():
            source = row["source"]
            target = row["target"]

            source_idx = self.node_index[source]
            target_idx = self.node_index[target]

            mask[target_idx, source_idx] = 1.0

        self.message_passing = MaskedLinear(
            in_features=n_nodes,
            out_features=n_nodes,
            mask=mask,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the graphnn KPNN for a fixed number of update steps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (n_examples, n_input_nodes).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (n_examples, n_output_nodes).
        """
        if x.ndim != 2:
            raise KPNNError(
                "Input tensor must be 2-dimensional."
            )

        expected_n_features = len(self.input_indices)

        if x.shape[1] != expected_n_features:
            raise KPNNError(
                "Input tensor has the wrong number of features. "
                f"Expected {expected_n_features}, got {x.shape[1]}."
            )

        batch_size = x.shape[0]
        n_nodes = len(self.node_names)

        state = torch.zeros(
            batch_size,
            n_nodes,
            dtype=x.dtype,
            device=x.device,
        )

        state[:, self.input_indices] = x

        for _ in range(self.steps):
            state = self.message_passing(state)
            state[:, self.input_indices] = x

        return state[:, self.output_indices]
