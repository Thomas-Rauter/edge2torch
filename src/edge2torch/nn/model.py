"""
Compiled PyTorch model classes for KPNN backends.

Why this file exists
--------------------
This file collects the concrete PyTorch model classes produced by the
different backend compilers. Keeping these model implementations in one
place makes the runtime behavior of compiled backends explicit and
separates model execution semantics from execution-plan construction and
public API orchestration.

Role in the package
-------------------
This is an internal neural-network implementation module. It defines the
compiled model classes for the feedforward backend and the shared
state-update core used by the recurrent and graphnn backends. It should
contain model execution logic and backend-specific model structure, not
public API validation, graph conversion, or compiler dispatch.
"""

from typing import cast

import pandas as pd
import torch
from torch import nn

from ..compile.execution_plan import (
    FeedforwardExecutionPlan,
    StateUpdateExecutionPlan,
)
from ..utils.errors import Edge2TorchError
from .blocks import FeedforwardLayerBlock
from .interpretation_sites import (
    parse_feedforward_site_id,
    parse_state_update_site_id,
)
from .masked_linear import ConstrainedMaskedLinear, MaskedLinear
from .step_block import (
    StateUpdateStep,
    build_node_state_linear,
    build_state_update_steps,
)


class EdgeModel(nn.Module):
    """
    Feedforward KPNN model compiled from an execution plan.

    The model contains only the compiled structural connectivity and pseudo-node
    overwrite logic. No activation functions or other architectural choices are
    imposed here.
    """

    def __init__(
        self,
        execution_plan: FeedforwardExecutionPlan,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.execution_plan = execution_plan
        self.backend = "feedforward"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the compiled feedforward KPNN.
        """
        for block in self.blocks:
            x = block(x)

        return x

    def _edge2torch_list_interpretation_site_ids(self) -> list[str]:
        """
        Return feedforward interpretation site identifiers.
        """
        return [
            layer_name
            for layer_name in self.layer_names
            if layer_name != "layer_0"
        ]

    def _edge2torch_get_interpretation_site(
        self,
        site_id: str,
    ) -> FeedforwardLayerBlock:
        """
        Return the feedforward block for an interpretation site.

        Parameters
        ----------
        site_id : str
            Site identifier like ``"layer_1"`` or ``"layer_2"``.

        Returns
        -------
        FeedforwardLayerBlock
            Feedforward block whose output corresponds to ``site_id``.

        Raises
        ------
        Edge2TorchError
            If the site identifier is invalid or has no corresponding block.
        """
        layer_idx = parse_feedforward_site_id(site_id)
        layer_name = f"layer_{layer_idx}"

        if layer_name not in self.layer_names:
            raise Edge2TorchError(f"Unknown interpretation site '{site_id}'.")

        block_idx = layer_idx - 1

        if block_idx >= len(self.blocks):
            raise Edge2TorchError(
                f"No block exists for interpretation site '{site_id}'."
            )

        return cast(FeedforwardLayerBlock, self.blocks[block_idx])

    @staticmethod
    def _sort_layer_names(layer_names: list[str]) -> list[str]:
        """
        Sort layer names like 'layer_0', 'layer_1', ...
        """
        try:
            return sorted(
                layer_names,
                key=lambda name: int(name.split("_")[1]),
            )
        except (IndexError, ValueError) as exc:
            raise Edge2TorchError(
                "Invalid layer name in execution plan."
            ) from exc

    @staticmethod
    def _select_block_edges(
        expanded_edges: pd.DataFrame,
        input_node_names: list[str],
        output_node_names: list[str],
    ) -> pd.DataFrame:
        """
        Select the edges connecting one layer to the next.
        """
        block_edges = cast(
            pd.DataFrame,
            expanded_edges[
                expanded_edges["source"].isin(input_node_names)
                & expanded_edges["target"].isin(output_node_names)
            ].copy(),
        )

        return block_edges.reset_index(drop=True)


class StateUpdateEdgeModel(nn.Module):
    """
    Topology-preserving KPNN model with fixed-step state updates.

    The model applies a shared masked linear update over all graph nodes
    for a fixed number of steps. No activation functions or other
    architectural choices are imposed here.

    Input features are injected into nodes with zero in-degree. The model
    returns the activations of nodes with zero out-degree.

    The recurrent and graphnn backends currently share this
    implementation and differ only by their ``backend`` label.
    """

    def __init__(
        self,
        execution_plan: StateUpdateExecutionPlan,
        steps: int = 3,
        bias: bool = True,
        backend: str = "state_update",
        linear_module_name: str = "state_linear",
    ) -> None:
        super().__init__()

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise Edge2TorchError("'steps' must be an integer.")

        if steps <= 0:
            raise Edge2TorchError("'steps' must be a positive integer.")

        self.execution_plan = execution_plan
        self.backend = backend
        self.steps = steps
        self.bias = bias
        self._linear_module_name = linear_module_name

        self.node_names = list(execution_plan.node_names)
        self.input_node_names = list(execution_plan.input_node_names)
        self.output_node_names = list(execution_plan.output_node_names)

        if not self.input_node_names:
            raise Edge2TorchError(
                "StateUpdateEdgeModel requires at least one input node."
            )

        if not self.output_node_names:
            raise Edge2TorchError(
                "StateUpdateEdgeModel requires at least one output node."
            )

        self.node_index = {
            node_name: idx for idx, node_name in enumerate(self.node_names)
        }

        self.input_indices = [
            self.node_index[node_name] for node_name in self.input_node_names
        ]
        self.output_indices = [
            self.node_index[node_name] for node_name in self.output_node_names
        ]

        linear = build_node_state_linear(
            original_edges=execution_plan.original_edges,
            node_names=self.node_names,
            node_index=self.node_index,
            bias=bias,
        )
        self.add_module(linear_module_name, linear)
        self.update_steps = build_state_update_steps(
            linear=self.state_linear,
            steps=steps,
        )

    @property
    def state_linear(self) -> MaskedLinear | ConstrainedMaskedLinear:
        """Canonical access to the masked state-update linear."""
        module = self._modules[self._linear_module_name]
        return cast(
            MaskedLinear | ConstrainedMaskedLinear,
            module,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run fixed-step state updates over the compiled graph.

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
            raise Edge2TorchError("Input tensor must be 2-dimensional.")

        expected_n_features = len(self.input_indices)

        if x.shape[1] != expected_n_features:
            raise Edge2TorchError(
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

        for step in self.update_steps:
            state = step(state, x, self.input_indices)

        return state[:, self.output_indices]

    def _edge2torch_list_interpretation_site_ids(self) -> list[str]:
        """
        Return state-update interpretation site identifiers.
        """
        return [f"step_{step_idx}" for step_idx in range(1, self.steps + 1)]

    def _edge2torch_get_interpretation_site(
        self,
        site_id: str,
    ) -> StateUpdateStep:
        """
        Return the state-update step module for an interpretation site.
        """
        step_idx = parse_state_update_site_id(site_id)

        if step_idx >= len(self.update_steps):
            raise Edge2TorchError(f"Unknown interpretation site '{site_id}'.")

        return cast(StateUpdateStep, self.update_steps[step_idx])


class RecurrentEdgeModel(StateUpdateEdgeModel):
    """
    Recurrent KPNN model compiled from a state-update execution plan.

    Thin backend-labeled wrapper around ``StateUpdateEdgeModel``.
    """

    def __init__(
        self,
        execution_plan: StateUpdateExecutionPlan,
        steps: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__(
            execution_plan=execution_plan,
            steps=steps,
            bias=bias,
            backend="recurrent",
            linear_module_name="recurrent",
        )


class EdgeGraphNNModel(StateUpdateEdgeModel):
    """
    Graphnn KPNN model compiled from a state-update execution plan.

    Thin backend-labeled wrapper around ``StateUpdateEdgeModel``.
    """

    def __init__(
        self,
        execution_plan: StateUpdateExecutionPlan,
        steps: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__(
            execution_plan=execution_plan,
            steps=steps,
            bias=bias,
            backend="graphnn",
            linear_module_name="message_passing",
        )
