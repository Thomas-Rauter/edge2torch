"""
State-update backend compilation.

Why this file exists
--------------------
This file isolates the logic for compiling a validated graph into the
state-update backend. The separation keeps topology-preserving
compilation semantics out of the general compiler dispatch.

Role in the package
-------------------
This is an internal backend-compilation module. It takes a graph,
derives the state-update execution plan, builds the corresponding PyTorch
model, and packages the result together with its artifact. It should
contain state-update compilation wiring, not public API handling or
generic backend dispatch.
"""

from ..graph.schema import EdgeGraph
from ..nn.model import StateUpdateEdgeModel
from .artifact import CompileArtifact
from .execution_plan import build_state_update_execution_plan
from .interpretation_metadata import (
    build_state_update_interpretation_sites,
    compute_hidden_nodes,
)


def compile_state_update(
    graph: EdgeGraph,
    bias: bool = True,
    steps: int = 3,
) -> tuple[StateUpdateEdgeModel, CompileArtifact]:
    """
    Compile a KPNN graph into a state-update PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.
    bias
        Whether compiled masked linear layers include bias terms. If True,
        each target node has a learned node-level offset in addition to its
        graph-defined weighted inputs. If False, node updates are computed
        only from graph-defined weighted inputs. Disabling bias gives the
        graph structure stricter control over node activations.
    steps
        Number of fixed state-update steps applied at runtime.

    Returns
    -------
    tuple
        A tuple of (model, artifact).
    """
    execution_plan = build_state_update_execution_plan(graph)

    model = StateUpdateEdgeModel(
        execution_plan=execution_plan,
        steps=steps,
        bias=bias,
    )

    input_nodes = list(execution_plan.input_node_names)
    output_nodes = list(execution_plan.output_node_names)

    artifact = CompileArtifact(
        backend="state_update",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer={},
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        hidden_nodes=compute_hidden_nodes(
            node_names=list(execution_plan.node_names),
            input_nodes=input_nodes,
            output_nodes=output_nodes,
        ),
        interpretation_sites=build_state_update_interpretation_sites(
            node_names=execution_plan.node_names,
            steps=steps,
        ),
        feature_names=input_nodes,
    )

    return model, artifact
