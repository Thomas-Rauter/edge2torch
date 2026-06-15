"""
GraphNN backend compilation.

Why this file exists
--------------------
This file isolates the logic for compiling a validated graph into the
graphnn backend. Keeping this path separate makes the graphnn-specific
compilation semantics explicit and allows the backend to evolve without
adding complexity to the general compiler dispatch.

Role in the package
-------------------
This is an internal backend-compilation module. It takes a graph,
derives the graphnn execution plan, builds the corresponding PyTorch
model, and packages the result together with its artifact. It should
contain graphnn-specific compilation logic, not public API handling or
generic backend dispatch.
"""

from ..graph.schema import EdgeGraph
from ..nn.model import EdgeGraphNNModel
from .artifact import CompileArtifact
from .execution_plan import build_graphnn_execution_plan
from .interpretation_metadata import (
    build_state_update_interpretation_sites,
    compute_hidden_nodes,
)


def compile_graphnn(
    graph: EdgeGraph,
    bias: bool = True,
    steps: int = 3,
) -> tuple[EdgeGraphNNModel, CompileArtifact]:
    """
    Compile a KPNN graph into a graph neural network PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.
    bias
        Whether compiled masked linear layers include bias terms. If True,
        each target node has a learned node-level offset in addition to its
        graph-defined weighted inputs. If False, node updates are computed only
        from graph-defined weighted inputs. Disabling bias gives the graph
        structure stricter control over node activations.
    steps
        Number of recurrent/message-passing update steps for the ``recurrent``
        and ``graphnn`` backends.

    Returns
    -------
    tuple
        A tuple of (model, artifact).
    """
    # Decides the structure
    execution_plan = build_graphnn_execution_plan(graph)

    # Builds the actual PyTorch modules from that structure.
    model = EdgeGraphNNModel(
        execution_plan=execution_plan,
        steps=steps,
        bias=bias,
    )

    # Stores the metadata needed later for alignment, interpretation, and
    # inspection.
    input_nodes = list(execution_plan.input_node_names)
    output_nodes = list(execution_plan.output_node_names)

    artifact = CompileArtifact(
        backend="graphnn",
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
