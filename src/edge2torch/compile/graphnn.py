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


def compile_graphnn(
    graph: EdgeGraph,
) -> tuple[EdgeGraphNNModel, CompileArtifact]:
    """
    Compile a KPNN graph into a graph neural network PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.

    Returns
    -------
    tuple
        A tuple of (model, artifact).
    """
    execution_plan = build_graphnn_execution_plan(graph)

    model = EdgeGraphNNModel(execution_plan=execution_plan)

    artifact = CompileArtifact(
        backend="graphnn",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer={},
        feature_names=execution_plan.input_node_names,
    )

    return model, artifact
