"""
Recurrent backend compilation.

Why this file exists
--------------------
This file isolates the logic for compiling a validated graph into the
recurrent backend. The separation keeps recurrent-specific compilation
semantics out of the general compiler dispatch and makes the recurrent
path easier to evolve independently.

Role in the package
-------------------
This is an internal backend-compilation module. It takes a graph,
derives the recurrent execution plan, builds the corresponding PyTorch
model, and packages the result together with its artifact. It should
contain recurrent-specific compilation logic, not public API handling or
generic backend dispatch.
"""

from ..graph.schema import EdgeGraph
from ..nn.model import RecurrentEdgeModel
from .artifact import CompileArtifact
from .execution_plan import build_recurrent_execution_plan


def compile_recurrent(
    graph: EdgeGraph,
) -> tuple[RecurrentEdgeModel, CompileArtifact]:
    """
    Compile a KPNN graph into a recurrent PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.

    Returns
    -------
    tuple
        A tuple of (model, artifact).
    """
    # Decides the structure
    execution_plan = build_recurrent_execution_plan(graph)

    # Builds the actual PyTorch modules from that structure.
    model = RecurrentEdgeModel(execution_plan=execution_plan)

    # Stores the metadata needed later for alignment, interpretation, and
    # inspection.
    artifact = CompileArtifact(
        backend="recurrent",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer={},
        feature_names=execution_plan.input_node_names,
    )

    return model, artifact
