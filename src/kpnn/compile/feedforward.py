"""
Feedforward backend compilation.

Why this file exists
--------------------
This file isolates the logic for compiling a validated graph into the
feedforward backend. The separation keeps backend-specific compilation
out of the general compiler dispatch and makes the feedforward path easy
to understand and extend independently.

Role in the package
-------------------
This is an internal backend-compilation module. It takes a graph,
derives the feedforward execution plan, builds the corresponding PyTorch
model, and packages the result together with its artifact. It should
contain feedforward-specific compilation logic, not public API handling
or generic backend dispatch.
"""

from .artifact import KPNNArtifact
from .execution_plan import build_feedforward_execution_plan
from ..nn.model import KPNNModel


def compile_feedforward(graph):
    """
    Compile a KPNN graph into a feedforward PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.

    Returns
    -------
    tuple
        A tuple of (model, artifact).
    """
    execution_plan = build_feedforward_execution_plan(graph)

    model = KPNNModel(execution_plan=execution_plan)

    artifact = KPNNArtifact(
        backend="feedforward",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer=execution_plan.node_names_by_layer,
        feature_names=execution_plan.input_node_names,
    )

    return model, artifact
