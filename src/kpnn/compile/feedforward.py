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

    model = KPNNModel(
        execution_plan=execution_plan,
        backend="feedforward",
    )

    artifact = KPNNArtifact(
        backend="feedforward",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer=execution_plan.node_names_by_layer,
        feature_names=execution_plan.input_node_names,
    )

    return model, artifact
