from .artifact import KPNNArtifact
from .execution_plan import build_graphnn_execution_plan
from ..nn.model import KPNNGraphNNModel


def compile_graphnn(graph):
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

    model = KPNNGraphNNModel(
        execution_plan=execution_plan,
        backend="graphnn",
    )

    artifact = KPNNArtifact(
        backend="graphnn",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer={},
        feature_names=execution_plan.input_node_names,
    )

    return model, artifact
