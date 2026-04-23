from .artifact import KPNNArtifact
from .execution_plan import build_recurrent_execution_plan
from ..nn.model import KPNNRecurrentModel


def compile_recurrent(graph):
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
    execution_plan = build_recurrent_execution_plan(graph)

    model = KPNNRecurrentModel(
        execution_plan=execution_plan,
        backend="recurrent",
    )

    artifact = KPNNArtifact(
        backend="recurrent",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer={},
        feature_names=execution_plan.input_node_names,
    )

    return model, artifact
