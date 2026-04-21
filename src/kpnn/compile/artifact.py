from dataclasses import dataclass


@dataclass
class KPNNArtifact:
    """
    Compilation artifact returned together with the compiled PyTorch model.

    Parameters
    ----------
    backend : str
        Backend used for compilation.
    graph
        Internal KPNN graph object used for compilation.
    execution_plan
        Compiled execution plan used to build the model.
    node_names_by_layer : dict[str, list[str]]
        Mapping from layer name to node names in that layer.
    feature_names : list[str]
        Names of the input features.
    """

    backend: str
    graph: object
    execution_plan: object
    node_names_by_layer: dict[str, list[str]]
    feature_names: list[str]
