def test_public_api_exports_expected_names():
    import edge2torch

    assert hasattr(edge2torch, "compile_graph")
    assert hasattr(edge2torch, "align_features_to_input_nodes")
    assert hasattr(edge2torch, "customize_model")
    assert hasattr(edge2torch, "interpret_model")
    assert hasattr(edge2torch, "CompileArtifact")
    assert hasattr(edge2torch, "CompileBackend")
    assert hasattr(edge2torch, "COMPILE_BACKENDS")
    assert hasattr(edge2torch, "__version__")


def test_public_api_all_contains_expected_names():
    import edge2torch

    assert set(edge2torch.__all__) == {
        "align_features_to_input_nodes",
        "compile_graph",
        "customize_model",
        "interpret_model",
        "CompileArtifact",
        "CompileBackend",
        "COMPILE_BACKENDS",
        "__version__",
    }


def test_public_api_star_import_exports_expected_names():
    namespace = {}

    exec("from edge2torch import *", namespace)

    assert "compile_graph" in namespace
    assert "align_features_to_input_nodes" in namespace
    assert "customize_model" in namespace
    assert "interpret_model" in namespace
    assert "CompileArtifact" in namespace
    assert "CompileBackend" in namespace
    assert "COMPILE_BACKENDS" in namespace


def test_public_api_imports_are_callable_or_types():
    from edge2torch import (
        COMPILE_BACKENDS,
        CompileArtifact,
        CompileBackend,
        align_features_to_input_nodes,
        compile_graph,
        customize_model,
        interpret_model,
    )

    assert callable(compile_graph)
    assert callable(align_features_to_input_nodes)
    assert callable(customize_model)
    assert callable(interpret_model)
    assert isinstance(CompileArtifact, type)
    assert COMPILE_BACKENDS == frozenset({"feedforward", "state_update"})
    assert CompileBackend.__args__ == ("feedforward", "state_update")
