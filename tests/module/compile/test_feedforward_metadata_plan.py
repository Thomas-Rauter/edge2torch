import math

import pandas as pd
import pytest

from edge2torch.compile.execution_plan import (
    build_feedforward_execution_plan,
)
from edge2torch.graph.io import edgelist_to_graph


def _build_plan(edgelist: pd.DataFrame):
    graph = edgelist_to_graph(edgelist)
    return build_feedforward_execution_plan(graph)


def test_direct_edges_preserve_metadata():
    edgelist = pd.DataFrame(
        {
            "source": ["feature_a", "hidden"],
            "target": ["hidden", "prediction"],
            "initial_weight": [0.25, -0.50],
            "constraint": ["positive", "negative"],
        }
    )

    plan = _build_plan(edgelist)

    assert list(plan.expanded_edges.columns) == [
        "source",
        "target",
        "initial_weight",
        "constraint",
    ]

    first_edge = plan.expanded_edges.iloc[0]
    second_edge = plan.expanded_edges.iloc[1]

    assert first_edge["source"] == "feature_a"
    assert first_edge["target"] == "hidden"
    assert first_edge["initial_weight"] == pytest.approx(0.25)
    assert first_edge["constraint"] == "positive"

    assert second_edge["source"] == "hidden"
    assert second_edge["target"] == "prediction"
    assert second_edge["initial_weight"] == pytest.approx(-0.50)
    assert second_edge["constraint"] == "negative"


def test_skip_internal_edges_get_default_metadata():
    edgelist = pd.DataFrame(
        {
            "source": [
                "feature_a",
                "feature_a",
                "hidden",
                "middle",
            ],
            "target": [
                "hidden",
                "prediction",
                "middle",
                "prediction",
            ],
            "initial_weight": [
                0.20,
                0.75,
                0.30,
                -0.40,
            ],
            "constraint": [
                "positive",
                "fixed",
                "positive",
                "negative",
            ],
        }
    )

    plan = _build_plan(edgelist)

    first_pseudo = "__edge2torch_pseudo__feature_a__prediction__layer_1"
    second_pseudo = "__edge2torch_pseudo__feature_a__prediction__layer_2"

    first_internal_edge = plan.expanded_edges[
        (plan.expanded_edges["source"] == "feature_a")
        & (plan.expanded_edges["target"] == first_pseudo)
    ].iloc[0]

    second_internal_edge = plan.expanded_edges[
        (plan.expanded_edges["source"] == first_pseudo)
        & (plan.expanded_edges["target"] == second_pseudo)
    ].iloc[0]

    assert pd.isna(first_internal_edge["initial_weight"])
    assert first_internal_edge["constraint"] == "unconstrained"

    assert pd.isna(second_internal_edge["initial_weight"])
    assert second_internal_edge["constraint"] == "unconstrained"


def test_skip_final_edge_gets_original_metadata():
    edgelist = pd.DataFrame(
        {
            "source": [
                "feature_a",
                "feature_a",
                "hidden",
                "middle",
            ],
            "target": [
                "hidden",
                "prediction",
                "middle",
                "prediction",
            ],
            "initial_weight": [
                0.20,
                0.75,
                0.30,
                -0.40,
            ],
            "constraint": [
                "positive",
                "fixed",
                "positive",
                "negative",
            ],
        }
    )

    plan = _build_plan(edgelist)

    second_pseudo = "__edge2torch_pseudo__feature_a__prediction__layer_2"

    final_skip_edge = plan.expanded_edges[
        (plan.expanded_edges["source"] == second_pseudo)
        & (plan.expanded_edges["target"] == "prediction")
    ].iloc[0]

    assert final_skip_edge["initial_weight"] == pytest.approx(0.75)
    assert final_skip_edge["constraint"] == "fixed"


def test_sparse_metadata_is_preserved_through_expansion():
    edgelist = pd.DataFrame(
        {
            "source": [
                "feature_a",
                "feature_a",
                "hidden",
                "middle",
            ],
            "target": [
                "hidden",
                "prediction",
                "middle",
                "prediction",
            ],
            "initial_weight": [
                math.nan,
                0.75,
                math.nan,
                -0.40,
            ],
            "constraint": [
                None,
                "fixed",
                "positive",
                None,
            ],
        }
    )

    plan = _build_plan(edgelist)

    direct_edge = plan.expanded_edges[
        (plan.expanded_edges["source"] == "hidden")
        & (plan.expanded_edges["target"] == "middle")
    ].iloc[0]

    assert pd.isna(direct_edge["initial_weight"])
    assert direct_edge["constraint"] == "positive"

    second_pseudo = "__edge2torch_pseudo__feature_a__prediction__layer_2"

    final_skip_edge = plan.expanded_edges[
        (plan.expanded_edges["source"] == second_pseudo)
        & (plan.expanded_edges["target"] == "prediction")
    ].iloc[0]

    assert final_skip_edge["initial_weight"] == pytest.approx(0.75)
    assert final_skip_edge["constraint"] == "fixed"

    unconstrained_edge = plan.expanded_edges[
        (plan.expanded_edges["source"] == "middle")
        & (plan.expanded_edges["target"] == "prediction")
    ].iloc[0]

    assert unconstrained_edge["initial_weight"] == pytest.approx(-0.40)
    assert unconstrained_edge["constraint"] == "unconstrained"
