from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from edge2torch.compile_graph import compile_graph
from edge2torch.customize_model import customize_model
from edge2torch.interpret_model import interpret_model
from edge2torch.utils.constants import PSEUDO_NODE_PREFIX

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "edgelists"


def _load_edgelist(filename: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / filename)


def _make_sample_data(
    artifact,
    *,
    n_samples: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            feature_name: rng.uniform(0.1, 1.0, n_samples)
            for feature_name in artifact.feature_names
        },
        index=[f"sample_{idx}" for idx in range(1, n_samples + 1)],
    )


def _site_sort_key(site_id: str) -> int:
    return int(site_id.split("_")[1])


def _expected_feedforward_summary(
    site_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    site_ids = sorted(site_results.keys(), key=_site_sort_key)
    summary = site_results[site_ids[0]]

    for site_id in site_ids[1:]:
        summary = pd.concat([summary, site_results[site_id]], axis=1)

    return summary


def _expected_max_abs_summary(
    site_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    site_ids = sorted(site_results.keys(), key=_site_sort_key)
    arrays = np.stack(
        [site_results[site_id].to_numpy() for site_id in site_ids],
        axis=0,
    )

    abs_arrays = np.abs(arrays)
    max_indices = abs_arrays.argmax(axis=0)
    n_samples, n_nodes = arrays.shape[1], arrays.shape[2]

    row_indices = np.arange(n_samples)[:, None]
    col_indices = np.arange(n_nodes)[None, :]
    summary_values = arrays[max_indices, row_indices, col_indices]

    return pd.DataFrame(
        summary_values,
        index=site_results[site_ids[0]].index,
        columns=site_results[site_ids[0]].columns,
    )


def _assert_summary_matches_sites(
    *,
    summary: pd.DataFrame,
    sites: dict[str, pd.DataFrame],
    backend: str,
    site_aggregation: str = "max_abs",
) -> None:
    if backend == "feedforward":
        expected = _expected_feedforward_summary(sites)
    else:
        assert site_aggregation == "max_abs"
        expected = _expected_max_abs_summary(sites)

    pd.testing.assert_frame_equal(summary, expected)


def _assert_no_pseudo_node_columns(
    result: pd.DataFrame | dict[str, pd.DataFrame],
) -> None:
    if isinstance(result, pd.DataFrame):
        tables = [result]
    else:
        tables = list(result.values())

    for table in tables:
        assert not any(
            column_name.startswith(PSEUDO_NODE_PREFIX)
            for column_name in table.columns
        )


def test_interpret_model_recurrent_cycle_fixture_returns_hidden_node_summary():
    edgelist = _load_edgelist("recurrent_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="recurrent",
        steps=2,
        quiet=True,
    )
    data = _make_sample_data(artifact)

    summary = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
    )

    assert isinstance(summary, pd.DataFrame)
    assert summary.shape == (len(data), len(artifact.hidden_nodes))
    assert list(summary.columns) == artifact.hidden_nodes
    assert list(summary.index) == list(data.index)


def test_interpret_model_graphnn_cycle_fixture_returns_hidden_node_summary():
    edgelist = _load_edgelist("graphnn_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        steps=2,
        quiet=True,
    )
    data = _make_sample_data(artifact)

    summary = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
    )

    assert isinstance(summary, pd.DataFrame)
    assert summary.shape == (len(data), len(artifact.hidden_nodes))
    assert list(summary.columns) == artifact.hidden_nodes
    assert list(summary.index) == list(data.index)


def test_interpret_model_recurrent_cycle_fixture_returns_step_site_tables():
    edgelist = _load_edgelist("recurrent_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="recurrent",
        steps=2,
        quiet=True,
    )
    data = _make_sample_data(artifact)

    sites = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
        level="sites",
        nodes="hidden",
    )

    assert isinstance(sites, dict)
    assert list(sites.keys()) == ["step_1", "step_2"]
    assert list(sites["step_1"].columns) == artifact.hidden_nodes
    assert sites["step_1"].shape == (len(data), len(artifact.hidden_nodes))


def test_interpret_model_graphnn_cycle_fixture_returns_step_site_tables():
    edgelist = _load_edgelist("graphnn_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        steps=2,
        quiet=True,
    )
    data = _make_sample_data(artifact)

    sites = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
        level="sites",
        nodes="hidden",
    )

    assert isinstance(sites, dict)
    assert list(sites.keys()) == ["step_1", "step_2"]
    assert list(sites["step_1"].columns) == artifact.hidden_nodes
    assert sites["step_1"].shape == (len(data), len(artifact.hidden_nodes))


def test_interpret_model_recurrent_cycle_summary_matches_max_abs_sites():
    edgelist = _load_edgelist("recurrent_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="recurrent",
        steps=2,
        quiet=True,
    )
    data = _make_sample_data(artifact, seed=1)

    common_kwargs = {
        "model": model,
        "artifact": artifact,
        "data": data,
        "target": "nodes",
        "method": "LayerActivation",
        "quiet": True,
        "nodes": "hidden",
    }

    summary = interpret_model(
        **common_kwargs,
        level="summary",
        site_aggregation="max_abs",
    )
    sites = interpret_model(
        **common_kwargs,
        level="sites",
    )

    _assert_summary_matches_sites(
        summary=summary,
        sites=sites,
        backend="recurrent",
        site_aggregation="max_abs",
    )


def test_interpret_model_graphnn_cycle_summary_matches_max_abs_sites():
    edgelist = _load_edgelist("graphnn_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        steps=2,
        quiet=True,
    )
    data = _make_sample_data(artifact, seed=2)

    common_kwargs = {
        "model": model,
        "artifact": artifact,
        "data": data,
        "target": "nodes",
        "method": "LayerActivation",
        "quiet": True,
        "nodes": "hidden",
    }

    summary = interpret_model(
        **common_kwargs,
        level="summary",
        site_aggregation="max_abs",
    )
    sites = interpret_model(
        **common_kwargs,
        level="sites",
    )

    _assert_summary_matches_sites(
        summary=summary,
        sites=sites,
        backend="graphnn",
        site_aggregation="max_abs",
    )


def test_interpret_model_skip_edge_fixture_summary_matches_sites():
    edgelist = _load_edgelist("feedforward_skip_edges.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )
    data = _make_sample_data(artifact, seed=3)

    common_kwargs = {
        "model": model,
        "artifact": artifact,
        "data": data,
        "target": "nodes",
        "method": "LayerActivation",
        "quiet": True,
        "nodes": "hidden",
    }

    summary = interpret_model(
        **common_kwargs,
        level="summary",
    )
    sites = interpret_model(
        **common_kwargs,
        level="sites",
    )

    _assert_summary_matches_sites(
        summary=summary,
        sites=sites,
        backend="feedforward",
    )
    assert set(summary.columns) == set(artifact.hidden_nodes)


def test_interpret_model_hides_pseudo_nodes_on_skip_edge_fixture():
    edgelist = _load_edgelist("feedforward_skip_edges.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )
    data = _make_sample_data(artifact, seed=4)

    assert artifact.execution_plan.pseudo_nodes

    summary = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
        nodes="hidden",
    )
    sites = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
        level="sites",
        nodes="non_input",
    )

    _assert_no_pseudo_node_columns(summary)
    _assert_no_pseudo_node_columns(sites)
    assert set(summary.columns) == set(artifact.hidden_nodes)


@pytest.mark.parametrize(
    ("fixture_name", "backend"),
    [
        ("feedforward_skip_edges.csv", "feedforward"),
        ("recurrent_cycle.csv", "recurrent"),
        ("graphnn_cycle.csv", "graphnn"),
    ],
)
def test_interpret_model_supports_nodes_after_customize_model(
    fixture_name,
    backend,
):
    edgelist = _load_edgelist(fixture_name)
    compile_kwargs = {"quiet": True}

    if backend in {"recurrent", "graphnn"}:
        compile_kwargs["steps"] = 2

    base_model, artifact = compile_graph(
        edgelist=edgelist,
        backend=backend,
        **compile_kwargs,
    )
    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        dropout=0.1,
    )
    data = _make_sample_data(artifact, seed=5)

    summary = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerActivation",
        quiet=True,
    )

    assert isinstance(summary, pd.DataFrame)
    assert summary.shape[0] == len(data)
    assert summary.shape[1] == len(artifact.hidden_nodes)
    assert set(summary.columns) == set(artifact.hidden_nodes)

    inputs = torch.tensor(data.to_numpy(), dtype=torch.float32)
    base_output = base_model(inputs)
    customized_output = model(inputs)

    assert base_output.shape == customized_output.shape
