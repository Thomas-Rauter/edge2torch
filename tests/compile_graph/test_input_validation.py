import pandas as pd
import pytest

from kpnn.compile.input_validation import validate_compile_graph_inputs
from kpnn.utils.errors import KPNNError


def test_validate_compile_graph_inputs_accepts_valid_inputs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    validate_compile_graph_inputs(
        edgelist=edgelist,
        backend="feedforward",
        quiet=False,
    )


def test_validate_compile_graph_inputs_raises_for_non_dataframe():
    edgelist = [
        {"source": "gene_1", "target": "pathway_1"},
    ]

    with pytest.raises(KPNNError, match="pandas DataFrame"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_missing_source_column():
    edgelist = pd.DataFrame(
        {
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(KPNNError, match="Missing: source"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_missing_target_column():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
        }
    )

    with pytest.raises(KPNNError, match="Missing: target"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_duplicate_source_column():
    edgelist = pd.DataFrame(
        [
            ["gene_1", "gene_1", "pathway_1"],
        ],
        columns=["source", "source", "target"],
    )

    with pytest.raises(KPNNError, match="exactly once"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_duplicate_target_column():
    edgelist = pd.DataFrame(
        [
            ["gene_1", "pathway_1", "pathway_1"],
        ],
        columns=["source", "target", "target"],
    )

    with pytest.raises(KPNNError, match="exactly once"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_missing_values():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", None],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(KPNNError, match="must not contain missing values"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_empty_source_name():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", ""],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(KPNNError, match="empty node names"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_whitespace_target_name():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "   "],
        }
    )

    with pytest.raises(KPNNError, match="empty node names"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_non_string_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(KPNNError, match="'backend' must be a string"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend=1,
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_unsupported_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(KPNNError, match="Unsupported backend"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="unknown_backend",
            quiet=False,
        )


def test_validate_compile_graph_inputs_raises_for_non_boolean_quiet():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(KPNNError, match="'quiet' must be a boolean"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet="no",
        )
