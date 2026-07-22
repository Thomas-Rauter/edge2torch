import pandas as pd
import pytest

from edge2torch.compile.input_validation import validate_compile_graph_inputs
from edge2torch.utils.errors import Edge2TorchError


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
        bias=True,
        steps=3,
    )


def test_validate_compile_graph_inputs_accepts_bias_false():
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
        bias=False,
        steps=3,
    )


def test_validate_compile_graph_inputs_raises_for_non_dataframe():
    edgelist = [
        {"source": "gene_1", "target": "pathway_1"},
    ]

    with pytest.raises(Edge2TorchError, match="pandas DataFrame"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_missing_source_column():
    edgelist = pd.DataFrame(
        {
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="Missing: source"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_missing_target_column():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="Missing: target"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_duplicate_source_column():
    edgelist = pd.DataFrame(
        [
            ["gene_1", "gene_1", "pathway_1"],
        ],
        columns=["source", "source", "target"],
    )

    with pytest.raises(Edge2TorchError, match="exactly once"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_duplicate_target_column():
    edgelist = pd.DataFrame(
        [
            ["gene_1", "pathway_1", "pathway_1"],
        ],
        columns=["source", "target", "target"],
    )

    with pytest.raises(Edge2TorchError, match="exactly once"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_missing_values():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", None],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(
        Edge2TorchError, match="must not contain missing values"
    ):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_empty_source_name():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", ""],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(Edge2TorchError, match="empty node names"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_whitespace_target_name():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "   "],
        }
    )

    with pytest.raises(Edge2TorchError, match="empty node names"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_int_str_label_collision():
    edgelist = pd.DataFrame(
        {
            "source": [1, "1", 2],
            "target": ["h1", "h2", "h3"],
        }
    )

    with pytest.raises(
        Edge2TorchError,
        match="become identical after converting to strings",
    ):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_strip_label_collision():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_a", " gene_a"],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(
        Edge2TorchError,
        match="become identical after converting to strings",
    ):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_accepts_homogeneous_int_labels():
    edgelist = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [3, 3],
        }
    )

    validate_compile_graph_inputs(
        edgelist=edgelist,
        backend="feedforward",
        quiet=False,
        bias=True,
        steps=3,
    )


def test_validate_compile_graph_inputs_raises_for_non_string_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="'backend' must be a string"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend=1,
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_unsupported_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="Unsupported backend"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="unknown_backend",
            quiet=False,
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_non_boolean_quiet():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="'quiet' must be a boolean"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet="no",
            bias=True,
            steps=3,
        )


def test_validate_compile_graph_inputs_raises_for_non_boolean_bias():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="'bias' must be a boolean"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="feedforward",
            quiet=False,
            bias="no",
            steps=3,
        )


def test_validate_compile_graph_inputs_accepts_valid_steps():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    validate_compile_graph_inputs(
        edgelist=edgelist,
        backend="state_update",
        quiet=False,
        bias=True,
        steps=5,
    )


def test_validate_compile_graph_inputs_rejects_zero_steps():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(
        Edge2TorchError,
        match="'steps' must be a positive integer",
    ):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="state_update",
            quiet=False,
            bias=True,
            steps=0,
        )


def test_validate_compile_graph_inputs_rejects_non_integer_steps():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="'steps' must be an integer"):
        validate_compile_graph_inputs(
            edgelist=edgelist,
            backend="state_update",
            quiet=False,
            bias=True,
            steps="3",
        )


def test_validate_compile_graph_inputs_ignores_feedforward_steps():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    validate_compile_graph_inputs(
        edgelist=edgelist,
        backend="feedforward",
        quiet=False,
        bias=True,
        steps=5,
    )
