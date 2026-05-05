import pytest
from torch import nn

from edge2torch.customize.input_validation import (
    validate_customize_model_inputs,
)
from edge2torch.utils.errors import Edge2TorchError


class _ModuleWithoutVisibleForward(nn.Module):
    def __getattribute__(self, name):
        if name == "forward":
            raise AttributeError
        return super().__getattribute__(name)


def test_validate_customize_model_inputs_rejects_non_module_model():
    with pytest.raises(Edge2TorchError, match="torch.nn.Module"):
        validate_customize_model_inputs(
            model=object(),
            activation=None,
            dropout=None,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_model_without_forward():
    model = _ModuleWithoutVisibleForward()

    with pytest.raises(Edge2TorchError, match="forward method"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout=None,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_non_module_activation():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="'activation' must be"):
        validate_customize_model_inputs(
            model=model,
            activation="relu",
            dropout=None,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_activation_equal_to_model():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="same object as 'model'"):
        validate_customize_model_inputs(
            model=model,
            activation=model,
            dropout=None,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_bool_dropout():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="'dropout' must be"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout=True,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_non_numeric_dropout():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="'dropout' must be"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout="0.5",
            head=None,
        )


def test_validate_customize_model_inputs_rejects_negative_dropout():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="0 <= dropout < 1"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout=-0.1,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_dropout_equal_to_one():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="0 <= dropout < 1"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout=1.0,
            head=None,
        )


def test_validate_customize_model_inputs_rejects_non_module_head():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="'head' must be"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout=None,
            head="head",
        )


def test_validate_customize_model_inputs_rejects_head_equal_to_model():
    model = nn.Linear(2, 1)

    with pytest.raises(Edge2TorchError, match="same object as 'model'"):
        validate_customize_model_inputs(
            model=model,
            activation=None,
            dropout=None,
            head=model,
        )


def test_validate_customize_model_inputs_rejects_activation_equal_to_head():
    model = nn.Linear(2, 1)
    activation = nn.ReLU()

    with pytest.raises(Edge2TorchError, match="must not be the same object"):
        validate_customize_model_inputs(
            model=model,
            activation=activation,
            dropout=None,
            head=activation,
        )


def test_validate_customize_model_inputs_accepts_valid_inputs():
    validate_customize_model_inputs(
        model=nn.Linear(2, 1),
        activation=nn.ReLU(),
        dropout=0.5,
        head=nn.Linear(1, 1),
    )


def test_validate_customize_model_inputs_accepts_integer_dropout():
    validate_customize_model_inputs(
        model=nn.Linear(2, 1),
        activation=None,
        dropout=0,
        head=None,
    )
