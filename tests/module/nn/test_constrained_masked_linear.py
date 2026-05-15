import pytest
import torch
import torch.nn.functional as f

from edge2torch.nn.masked_linear import (
    CONSTRAINT_FIXED,
    CONSTRAINT_NEGATIVE,
    CONSTRAINT_POSITIVE,
    CONSTRAINT_UNCONSTRAINED,
    ConstrainedMaskedLinear,
    _softplus_inverse,
    constraint_name_to_code,
)


def test_constraint_name_to_code_maps_supported_names():
    assert constraint_name_to_code("unconstrained") == CONSTRAINT_UNCONSTRAINED
    assert constraint_name_to_code("positive") == CONSTRAINT_POSITIVE
    assert constraint_name_to_code("negative") == CONSTRAINT_NEGATIVE
    assert constraint_name_to_code("fixed") == CONSTRAINT_FIXED


def test_constraint_name_to_code_normalizes_input():
    assert constraint_name_to_code(" Positive ") == CONSTRAINT_POSITIVE
    assert constraint_name_to_code("NEGATIVE") == CONSTRAINT_NEGATIVE


def test_constraint_name_to_code_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported constraint"):
        constraint_name_to_code("bounded")


def test_softplus_inverse_round_trips_positive_values():
    values = torch.tensor([0.01, 0.1, 1.0, 10.0])

    recovered = f.softplus(_softplus_inverse(values))

    torch.testing.assert_close(recovered, values)


def test_effective_weight_uses_all_constraint_types():
    mask = torch.ones(2, 2)
    initial_weight = torch.tensor(
        [
            [0.25, -0.50],
            [0.75, 1.25],
        ]
    )
    constraint = torch.tensor(
        [
            [CONSTRAINT_POSITIVE, CONSTRAINT_NEGATIVE],
            [CONSTRAINT_UNCONSTRAINED, CONSTRAINT_FIXED],
        ]
    )

    layer = ConstrainedMaskedLinear(
        in_features=2,
        out_features=2,
        mask=mask,
        initial_weight=initial_weight,
        constraint=constraint,
        bias=False,
    )

    expected = torch.tensor(
        [
            [0.25, -0.50],
            [0.75, 1.25],
        ]
    )

    torch.testing.assert_close(layer.effective_weight, expected)


def test_missing_initial_weights_are_finite_after_reset():
    torch.manual_seed(0)

    mask = torch.ones(2, 2)
    initial_weight = torch.full((2, 2), float("nan"))
    constraint = torch.tensor(
        [
            [CONSTRAINT_POSITIVE, CONSTRAINT_NEGATIVE],
            [CONSTRAINT_UNCONSTRAINED, CONSTRAINT_UNCONSTRAINED],
        ]
    )

    layer = ConstrainedMaskedLinear(
        in_features=2,
        out_features=2,
        mask=mask,
        initial_weight=initial_weight,
        constraint=constraint,
        bias=False,
    )

    assert torch.isfinite(layer.effective_weight).all()
    assert layer.effective_weight[0, 0] > 0
    assert layer.effective_weight[0, 1] < 0


def test_mask_zeros_out_nonexistent_edges():
    mask = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    initial_weight = torch.tensor(
        [
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )

    layer = ConstrainedMaskedLinear(
        in_features=2,
        out_features=2,
        mask=mask,
        initial_weight=initial_weight,
        bias=False,
    )

    x = torch.tensor([[1.0, 1.0]])

    output = layer(x)

    expected = torch.tensor([[2.0, 5.0]])
    torch.testing.assert_close(output, expected)


def test_reset_restores_explicit_initial_weights():
    torch.manual_seed(0)

    mask = torch.ones(2, 2)
    initial_weight = torch.tensor(
        [
            [0.25, -0.50],
            [0.75, float("nan")],
        ]
    )
    constraint = torch.tensor(
        [
            [CONSTRAINT_POSITIVE, CONSTRAINT_NEGATIVE],
            [CONSTRAINT_FIXED, CONSTRAINT_UNCONSTRAINED],
        ]
    )

    layer = ConstrainedMaskedLinear(
        in_features=2,
        out_features=2,
        mask=mask,
        initial_weight=initial_weight,
        constraint=constraint,
        bias=False,
    )

    with torch.no_grad():
        layer.raw_weight.add_(10.0)

    layer.reset_parameters()

    weight = layer.effective_weight.detach()

    assert float(weight[0, 0]) == pytest.approx(0.25)
    assert float(weight[0, 1]) == pytest.approx(-0.50)
    assert float(weight[1, 0]) == pytest.approx(0.75)
    assert torch.isfinite(weight[1, 1])


def test_fixed_edges_require_initial_weights():
    mask = torch.ones(1, 1)
    initial_weight = torch.tensor([[float("nan")]])
    constraint = torch.tensor([[CONSTRAINT_FIXED]])

    with pytest.raises(ValueError, match="fixed"):
        ConstrainedMaskedLinear(
            in_features=1,
            out_features=1,
            mask=mask,
            initial_weight=initial_weight,
            constraint=constraint,
            bias=False,
        )


def test_fixed_weight_is_not_trainable():
    mask = torch.ones(1, 1)
    initial_weight = torch.tensor([[0.5]])
    constraint = torch.tensor([[CONSTRAINT_FIXED]])

    layer = ConstrainedMaskedLinear(
        in_features=1,
        out_features=1,
        mask=mask,
        initial_weight=initial_weight,
        constraint=constraint,
        bias=False,
    )

    parameter_names = {name for name, _ in layer.named_parameters()}
    buffer_names = {name for name, _ in layer.named_buffers()}

    assert "raw_weight" in parameter_names
    assert "fixed_weight" not in parameter_names
    assert "fixed_weight" in buffer_names


def test_gradients_flow_to_trainable_raw_weight():
    mask = torch.ones(1, 2)
    initial_weight = torch.tensor([[0.5, 1.0]])
    constraint = torch.tensor(
        [[CONSTRAINT_POSITIVE, CONSTRAINT_UNCONSTRAINED]]
    )

    layer = ConstrainedMaskedLinear(
        in_features=2,
        out_features=1,
        mask=mask,
        initial_weight=initial_weight,
        constraint=constraint,
        bias=False,
    )

    x = torch.tensor([[2.0, 3.0]])
    loss = layer(x).sum()
    loss.backward()

    assert layer.raw_weight.grad is not None
    assert layer.raw_weight.grad[0, 0] != 0
    assert layer.raw_weight.grad[0, 1] != 0


def test_fixed_edges_do_not_change_after_optimizer_step():
    mask = torch.ones(1, 2)
    initial_weight = torch.tensor([[0.5, 1.0]])
    constraint = torch.tensor(
        [[CONSTRAINT_FIXED, CONSTRAINT_UNCONSTRAINED]]
    )

    layer = ConstrainedMaskedLinear(
        in_features=2,
        out_features=1,
        mask=mask,
        initial_weight=initial_weight,
        constraint=constraint,
        bias=False,
    )

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    fixed_before = float(layer.effective_weight[0, 0].detach())

    x = torch.tensor([[2.0, 3.0]])
    loss = layer(x).sum()
    loss.backward()
    optimizer.step()

    fixed_after = float(layer.effective_weight[0, 0].detach())

    weight = layer.effective_weight.detach()

    assert fixed_after == pytest.approx(fixed_before)
    assert float(weight[0, 1]) != pytest.approx(1.0)