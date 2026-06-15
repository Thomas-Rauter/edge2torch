import pytest
import torch
from torch import nn

from edge2torch.nn.customized_model import CustomizedEdgeModel


class _BaseModelWithInterpretationSites(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(2, 1)

    def forward(self, x):
        return self.block(x)

    def _edge2torch_list_interpretation_site_ids(self):
        return ["layer_1"]

    def _edge2torch_get_interpretation_site(self, site_id: str):
        if site_id != "layer_1":
            raise ValueError("unexpected site id")

        return self.block


class _BaseModelWithoutInterpretationSites(nn.Module):
    def forward(self, x):
        return x


def test_customized_edge_model_delegates_interpretation_site_access():
    base_model = _BaseModelWithInterpretationSites()
    model = CustomizedEdgeModel(base_model=base_model)

    assert model._edge2torch_list_interpretation_site_ids() == ["layer_1"]

    result = model._edge2torch_get_interpretation_site("layer_1")

    assert result is base_model.block


def test_customized_edge_model_raises_if_base_model_has_no_site_access():
    base_model = _BaseModelWithoutInterpretationSites()
    model = CustomizedEdgeModel(base_model=base_model)

    with pytest.raises(
        AttributeError,
        match="object has no attribute "
        "'_edge2torch_list_interpretation_site_ids'",
    ):
        model._edge2torch_list_interpretation_site_ids()

    with pytest.raises(
        AttributeError,
        match="object has no attribute '_edge2torch_get_interpretation_site'",
    ):
        model._edge2torch_get_interpretation_site("layer_1")


def test_customized_edge_model_forward_applies_optional_components():
    base_model = nn.Linear(2, 2, bias=False)
    activation = nn.ReLU()
    head = nn.Linear(2, 1, bias=False)

    with torch.no_grad():
        base_model.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        )
        head.weight.copy_(torch.tensor([[1.0, 1.0]]))

    model = CustomizedEdgeModel(
        base_model=base_model,
        activation=activation,
        dropout=None,
        head=head,
    )

    x = torch.tensor([[2.0, 3.0]])

    result = model(x)

    expected = torch.tensor([[2.0]])

    assert torch.equal(result, expected)
