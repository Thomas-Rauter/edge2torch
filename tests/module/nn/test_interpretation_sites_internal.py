import pytest
from torch import nn

from edge2torch.nn.interpretation_sites import (
    find_interpretation_site_provider,
    parse_feedforward_site_id,
    parse_state_update_site_id,
)
from edge2torch.utils.errors import Edge2TorchError


class _ModelWithNestedSiteAccess(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled = _CompiledModelWithSiteAccess()

    def forward(self, x):
        return self.compiled(x)


class _CompiledModelWithSiteAccess(nn.Module):
    def forward(self, x):
        return x

    def _edge2torch_list_interpretation_site_ids(self):
        return ["layer_1"]

    def _edge2torch_get_interpretation_site(self, site_id: str):
        return self


def test_parse_feedforward_site_id_rejects_input_layer():
    with pytest.raises(Edge2TorchError, match="not an interpretation site"):
        parse_feedforward_site_id("layer_0")


def test_parse_state_update_site_id_maps_step_numbers_to_zero_based_index():
    assert parse_state_update_site_id("step_1") == 0
    assert parse_state_update_site_id("step_3") == 2


def test_find_interpretation_site_provider_finds_nested_compiled_model():
    model = _ModelWithNestedSiteAccess()

    provider = find_interpretation_site_provider(model)

    assert provider is model.compiled
