"""
Read-only graph topology view over a compiled artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from edge2torch.compile.artifact import CompileArtifact
from edge2torch.utils.constants import COMPILE_BACKENDS, CompileBackend
from edge2torch.utils.errors import Edge2TorchError


def graph_topology(artifact: CompileArtifact) -> GraphTopology:
    """
    Build a stable, read-only topology view from a compilation artifact.

    The returned object exposes graph structure and interpretation-site
    metadata without surfacing internal graph objects, execution plans, or
    model internals.
    """
    _validate_topology_artifact(artifact)

    sites = {
        site_id: tuple(nodes)
        for site_id, nodes in artifact.interpretation_sites.items()
    }

    return GraphTopology(
        backend=artifact.backend,
        feature_names=tuple(artifact.feature_names),
        input_nodes=tuple(artifact.input_nodes),
        output_nodes=tuple(artifact.output_nodes),
        hidden_nodes=tuple(artifact.hidden_nodes),
        interpretation_sites=MappingProxyType(sites),
    )


@dataclass(frozen=True, slots=True)
class GraphTopology:
    """
    Immutable topology metadata for a compiled graph.
    """

    backend: CompileBackend
    feature_names: tuple[str, ...]
    input_nodes: tuple[str, ...]
    output_nodes: tuple[str, ...]
    hidden_nodes: tuple[str, ...]
    interpretation_sites: Mapping[str, tuple[str, ...]]

    @property
    def site_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.interpretation_sites.keys(),
                key=_site_sort_key,
            )
        )

    @property
    def is_feedforward(self) -> bool:
        return self.backend == "feedforward"

    @property
    def is_state_update(self) -> bool:
        return self.backend == "state_update"


def _validate_topology_artifact(artifact: object) -> None:
    if not isinstance(artifact, CompileArtifact):
        raise Edge2TorchError("'artifact' must be a CompileArtifact instance.")

    if artifact.backend not in COMPILE_BACKENDS:
        supported = ", ".join(sorted(COMPILE_BACKENDS))
        raise Edge2TorchError(
            f"Unsupported artifact backend '{artifact.backend}'. "
            f"Expected one of: {supported}."
        )

    _validate_str_tuple_field(
        artifact.feature_names,
        field_name="feature_names",
    )
    _validate_str_tuple_field(
        artifact.input_nodes,
        field_name="input_nodes",
    )
    _validate_str_tuple_field(
        artifact.output_nodes,
        field_name="output_nodes",
    )
    _validate_str_tuple_field(
        artifact.hidden_nodes,
        field_name="hidden_nodes",
    )

    if not isinstance(artifact.interpretation_sites, dict):
        raise Edge2TorchError(
            "'artifact.interpretation_sites' must be a dictionary."
        )

    for site_id, nodes in artifact.interpretation_sites.items():
        if not isinstance(site_id, str):
            raise Edge2TorchError(
                "'artifact.interpretation_sites' keys must be strings."
            )
        _site_sort_key(site_id)
        _validate_str_tuple_field(
            nodes,
            field_name=f"interpretation_sites['{site_id}']",
        )


def _validate_str_tuple_field(values: object, *, field_name: str) -> None:
    if not isinstance(values, list):
        raise Edge2TorchError(f"'artifact.{field_name}' must be a list.")

    if not all(isinstance(value, str) for value in values):
        raise Edge2TorchError(
            f"'artifact.{field_name}' must contain only strings."
        )


def _site_sort_key(site_id: str) -> tuple[int, int]:
    if site_id.startswith("layer_"):
        try:
            return (0, int(site_id.split("_")[1]))
        except (IndexError, ValueError) as exc:
            raise Edge2TorchError(
                f"Invalid interpretation site '{site_id}' in artifact."
            ) from exc

    if site_id.startswith("step_"):
        try:
            return (1, int(site_id.split("_")[1]))
        except (IndexError, ValueError) as exc:
            raise Edge2TorchError(
                f"Invalid interpretation site '{site_id}' in artifact."
            ) from exc

    raise Edge2TorchError(
        f"Invalid interpretation site '{site_id}' in artifact."
    )
