# Backends

`edge2torch` can compile the same architecture edgelist into different
backend-specific sparse PyTorch model classes.

Each backend answers a slightly different question:

- how should graph structure be translated into neural-network computation?
- which graph properties are allowed?
- what internal execution structure should be created?

The goal of `edge2torch` is to stay **minimally opinionated**. A backend
defines only the structural semantics required for compilation. Broader
modeling choices such as activations, heads, losses, optimizers, and training
loops remain the user's responsibility.

## Overview

The currently implemented backends are:

- `feedforward`
- `state_update`

Both backends share the same high-level interface:

- they are created with `compile_graph(...)`
- they return a normal PyTorch `nn.Module`
- they return a `CompileArtifact` that preserves information needed for
  alignment, inspection, and interpretation

Across all backends, input and output nodes are inferred from graph structure:

- input nodes are nodes with no incoming edges
- output nodes are nodes with no outgoing edges

The current implementation enforces sparse graph-derived connectivity with
masked dense PyTorch layers. These masks constrain trainable edge weights: a
source-to-target weighted connection can contribute only where the compiled
graph contains the corresponding directed edge.

Compiled layers include bias terms by default. Biases are node-level offsets,
not graph edges, and are not constrained by the edge mask. Set `bias=False` in
`compile_graph()` to remove these offsets so node updates depend only on
graph-defined weighted inputs.

The use of masked dense layers should not be interpreted as sparse tensor
acceleration, which is planned as a future addition.

## Common input format

All backends currently start from the same edgelist representation.

The edgelist must contain two required columns:

- `source`
- `target`

For example:

```python
import pandas as pd

edgelist = pd.DataFrame(
    {
        "source": ["feature_a", "feature_b", "hidden"],
        "target": ["hidden", "hidden", "prediction"],
    }
)
```

Each row defines a directed edge from one named node to another. The edgelist
may come from domain knowledge, a manually designed sparse architecture, a
discovered graph, or any other source that can be represented as directed
connections between named nodes.

The backend changes the **compiled execution semantics**, not the graph input
format.

## Feedforward backend

### Meaning

The `feedforward` backend compiles the graph into a strictly layer-wise sparse
PyTorch model.

This backend assumes that the graph can be organized into successive layers.
If the graph contains skip edges, these are expanded internally using pseudo
nodes so that the final compiled computation remains strictly adjacent
layer-to-layer.

### Structural properties

The `feedforward` backend:

- expects an acyclic, layerable directed graph
- compiles computation into successive layer blocks
- uses masks to enforce graph-derived connectivity between adjacent layers
- expands skip edges internally through pseudo nodes when needed

### Internal execution pattern

Conceptually, the `feedforward` backend behaves like a sparse multilayer
perceptron whose connectivity pattern is derived from the edgelist.

The compiled model contains one computation block per adjacent layer pair.
Each block applies a masked linear transformation so that only graph-defined
weighted connections contribute through edge weights. If `bias=True`, target
nodes may also have learned node-level offsets. Set `bias=False` to remove
these offsets.

### Pseudo nodes

Pseudo nodes are internal compiler-generated nodes used by the `feedforward`
backend to represent skip edges in a strictly layer-wise form.

They are stored in the compilation artifact for internal bookkeeping, but they
are hidden from user-facing node attribution outputs. User-facing node
interpretation reports named graph nodes rather than compiler-generated
pseudo nodes.

### When to use it

Use `feedforward` when:

- the graph is naturally hierarchical or approximately hierarchical
- you want a sparse layered architecture


## State-update backend

### Meaning

The `state_update` backend compiles the graph into a topology-preserving
node-state model.

Instead of layering the graph into feedforward blocks, this backend keeps the
original graph topology and applies repeated masked linear state updates for a
configurable number of steps.

### Structural properties

The `state_update` backend:

- allows cyclic graphs
- keeps the original graph topology
- requires at least one inferred input node
- requires at least one inferred output node
- requires every node that can influence an output node to be reachable from at
  least one inferred input node
- updates node states repeatedly over multiple steps
- uses masks to enforce graph-defined connectivity
- re-injects input-node values after each state-update step

Cycles are allowed, but disconnected components that can influence an output are
rejected because their contributions cannot depend on the provided input
features.

### Internal execution pattern

Conceptually, the `state_update` backend behaves like a sparse iterative
state-refinement system over graph nodes.

Each node has a position in a global node-state vector. At each step, the model
updates node states using the graph-defined connectivity mask. Input nodes are
then re-injected so that external inputs remain anchored across updates.

The `steps` argument to `compile_graph()` controls how many state-update steps
are applied during each forward pass. It is not a training epoch count and does
not represent a sequence length in the input data.

Access the shared masked linear layer through `model.state_linear`.

### Pseudo nodes

Pseudo nodes are not used by the `state_update` backend.

The backend does not need to force computation into adjacent feedforward layers.

### When to use it

Use `state_update` when:

- the graph contains cycles or feedback
- repeated state updates are a natural representation of the system
- you want to preserve the original edgelist topology rather than layerizing it

## Summary of backend behavior

| Backend | Layered execution | Cycles allowed | Pseudo nodes | Main internal idea |
|---|---:|---:|---:|---|
| `feedforward` | yes | no | yes | sparse layer-wise computation |
| `state_update` | no | yes | no | fixed-step masked state updates |

## Interpretation support

Node and feature interpretation are available for all implemented backends.

| Backend | Feature attribution | Node attribution |
|---|---:|---:|
| `feedforward` | yes | yes |
| `state_update` | yes | yes |

Feature attribution is available through feature-level Captum methods such as
`IntegratedGradients`, `Saliency`, and `DeepLift`.

Node attribution is available through layer-level Captum methods such as
`LayerConductance` and `LayerIntegratedGradients`.

### Interpretation sites

Node attribution is computed at **interpretation sites** inside the compiled
model:

- `feedforward`: one site per non-input layer (`layer_1`, `layer_2`, ...)
- `state_update`: one site per unrolled state-update step (`step_1`, `step_2`,
  ...)

Pseudo nodes used internally by the feedforward backend are never exposed in
user-facing node interpretation output.

### Summary vs per-site node results

`interpret_model(..., target="nodes")` supports two detail levels:

| `level` | Return type | Meaning |
|---|---|---|
| `"summary"` (default) | `pandas.DataFrame` | One node-importance table per sample |
| `"sites"` | `dict[str, pandas.DataFrame]` | One table per interpretation site |

Use the `nodes` parameter to filter which graph nodes appear in the result:

| `nodes` | Included nodes |
|---|---|
| `"hidden"` (default) | Internal graph nodes only |
| `"non_input"` | All nodes except inputs (includes outputs) |
| `"all"` | All visible graph nodes |

For the `state_update` backend, summary results aggregate repeated node
columns across steps. Control this with `site_aggregation`:

| `site_aggregation` | Behavior |
|---|---|
| `"max_abs"` (default) | Keep the step value with largest absolute magnitude |
| `"mean_abs"` | Average absolute values across steps |
| `"last"` | Use the final step only |

Feedforward summary results merge disjoint site columns. `site_aggregation` is
ignored for feedforward summary output and for `level="sites"`.

Models returned by `customize_model()` support node interpretation when the
wrapped compiled model remains accessible for interpretation-site lookup.
Adding a custom output `head` can change output dimensionality and may require
Captum `attribute_kwargs` such as `target` for multi-output graphs.

See [Scope and limitations](scope.md) for what each backend is designed to
cover.
