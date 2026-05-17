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
- `recurrent`
- `graphnn`

All three backends share the same high-level interface:

- they are created with `compile_graph(...)`
- they return a normal PyTorch `nn.Module`
- they return a `CompileArtifact` that preserves information needed for
  alignment, inspection, and interpretation

Across all backends, input and output nodes are inferred from graph structure:

- input nodes are nodes with no incoming edges
- output nodes are nodes with no outgoing edges

The current implementation enforces sparse graph-derived connectivity with
masked dense PyTorch layers. This guarantees that compiled models respect the
edgelist connectivity, but it should not be interpreted as sparse tensor
acceleration (which is planned as a future addition).

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
connections contribute to the forward pass.

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


## Recurrent backend

### Meaning

The `recurrent` backend compiles the graph into a recurrent node-state model.

Instead of layering the graph into a sequence of feedforward blocks, this
backend keeps the original graph topology and applies repeated state updates
over the graph for a configurable number of steps.

### Structural properties

The `recurrent` backend:

- allows cyclic graphs
- keeps the original graph topology
- requires at least one inferred input node
- requires at least one inferred output node
- requires every output node to be reachable from at least one inferred input
  node
- updates node states repeatedly over multiple steps
- uses masks to enforce graph-defined recurrent connectivity
- re-injects input-node values after each recurrent update step

Cycles are allowed, but output nodes in disconnected cyclic components are
rejected because they cannot depend on the provided input features.

### Internal execution pattern

Conceptually, the `recurrent` backend behaves like a sparse recurrent neural
system over graph nodes.

Each node has a position in a global node-state vector. At each step, the model
updates node states using the graph-defined connectivity mask. Input nodes are
then re-injected so that external inputs remain anchored across recurrent
updates.

The `steps` argument to `compile_graph()` controls how many recurrent update
steps are applied during each forward pass. It is not a training epoch count
and does not represent a sequence length in the input data.

### Pseudo nodes

Pseudo nodes are not used by the `recurrent` backend.

They are not required by the current recurrent compilation strategy because
the backend does not need to force computation into adjacent feedforward
layers.

### When to use it

Use `recurrent` when:

- the graph contains cycles
- repeated state updates are a natural representation of the system
- you want a graph-structured recurrent model rather than a layered one

## GraphNN backend

### Meaning

The `graphnn` backend compiles the graph into a minimal graph-oriented
state-update model over named node states.

Like the recurrent backend, it keeps the original graph topology instead of
forcing the graph into a layer-wise feedforward structure. In the current
implementation, `graphnn` is intentionally close to `recurrent`: both use a
single node-state vector, masked dense updates, configurable fixed-step
iteration, and input-node re-injection.

The `graphnn` backend should therefore be understood as a lightweight
message-passing-style interface and extension point, not as a full-featured
graph neural network library backend.

### Structural properties

The `graphnn` backend:

- allows cyclic or non-layerable graphs
- keeps the original graph structure
- requires at least one inferred input node
- requires at least one inferred output node
- requires every output node to be reachable from at least one inferred input
  node
- performs repeated graph-defined node-state updates
- uses masks to enforce graph-derived connectivity
- re-injects input-node values after each update step
- currently uses the same minimal masked-update primitive as the recurrent
  backend

Cycles are allowed, but output nodes in disconnected cyclic components are
rejected because they cannot depend on the provided input features.

### Internal execution pattern

Conceptually, the current `graphnn` backend applies a configurable number of
message-passing-style updates over scalar node states.

It provides a conservative graph-oriented backend that fits the same
compile/train/interpret interface as the other backends, while leaving room for
future backend-specific extensions such as richer message functions,
aggregation choices, normalization, residual updates, or edge features.

The `steps` argument to `compile_graph()` controls how many graph update steps
are applied during each forward pass. It is not a training epoch count and does
not represent a sequence length in the input data.

### Pseudo nodes

Pseudo nodes are not used by the `graphnn` backend.

GraphNN-style models operate directly on the original graph topology, so
pseudo-node expansion is not used as a default mechanism for this backend.

### When to use it

Use `graphnn` when:

- you want to preserve the original graph topology
- the graph is not naturally feedforward
- you want a minimal graph-oriented state-update backend
- you want a backend that can serve as a future extension point for
  message-passing-style models

## Summary of backend behavior

| Backend | Layered execution | Cycles allowed | Pseudo nodes | Main internal idea |
|---|---:|---:|---:|---|
| `feedforward` | yes | no | yes | sparse layer-wise computation |
| `recurrent` | no | yes | no | sparse recurrent node-state updates |
| `graphnn` | no | yes | no | minimal message-passing-style node updates |

## Interpretation support

Current interpretation support is backend-dependent.

| Backend | Feature attribution | Node attribution |
|---|---:|---:|
| `feedforward` | yes | yes |
| `recurrent` | yes | planned |
| `graphnn` | yes | planned |

Feature attribution is available for all implemented backends through
feature-level Captum methods such as `IntegratedGradients`, `Saliency`, and
`DeepLift`.

Node attribution is currently available for the `feedforward` backend through
layer-level Captum methods such as `LayerConductance` and
`LayerIntegratedGradients`.
