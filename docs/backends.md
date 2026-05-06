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
acceleration.

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
- you want the most mature current node-interpretation support

## Recurrent backend

### Meaning

The `recurrent` backend compiles the graph into a recurrent node-state model.

Instead of layering the graph into a sequence of feedforward blocks, this
backend keeps the original graph topology and applies repeated state updates
over the graph for a fixed number of steps.

### Structural properties

The `recurrent` backend:

- allows cyclic graphs
- keeps the original graph topology
- updates node states repeatedly over multiple steps
- uses masks to enforce graph-defined recurrent connectivity
- re-injects input-node values after each recurrent update step

### Internal execution pattern

Conceptually, the `recurrent` backend behaves like a sparse recurrent neural
system over graph nodes.

Each node has a position in a global node-state vector. At each step, the model
updates node states using the graph-defined connectivity mask. Input nodes are
then re-injected so that external inputs remain anchored across recurrent
updates.

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

The `graphnn` backend compiles the graph into a graph-oriented
message-passing-style model over node states.

Like the recurrent backend, it keeps the original graph topology instead of
forcing the graph into a layer-wise feedforward structure.

### Structural properties

The `graphnn` backend:

- allows cyclic or non-layerable graphs
- keeps the original graph structure
- performs repeated graph-defined node-state updates
- uses masks to enforce graph-derived message-passing connectivity
- re-injects input-node values after each update step

### Internal execution pattern

Conceptually, the current `graphnn` backend is a minimal graph-oriented
message-passing model over named nodes.

It provides a conservative GraphNN-style backend that fits the same
compile/train/interpret interface as the other backends, while leaving room for
future backend-specific extensions.

### Pseudo nodes

Pseudo nodes are not used by the `graphnn` backend.

GraphNN-style models are generally intended to operate directly on the original
graph topology, so pseudo-node expansion is not an obvious default mechanism
for this backend.

### When to use it

Use `graphnn` when:

- you want to preserve the original graph topology
- the graph is not naturally feedforward
- you want a graph-oriented backend rather than a layered one

## Summary of backend behavior

| Backend | Layered execution | Cycles allowed | Pseudo nodes | Main internal idea |
|---|---:|---:|---:|---|
| `feedforward` | yes | no | yes | sparse layer-wise computation |
| `recurrent` | no | yes | no | sparse recurrent node-state updates |
| `graphnn` | no | yes | no | sparse message-passing node updates |

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

## Choosing a backend

A good rule of thumb is:

- choose `feedforward` for hierarchical graphs and the most mature current
  node-interpretation path
- choose `recurrent` for cyclic graphs and recurrent state-update semantics
- choose `graphnn` when you want a graph-oriented backend that preserves the
  original topology more directly

For an initial model, `feedforward` is usually the best starting point when the
graph is acyclic and can be interpreted as a hierarchy. The `recurrent` and
`graphnn` backends are useful when the graph topology itself is not naturally
layerable.
