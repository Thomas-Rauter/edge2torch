# Backends

`kpnn` can compile the same prior-knowledge graph into different backend-
specific PyTorch model classes.

Each backend answers a slightly different question:

- how should graph structure be translated into neural-network computation?
- which graph properties are allowed?
- what internal execution structure should be created?

The goal of `kpnn` is to stay **minimally opinionated**. This means that a
backend defines only the structural semantics required for compilation, while
broader modeling choices such as activations, heads, losses, optimizers, and
training loops remain the user's responsibility.

## Overview

The currently implemented backends are:

- `feedforward`
- `recurrent`
- `graphnn`

All three backends share the same high-level interface:

- they are created with `compile_graph(...)`
- they return a normal PyTorch `nn.Module`
- they return a `KPNNArtifact` that preserves the information needed later for
  interpretation

## Common input format

All backends currently start from the same edgelist representation.

The edgelist must contain two required columns:

- `source`
- `target`

For example:

```python
edgelist = pd.DataFrame(
    {
        "source": ["gene_1", "gene_2", "pathway_1"],
        "target": ["pathway_1", "pathway_1", "output_1"],
    }
)
```

The backend changes the **compiled execution semantics**, not the graph input
format.

## Feedforward backend

### Meaning

The `feedforward` backend compiles the graph into a strictly layer-wise
feedforward PyTorch model.

This backend assumes that the graph can be organized into successive layers.
If the graph contains skip edges, these are expanded internally using pseudo
nodes so that the final compiled computation remains strictly adjacent
layer-to-layer.

### Structural properties

The feedforward backend:

- expects a layerable directed graph
- compiles computation into successive layer blocks
- uses masked sparse connectivity between adjacent layers
- expands skip edges internally through pseudo nodes when needed

### Internal execution pattern

Conceptually, the feedforward backend behaves like a sparse multilayer
perceptron whose sparsity pattern is derived from the graph.

The compiled model contains one computation block per adjacent layer pair.

### Inputs and outputs

By default:

- zero-in-degree nodes define the input feature space
- zero-out-degree nodes define the returned output space

### Pseudo nodes

Pseudo nodes are currently implemented explicitly for the feedforward backend.

They are:

- internal compiler-generated nodes
- used to represent skip edges in a strictly layer-wise form
- stored in the artifact
- hidden from user-facing node interpretation outputs

### When to use it

Use `feedforward` when:

- the graph is naturally hierarchical or approximately hierarchical
- you want a sparse layered architecture
- you want the most mature current interpretation support

### Example

```python
model, artifact = compile_graph(
    edgelist=edgelist,
    backend="feedforward",
)
```

## Recurrent backend

### Meaning

The `recurrent` backend compiles the graph into a recurrent node-state model.

Instead of layering the graph into a sequence of feedforward blocks, this
backend keeps the original graph structure and applies repeated state updates
over the graph for a fixed number of steps.

### Structural properties

The recurrent backend:

- allows cyclic graphs
- keeps the original graph topology rather than expanding it into layers
- updates node states repeatedly over multiple steps
- uses graph-defined sparse connectivity for the recurrent update

### Internal execution pattern

Conceptually, the recurrent backend behaves like a sparse recurrent neural
system over graph nodes.

Each node has a position in a global node-state vector, and the model updates
that state iteratively.

### Inputs and outputs

By default:

- zero-in-degree nodes define the external input interface
- zero-out-degree nodes define the returned output interface

### Pseudo nodes

Pseudo nodes are **not currently implemented** for the recurrent backend.

They are conceivable in principle as an internal compiler mechanism, but they
are not required by the current recurrent compilation strategy and are not
used at present.

### When to use it

Use `recurrent` when:

- the graph contains cycles
- repeated state updates are a natural representation of the system
- you want a graph-structured recurrent model rather than a layered one

### Example

```python
model, artifact = compile_graph(
    edgelist=edgelist,
    backend="recurrent",
)
```

## GraphNN backend

### Meaning

The `graphnn` backend compiles the graph into a graph-based message-passing
style model over node states.

Like the recurrent backend, it keeps the original graph topology instead of
forcing a layer-wise feedforward structure.

### Structural properties

The graphnn backend:

- allows cyclic or non-layerable graphs
- keeps the original graph structure
- performs repeated graph-defined node-state updates
- is intended as the graph-neural-network-oriented backend in the package

### Internal execution pattern

Conceptually, the current implementation is a minimal message-passing style
model over graph nodes.

It is best understood as an initial graphnn backend skeleton that already
fits cleanly into the package architecture, while leaving room for future
backend-specific refinement.

### Inputs and outputs

By default:

- zero-in-degree nodes define the external input interface
- zero-out-degree nodes define the returned output interface

### Pseudo nodes

Pseudo nodes are **not currently implemented** for the graphnn backend.

More importantly, pseudo nodes are not an obvious default mechanism for graph
neural network compilation, because graphnn models are typically intended to
operate directly on the original graph topology.

### When to use it

Use `graphnn` when:

- you want to preserve the original graph topology
- the graph is not naturally feedforward
- you want a graph-oriented backend rather than a layered one

### Example

```python
model, artifact = compile_graph(
    edgelist=edgelist,
    backend="graphnn",
)
```

## Summary of backend behavior

| Backend     | Layered execution | Cycles allowed | Pseudo nodes currently implemented | Main internal idea                  |
|-------------|-------------------|----------------|------------------------------------|-------------------------------------|
| feedforward | yes               | no             | yes                                | sparse layer-wise computation       |
| recurrent   | no                | yes            | no                                 | sparse recurrent node-state updates |
| graphnn     | no                | yes            | no                                 | sparse message-passing node updates |

## Interpretation support

Current interpretation support is backend-dependent.

| Backend     | `target="features"` + `integrated_gradients` | `target="nodes"` + `layer_conductance` | `target="nodes"` + `layer_integrated_gradients` |
|-------------|----------------------------------------------|----------------------------------------|-------------------------------------------------|
| feedforward | yes                                          | yes                                    | yes                                             |
| recurrent   | yes                                          | no                                     | no                                              |
| graphnn     | yes                                          | no                                     | no                                              |

## Choosing a backend

A good rule of thumb is:

- choose `feedforward` for hierarchical graphs and the most mature current
  interpretation path
- choose `recurrent` for cyclic graphs and recurrent state-update semantics
- choose `graphnn` when you want a graph-oriented backend that preserves the
  original topology more directly

## Design principle

All backends in `kpnn` follow the same design principle:

- the backend defines only the structural semantics needed for compilation
- the returned object is still an ordinary PyTorch model
- downstream architectural choices remain outside `compile_graph(...)`

This is why it is natural to use `customize_model()` after compilation, or to
wrap compiled models manually with ordinary PyTorch code.
