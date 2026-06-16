# Scope and limitations

This page describes what kinds of neural network you can build with
`edge2torch`, and how much custom PyTorch each case typically requires. It
complements the execution semantics in [Backends](backends.md).

`edge2torch` compiles a prior-knowledge edgelist into a sparse masked-linear
PyTorch `nn.Module` and optional Captum-based attributions. The package is
deliberately thin: training, losses, and most modeling choices stay in ordinary
PyTorch. The categories below are about **technical fit**, not whether a model
is easy to train in practice.

## How to read the categories

| Category | Meaning |
|----------|---------|
| **Well supported** | The architecture matches what the compiler assumes. You use `compile_graph()`, alignment, optional `customize_model()`, and standard PyTorch training around the result. |
| **Supported with extra work** | The model family is still compatible with masked-linear graph compilation, but you need substantial custom PyTorch—wrapping the compiled module, inserting nonlinearities inside the forward path, or building logic the compiler does not expose. |
| **Not supported** | The design conflicts with assumptions baked into `compile_graph()` and the backends. No amount of wrapping is expected to map cleanly onto the current API. |

## Shared across all backends

| Category | |
|----------|---|
| **Well supported** | Edgelists with `source` and `target` columns as the architecture spec. Automatic inference of input nodes (no incoming edges) and output nodes (no outgoing edges). Masked linear connectivity with optional per-edge `initial_weight` and `constraint` (`positive`, `negative`, `fixed`, `unconstrained`). Optional node biases (`bias=True` by default). One static feature vector per sample, shape `(batch, n_features)`. `align_features_to_input_nodes()` for name-based alignment. `customize_model()` for activation, dropout, and a task head **after** the compiled graph core. Training and evaluation with standard PyTorch (loss, optimizer, loops). Feature attribution (`target="features"`) and node attribution (`target="nodes"`) when the `[interpret]` extra is installed. |
| **Supported with extra work** | Nonlinearities **inside** the compiled forward path (between internal layers or recurrence steps), because the compiled core is linear masked updates only. Readout or aggregation that does not map to terminal output nodes alone (custom head or wrapper around the compiled module). Multi-output graphs with node interpretation (extra Captum `attribute_kwargs`, for example `target=0`). Edgelists produced by external pipelines (pathway databases, curation scripts); the compiler consumes the DataFrame, it does not import those sources. Alternative attribution methods wired by the user outside `interpret_model()`. |
| **Not supported** | Graph structure that changes per sample at compile time. Edge features, neighbor attention, or heterogeneous node types as first-class compile inputs. A native sequence or time axis in the input API (each sample is one static feature vector). Learnable message-passing primitives (GCN, GAT, GraphSAGE, and similar) inside the compiler. Sparse-tensor acceleration (implementation uses masked dense layers). Built-in trainers, dataloaders, or experiment managers. |

## Feedforward backend

Compiles acyclic, layerable graphs into successive sparse layer blocks. Skip
edges are expanded via internal pseudo nodes so execution stays strictly
adjacent layer-to-layer.

| Category | |
|----------|---|
| **Well supported** | Acyclic graphs that can be ordered into successive layers. Sparse connectivity defined only by the edgelist. Skip edges via pseudo-node expansion (see [Feedforward skip edges](feedforward-skip-edges.ipynb)). Signed or fixed edge weights. Static inputs on input nodes; predictions from terminal output nodes. Node interpretation at `layer_*` sites. Post-core activation, dropout, and task head via `customize_model()`. |
| **Supported with extra work** | Per-layer or mid-network nonlinearities (wrap or subclass the compiled module so activations run between compiled layer blocks). Multiple logical readouts combined in custom PyTorch after the graph core. Graphs with many skip edges (supported via pseudo nodes, but the expanded internal structure is larger than the original edgelist). |
| **Not supported** | Cyclic graphs (use `recurrent` or `graphnn`). Directed graphs that cannot be layerized into a feedforward execution order. Attention or message-passing layers as compiled primitives. |

## Recurrent backend

Fixed-step updates over a **full node-state vector** on the original graph
topology, including cycles. Learnable weights are shared across steps; static
input features are re-injected after each step. `steps` counts internal
state-update iterations per forward pass, not timesteps in the input data.

| Category | |
|----------|---|
| **Well supported** | Cyclic graphs and feedback loops among named nodes. Static exogenous inputs on input nodes. Configurable unrolling via `steps`. Node interpretation at `step_*` sites, with optional summary aggregation across steps (`site_aggregation`). Signed or fixed edge weights. Post-core head and activation via `customize_model()`. |
| **Supported with extra work** | Nonlinear state updates per step (custom wrapper around the compiled forward pass). Steady-state or fixed-point-style behavior implemented in user code (for example large `steps`, repeated forwards, or a custom loss on the final state). Readout logic that pools or transforms internal node states beyond the default output nodes. |
| **Not supported** | Sequence models with temporal input (LSTM, GRU, or an input tensor with a time dimension). Different learnable weight matrices per unrolled step. Gated recurrence inside the compiled graph. Adaptive iterate-until-convergence loops inside `compile_graph()`. |

## GraphNN backend

Preserves the original graph topology without feedforward layerization. In
the current release the compiled update rule is the same masked linear
recurrence as `recurrent` (shared weights across steps, input re-injection,
fixed `steps`). The backend name reflects graph-native topology handling; it
is not a full graph neural network library.

| Category | |
|----------|---|
| **Well supported** | Cyclic graphs and direct skip edges without pseudo-node layer expansion. Static inputs and fixed-step unrolling with `step_*` interpretation sites. Same edge metadata, alignment, and attribution workflow as the other backends. |
| **Supported with extra work** | The same extensions as for `recurrent` (internal nonlinearities, custom readout, user-driven steady-state logic), when you choose `graphnn` to keep the edgelist topology explicit rather than layerized. |
| **Not supported** | Standard GNN layer types as compiled primitives. Edge embeddings or learned attention over neighbors inside the compiler. Per-node heterogeneous state sizes or typed message functions. Built-in normalization, residual connections, or graph-batch abstractions beyond ordinary PyTorch batching of aligned tensors. |

## Choosing a backend

| If your graph … | Use |
|-----------------|-----|
| Is acyclic and can be organized into layers | `feedforward` |
| Has cycles or feedback and static inputs per sample | `recurrent` |
| Must keep the raw edgelist topology and the minimal linear recurrence is acceptable | `graphnn` (core dynamics currently match `recurrent`) |

For execution details and examples, see [Backends](backends.md) and the
backend-specific notebooks.
