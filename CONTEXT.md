# edge2torch — AI context document

This file is **AI-first documentation** for assistants working in this repository
or explaining the package to users. It is more detailed and operational than
`README.md`. Human docs live at https://Thomas-Rauter.github.io/edge2torch/.

**Current release:** `0.2.0` (see `pyproject.toml`, `CHANGELOG.md`).

---

## One-sentence summary

`edge2torch` compiles a **pandas edgelist of named nodes** into a **sparse masked
PyTorch `nn.Module`**, lets users train it with ordinary PyTorch, and optionally
attributes predictions back to **input features** or **named graph nodes** via
Captum.

---

## What this package does

1. **Input:** A directed graph as a DataFrame with columns `source` and `target`
   (one row = one edge). Optional columns `initial_weight` and `constraint` per
   edge.
2. **Compile:** `compile_graph()` validates the graph, picks a **backend**
   (`feedforward`, `recurrent`, or `graphnn`), and returns `(model, artifact)`.
3. **Align data:** `align_features_to_input_nodes()` maps user DataFrames (or
   AnnData / tensors) to the tensor layout expected by `model`.
4. **Customize (optional):** `customize_model()` wraps the compiled model with
   activation, dropout, and/or a task head **after** the graph core (not inside
   it).
5. **Train:** User writes standard PyTorch training loops (loss, optimizer,
   epochs). The package does **not** provide trainers.
6. **Interpret (optional):** `interpret_model()` runs Captum attribution for
   `target="features"` or `target="nodes"`.

### Primary use cases

- **Knowledge-primed neural networks (KPNNs):** prior knowledge defines
  connectivity; weights are learned from data.
- **Biologically informed models:** genes, TFs, ligands, pathways as named
  nodes (similar in spirit to frameworks like LEMBAS, but edge2torch is a
  general compiler, not a signaling-specific simulator).
- **Sparse structured NNs** where full `nn.Sequential` wiring is tedious.

### What edge2torch is NOT

- Not a full GNN library (no generic message-passing variants, edge features, or
  graph batching abstractions beyond fixed-step masked updates).
- Not an LSTM/GRU/sequence-time-series toolkit. `steps` on recurrent/graphnn
  backends are **internal state-update iterations per forward pass**, not
  timesteps in the input data.
- Not a training framework (no built-in losses for steady state, spectral
  radius, knockouts, etc.).
- Not sparse-tensor-accelerated yet (masked **dense** linear layers).
- Compiled cores are **linear** masked updates; nonlinearities only via
  `customize_model()` on the **output** of the compiled model unless the user
  wraps further in custom PyTorch.

---

## Package philosophy

**Minimally opinionated:** edge2torch owns graph → sparse module compilation and
interpretation plumbing. Users own activations inside recurrence (if needed),
heads, losses, regularizers, CV, and domain-specific biology.

Division of labor:

| Layer | Owner |
|-------|--------|
| Graph structure, masks, backends | edge2torch |
| Training, evaluation, domain science | User (PyTorch) |
| Feature/node attribution wiring | edge2torch (Captum) |

---

## Public API (stable surface)

Exported from `edge2torch` (`src/edge2torch/__init__.py`):

| Symbol | Role |
|--------|------|
| `compile_graph` | Edgelist → `(nn.Module, CompileArtifact)` |
| `align_features_to_input_nodes` | Named data → `torch.Tensor` in input-node order |
| `customize_model` | Optional activation / dropout / head wrapper |
| `interpret_model` | Captum feature or node attribution |
| `CompileArtifact` | Compilation metadata dataclass |
| `__version__` | Package version string |

`__all__` contains exactly these names. Internal modules under `src/edge2torch/`
are not public API unless documented otherwise.

### `compile_graph(edgelist, backend="feedforward", quiet=False, bias=True, steps=3)`

**Parameters:**

- `edgelist`: `pd.DataFrame` with required `source`, `target`. Optional
  `initial_weight`, `constraint` per row.
- `backend`: `"feedforward"` | `"recurrent"` | `"graphnn"`.
- `bias`: If `True` (default), node-level biases in masked linear layers. Not
  graph edges.
- `steps`: Only for `recurrent` and `graphnn`. Positive int; number of unrolled
  state-update steps **per forward pass**. Passing `steps` to `feedforward` raises
  `Edge2TorchError`.

**Node inference (all backends):**

- **Input nodes** = nodes with no incoming edges → `artifact.feature_names`
- **Output nodes** = nodes with no outgoing edges
- **Hidden nodes** = neither input nor output nor internal pseudo nodes

**Returns:** `(model, artifact)`. `model(x)` expects `x` shape
`(batch, len(artifact.feature_names))`.

### `align_features_to_input_nodes(data, artifact)`

**Input types:**

- `pd.DataFrame`: columns must match `artifact.feature_names` exactly (any
  order); reordered by name.
- `anndata.AnnData` (if `anndata` installed): align on `var_names`.
- `torch.Tensor`: shape `(n_samples, n_features)` only; column order **assumed**
  to already match `feature_names`.

**Returns:** `torch.float32` tensor `(n_samples, n_features)`.

### `customize_model(model, activation=None, dropout=None, head=None)`

Wraps compiled model in `CustomizedEdgeModel`: `base_model` → optional activation
→ optional dropout → optional head. Does **not** modify internal graph blocks.
Repeated calls nest wrappers.

Node interpretation requires the compiled `base_model` to remain discoverable
inside the wrapper (default wrapper preserves this).

**Captum note:** For scalar classification, use `head=nn.Linear(n_outputs, 1)`.
Multi-output graphs may need `attribute_kwargs={"target": i}` for node methods.

### `interpret_model(model, artifact, data, target="features", method=..., ...)`

Requires `pip install "edge2torch[interpret]"` (Captum).

**`target="features"`** → always `pd.DataFrame` (rows = samples, columns =
feature names). Methods: `IntegratedGradients`, `Saliency`, `DeepLift`, etc.
(see `interpret/method_registry.py`).

**`target="nodes"`** → layer/site attribution via Captum layer methods
(`LayerConductance`, `LayerIntegratedGradients`, etc.).

**0.2.0 node API (breaking vs 0.1.0):**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `level` | `"summary"` | One `DataFrame` per sample |
| `level` | `"sites"` | `dict[str, DataFrame]` keyed by `layer_*` or `step_*` |
| `nodes` | `"hidden"` | Columns = hidden nodes only |
| `nodes` | `"non_input"` | Includes output nodes at final sites |
| `nodes` | `"all"` | All visible graph nodes |
| `site_aggregation` | `"max_abs"` | For recurrent/graphnn **summary** only: aggregate repeated node columns across steps (`max_abs`, `mean_abs`, `last`). Ignored for feedforward summary and for `level="sites"`. |

**Returns:**

- `target="features"` → `pd.DataFrame`
- `target="nodes"`, `level="summary"` → `pd.DataFrame`
- `target="nodes"`, `level="sites"` → `dict[str, pd.DataFrame]`

`constructor_kwargs` and `attribute_kwargs` pass through to Captum unchanged.

Model is set to eval mode during attribution, then restored.

---

## Edgelist format

```python
edgelist = pd.DataFrame({
    "source": ["feature_a", "feature_b", "hidden"],
    "target": ["hidden", "hidden", "output_1"],
})
```

**Optional edge metadata** (row-wise sparse — omit on rows that use defaults):

| Column | Meaning |
|--------|---------|
| `initial_weight` | Initial effective weight for that edge |
| `constraint` | `unconstrained` (default), `positive`, `negative`, `fixed` |

`fixed` requires `initial_weight`. Sign constraints use softplus transforms on
latent parameters (`docs/edge-weight-metadata.ipynb`).

Edges point in the **direction of computation** (source → target).

---

## Backends (execution semantics)

Same edgelist can compile to different PyTorch classes. Backend changes **how**
the graph is executed, not the input format.

### Comparison table

| Backend | Cycles | Layered | Pseudo nodes | Interpretation sites | `steps` |
|---------|--------|---------|--------------|----------------------|---------|
| `feedforward` | no | yes | yes (skip edges) | `layer_1`, `layer_2`, ... | N/A |
| `recurrent` | yes | no | no | `step_1`, `step_2`, ... | required |
| `graphnn` | yes | no | no | `step_1`, `step_2`, ... | required |

### `feedforward`

- Graph must be acyclic and **layerable**.
- Skip edges (non-adjacent layer connections) expanded via **pseudo nodes**
  (internal only; never shown in user-facing node interpretation).
- One `FeedforwardLayerBlock` per layer transition; masked linear between
  adjacent layers.

### `recurrent` and `graphnn`

Both currently share the same core implementation pattern (`StateUpdateStep` in
`nn/step_block.py`):

1. Initialize full node-state vector; copy inputs into input node indices.
2. For each of `steps` iterations: apply **shared** masked linear on full
   state, then **re-inject** input feature values at input indices.
3. Return values at output node indices.

**Important:** This is graph-structured **iterative state refinement** from a
**static input vector per sample**, not RNN-over-time. Conceptually similar to
steady-state signaling models (e.g. LEMBAS) at the structural level, but without
built-in mechanistic activations, convergence loops, or stability regularizers.

`graphnn` is an extension-point name for future richer message passing; today it
is intentionally close to `recurrent`.

**Reachability rule:** Every node that can influence an output must be reachable
from at least one input node.

---

## `CompileArtifact` fields

**Stable for users:** `backend`, `feature_names`.

**Useful for interpretation / debugging (may change across releases):**

- `input_nodes`, `output_nodes`, `hidden_nodes`
- `interpretation_sites`: `dict[site_id, list[node_name]]`
- `graph`, `execution_plan`, `node_names_by_layer` (feedforward)

---

## Typical end-to-end workflow

```python
import pandas as pd
import torch
from torch import nn
import edge2torch as e2t

edgelist = pd.DataFrame({...})
model, artifact = e2t.compile_graph(edgelist, backend="recurrent", steps=2)

x_df = pd.DataFrame({...})  # columns = artifact.feature_names
x = e2t.align_features_to_input_nodes(x_df, artifact)

customized = e2t.customize_model(model, head=nn.Linear(1, 1))
# ... train with PyTorch ...

features = e2t.interpret_model(customized, artifact, x_df, target="features",
                             method="IntegratedGradients")
nodes = e2t.interpret_model(customized, artifact, x_df, target="nodes",
                          method="LayerConductance")
sites = e2t.interpret_model(customized, artifact, x_df, target="nodes",
                            level="sites", nodes="non_input",
                            method="LayerConductance")
```

---

## Common pitfalls (warn users / avoid when coding)

1. **`align_features_to_input_nodes` returns a `Tensor`, not a DataFrame.**
   Do not call `.values` on the result.

2. **`steps` is not for `feedforward`.** Use only with `recurrent` / `graphnn`.

3. **`steps` is not dataset time dimension.** One row = one static feature
   vector; `steps` = internal unrolling count per forward.

4. **Output must be reachable from informative inputs.** Example bug: graphnn
   notebook initially connected `readout` only via a noise input — training and
   node attribution failed silently (high loss, zero hidden attributions). Always
   verify paths from signal inputs to outputs.

5. **Recurrent example attribution:** `regulator_a` may get all importance and
   `regulator_b` zero when only the `regulator_a` branch carries label signal
   and only `regulator_a` connects to the output. Expected for the toy graph, not
   a package bug.

6. **Multi-output graphs:** node/feature attribution may need
   `attribute_kwargs={"target": 0}` (or appropriate index).

7. **`customize_model` head output dim** must match Captum expectations for the
   chosen `target` when interpreting.

8. **Pseudo nodes** exist only in feedforward skip-edge expansion; never
   interpret them directly.

9. **Sites with no nodes matching `nodes` filter are skipped** (not an error);
   empty overall attribution still errors.

10. **Interpretation requires `[interpret]` extra** (Captum). Import of Captum
    classes is lazy.

---

## Repository layout

```
src/edge2torch/
  compile_graph.py          # Public compile_graph
  align_features_to_input_nodes.py
  customize_model.py
  interpret_model.py
  compile/                  # Backend compilers, artifact, validation
  graph/                    # Edgelist I/O, schema, validation
  nn/                       # EdgeModel, blocks, step_block, masked_linear
  interpret/                # Captum adapter, site attribution, registry
  customize/                # customize_model validation
  utils/errors.py           # Edge2TorchError

tests/
  api/                      # Public API integration tests
  module/                   # Internal unit tests
  integration/              # Heavier integration tests
  fixtures/edgelists/       # CSV edgelists for architecture tests

docs/
  getting-started.ipynb     # Main tutorial (feedforward)
  recurrent-example.ipynb   # Cyclic recurrent backend
  graphnn-example.ipynb     # Cyclic graphnn backend
  feedforward-skip-edges.ipynb
  edge-weight-metadata.ipynb
  backends.md, api.md, ...

CONTEXT.md                  # This file (AI-oriented)
README.md                   # Human-oriented overview
CHANGELOG.md
pyproject.toml
```

---

## Dependencies

**Core:** `torch>=2.1`, `pandas>=2`, `numpy>=1.24`, Python `>=3.10`.

**Optional extras (`pyproject.toml`):**

| Extra | Packages |
|-------|----------|
| `interpret` | captum |
| `anndata` | anndata |
| `all` | captum + anndata |
| `dev` | pytest, ruff, mypy, captum, anndata, sklearn, ... |
| `docs` | mkdocs, mkdocs-material, mkdocs-jupyter, mike, ... |

Install for development: `pip install -e ".[dev,docs]"` or `".[all,dev,docs]"`.

---

## Testing and CI

- **423+ tests** (`pytest`), ruff lint/format, mypy on `src/`.
- CI (`.github/workflows/ci.yml`): matrix Python 3.10–3.12; docs job executes
  all five notebooks then `mkdocs build --strict`.
- Release tag `v*`: PyPI publish (`release.yml`), versioned docs via Mike
  (`docs.yml`).

Local docs check:

```bash
pip install -e ".[docs]"
# execute notebooks (see ci.yml) then:
mkdocs build --strict
```

---

## Documentation map (for humans and AI)

| Resource | Content |
|----------|---------|
| `README.md` | Overview, install, minimal example |
| `CONTEXT.md` | This file — dense AI context |
| `docs/getting-started.ipynb` | Full feedforward workflow |
| `docs/recurrent-example.ipynb` | Cyclic recurrent + node interpret |
| `docs/graphnn-example.ipynb` | Cyclic graphnn + node interpret |
| `docs/backends.md` | Backend semantics and interpretation |
| `docs/api.md` | API entry + mkdocstrings |
| `CHANGELOG.md` | Version history; **0.2.0 breaking node API** |

---

## Migration notes (0.1.0 → 0.2.0)

- `interpret_model(..., target="nodes")` now returns a **summary DataFrame** by
  default instead of a dict of per-layer tables.
- Per-site node tables: pass `level="sites"`.
- Recurrent/graphnn: state updates unrolled into `step_*` modules for
  interpretation; use `site_aggregation` on summary.

---

## Guidance for AI assistants

**When user asks to add a feature:**

- Prefer extending via **PyTorch training/wrappers** unless the request is
  truly about graph compilation, backends, or attribution plumbing.
- Do not add training loops, losses, or domain-specific biology inside the
  compiler unless explicitly requested.

**When user asks about recurrent / signaling / LEMBAS-style models:**

- Affirm structural alignment (sparse prior graph, cycles, static ligand-like
  inputs, iterative internal updates).
- Clarify that mechanistic activations, steady-state convergence, KO protocols,
  and spectral-radius losses are **user PyTorch**, not built-in.

**When user reports zero node attributions:**

- Check graph topology (path from informative inputs to outputs).
- Check training actually reduced loss.
- Check `nodes` filter and `level`.
- For recurrent/graphnn, inspect `level="sites"`.

**When editing notebooks for mkdocs:**

- Notebook outputs need valid nbformat (`metadata` on `execute_result`,
  `execution_count` on code cells) or `mkdocs build --strict` fails.

**When committing:** User prefers explicit commit requests; do not commit
proactively.

---

## License

MIT. See `LICENSE`. Citation metadata in `CITATION.cff`.
