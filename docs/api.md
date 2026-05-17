# API reference

The main public workflow consists of four functions:

- `compile_graph()`
- `align_features_to_input_nodes()`
- `customize_model()`
- `interpret_model()`

`CompileArtifact` is also exported because it is returned by `compile_graph()`
and accepted by helper functions. Its stable user-facing fields are `backend`
and `feature_names`. Additional fields such as `graph`, `execution_plan`, and
`node_names_by_layer` expose compilation internals for inspection and debugging
and may change across releases.

::: edge2torch.compile_graph

::: edge2torch.CompileArtifact

::: edge2torch.align_features_to_input_nodes

::: edge2torch.customize_model

::: edge2torch.interpret_model