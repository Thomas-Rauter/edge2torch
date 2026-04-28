
## Current interpretation support

The table below summarizes which interpretation targets and methods are
currently supported for each backend.

| Backend     | `target="features"` + `integrated_gradients` | `target="nodes"` + `layer_conductance` | `target="nodes"` + `layer_integrated_gradients` |
|-------------|----------------------------------------------|----------------------------------------|-------------------------------------------------|
| feedforward | yes                                          | yes                                    | yes                                             |
| recurrent   | yes                                          | no                                     | no                                              |
| graphnn     | yes                                          | no                                     | no                                              |
