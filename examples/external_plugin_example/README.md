# ness-example-plugin

A complete third-party plugin package for NESS. It adds, **without editing any file under
`ness/`**:

| plugin id | kind | what it proves |
|---|---|---|
| `toy_linear_ar_substrate` | substrate | a third lower-provider family behind the same ports (T31/T33) |
| `deterministic_threshold_reasoner` | reasoner | reasoner substitution with honest `deterministic_assignment` semantics (T37) |
| `bounded_point_writer` | writer | changing the cap writer by configuration |
| `heavy_dependency_substrate` | substrate | a declared-but-absent dependency fails closed and never poisons unrelated configs (T41) |
| `window_max` (entry point `ness.primitives`) | program primitive | a typed-program operator added by a third-party package; programs in any config may `call` it |

Install with `pip install -e examples/external_plugin_example`, then
`ness plugins` lists them and `ness run examples/external_plugin_example/configs/external_substitution.yaml`
runs an arm built from them. Tests use the contributor test kit (`ness.plugin_api.testkit`).
