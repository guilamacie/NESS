# ADR-0001: One execution verb (`forward`) for every macro module

**Status** accepted. **Context** The PDF sketches family-specific methods (`Substrate.read`,
`SemanticAdapter.propose`, `ProbabilisticReasoner.infer`). The addendum asks for a generic
composition graph where any declared port can feed any permitted later port.
**Decision** All nodes implement `MacroModule.forward(inputs, state, ctx) -> ModuleOutputs`;
family distinctions live in `module_kind`, descriptors, capabilities and typed evidence.
Reasoners keep `compile/infer` as their scientific contract and wrap it in `forward`.
**Consequences** The executor and compiler are family-agnostic (T17). Scientific distinctions
are enforced through kinds/schemes/evidence types rather than method names.
**Revisit** if a family needs a fundamentally different execution protocol (e.g. streaming).
