"""Source selectors: ``scheme://node/port`` URIs validated against producer kinds."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import CompositionError, SCHEME_FOR_KIND

SCHEMES = ("observation", "substrate", "module", "semantic", "program", "reasoner", "memory")
OBSERVATION_NODE = "observation"


@dataclass(frozen=True, slots=True, order=True)
class SourceSelector:
    scheme: str
    node_id: str
    port: str

    @classmethod
    def parse(cls, text: str) -> "SourceSelector":
        if not isinstance(text, str) or "://" not in text:
            raise CompositionError(f"malformed source selector {text!r}; expected scheme://node/port")
        scheme, rest = text.split("://", 1)
        if scheme not in SCHEMES:
            raise CompositionError(f"unknown selector scheme {scheme!r} in {text!r}; allowed {SCHEMES}")
        parts = [p for p in rest.split("/") if p]
        if scheme == "observation":
            if len(parts) != 1:
                raise CompositionError(f"observation selector must be observation://<field>: {text!r}")
            return cls(scheme, OBSERVATION_NODE, parts[0])
        if len(parts) < 2:
            raise CompositionError(f"selector {text!r} needs a node and a port")
        return cls(scheme, parts[0], "/".join(parts[1:]))

    def __str__(self) -> str:
        if self.scheme == "observation":
            return f"observation://{self.port}"
        return f"{self.scheme}://{self.node_id}/{self.port}"

    def canonical(self) -> str:
        return str(self)

    def expects_kind(self, module_kind: str) -> bool:
        """A selector's scheme must agree with the producer's declared module kind."""
        if self.scheme == "observation":
            return module_kind == "observation"
        return SCHEME_FOR_KIND.get(module_kind) == self.scheme
