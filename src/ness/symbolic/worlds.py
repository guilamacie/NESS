"""A small typed relation world used by the reference interpreter and tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import content_hash


@dataclass(frozen=True, slots=True)
class Fact:
    relation: str
    subject: str
    object: str
    fact_id: str = ""
    available_at: int = 0
    source: str = ""


@dataclass(frozen=True, slots=True)
class RelationWorld:
    """Facts plus the set of relations for which this view is declared complete
    (closed world). No witness in an incomplete relation is *unknown*, not false."""

    facts: tuple[Fact, ...]
    closed_world_relations: frozenset[str] = frozenset()
    world_id: str = ""
    entity_types: dict[str, str] = field(default_factory=dict)

    def follow(self, entities: frozenset[str], relation: str) -> tuple[frozenset[str], bool]:
        """Return (targets, complete). ``inverse_<r>`` traverses r backwards."""
        inverse = relation.startswith("inverse_")
        rel = relation[len("inverse_"):] if inverse else relation
        out = set()
        for f in self.facts:
            if f.relation != rel:
                continue
            if inverse and f.object in entities:
                out.add(f.subject)
            elif not inverse and f.subject in entities:
                out.add(f.object)
        return frozenset(out), rel in self.closed_world_relations

    def witnesses(self, entities: frozenset[str], relation: str) -> tuple[Fact, ...]:
        inverse = relation.startswith("inverse_")
        rel = relation[len("inverse_"):] if inverse else relation
        return tuple(
            f for f in self.facts
            if f.relation == rel and ((inverse and f.object in entities) or (not inverse and f.subject in entities))
        )

    def canonical(self) -> dict:
        return {
            "facts": sorted([f.relation, f.subject, f.object] for f in self.facts),
            "closed": sorted(self.closed_world_relations),
        }

    @property
    def content_hash(self) -> str:
        return content_hash(self.canonical())
