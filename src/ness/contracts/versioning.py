"""Schema and plugin version identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import IncompatibleVersion

_SCHEMA_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_.]*)/(?P<major>\d+)$")
_SEMVER_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """A schema identifier such as ``ness.experiment/3``. Meanings are immutable."""

    name: str
    major: int

    @classmethod
    def parse(cls, text: str) -> "SchemaVersion":
        m = _SCHEMA_RE.match(text)
        if not m:
            raise IncompatibleVersion(f"malformed schema version {text!r}; expected 'name/major'")
        return cls(m.group("name"), int(m.group("major")))

    def __str__(self) -> str:
        return f"{self.name}/{self.major}"


def require_schema(text: str, expected: str) -> SchemaVersion:
    got = SchemaVersion.parse(text)
    want = SchemaVersion.parse(expected)
    if got != want:
        raise IncompatibleVersion(f"schema {got} is not the expected {want}")
    return got


@dataclass(frozen=True, slots=True, order=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> "SemVer":
        m = _SEMVER_RE.match(text)
        if not m:
            raise IncompatibleVersion(f"malformed semantic version {text!r}")
        return cls(int(m["major"]), int(m["minor"]), int(m["patch"]))

    def compatible_with(self, other: "SemVer") -> bool:
        """Same major version means state produced by ``other`` is restorable here."""
        return self.major == other.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
