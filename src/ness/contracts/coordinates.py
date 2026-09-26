"""Coordinate schemas: axes are named, never inferred from array rank."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ContractViolation


@dataclass(frozen=True, slots=True)
class CoordinateSchema:
    """Names the axes and units of a payload.

    ``axes`` are logical axis names in storage order (e.g. ``("time", "channel")``).
    Compatibility is decided by schema id equality or a declared conversion; equal
    array shapes never imply compatible coordinates.
    """

    schema_id: str
    axes: tuple[str, ...]
    units: str | None = None
    cadence: str | None = None
    channel_names: tuple[str, ...] | None = None
    horizons: tuple[int, ...] | None = None
    quantile_levels: tuple[float, ...] | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ContractViolation("CoordinateSchema requires a schema_id")
        if self.quantile_levels is not None:
            lv = self.quantile_levels
            if any(not (0.0 < q < 1.0) for q in lv) or list(lv) != sorted(lv) or len(set(lv)) != len(lv):
                raise ContractViolation(f"quantile levels must be strictly increasing in (0,1): {lv}")

    def compatible_with(self, other: "CoordinateSchema") -> bool:
        return self.schema_id == other.schema_id and self.axes == other.axes

    def require_compatible(self, other: "CoordinateSchema", where: str = "") -> None:
        if not self.compatible_with(other):
            raise ContractViolation(
                f"incompatible coordinates {self.schema_id}{self.axes} vs {other.schema_id}{other.axes} {where}".strip()
            )


def series_schema(channel_names: tuple[str, ...], cadence: str = "step", units: str | None = None) -> CoordinateSchema:
    return CoordinateSchema(
        schema_id=f"series[{','.join(channel_names)}]@{cadence}",
        axes=("time", "channel"),
        units=units,
        cadence=cadence,
        channel_names=channel_names,
    )


def horizon_schema(channel_names: tuple[str, ...], horizons: tuple[int, ...], cadence: str = "step") -> CoordinateSchema:
    return CoordinateSchema(
        schema_id=f"horizon[{','.join(channel_names)}]x{len(horizons)}@{cadence}",
        axes=("horizon", "channel"),
        cadence=cadence,
        channel_names=channel_names,
        horizons=horizons,
    )


def quantile_schema(
    channel_names: tuple[str, ...], horizons: tuple[int, ...], levels: tuple[float, ...], cadence: str = "step"
) -> CoordinateSchema:
    return CoordinateSchema(
        schema_id=f"quantile[{','.join(channel_names)}]x{len(horizons)}x{len(levels)}@{cadence}",
        axes=("horizon", "channel", "quantile"),
        cadence=cadence,
        channel_names=channel_names,
        horizons=horizons,
        quantile_levels=levels,
    )
