"""Versioned data contracts. Imports nothing heavier than numpy."""

from .availability import PREDICTION_TIME_ROLES, AvailabilityCut, FieldRole, ProducerRef, Provenance
from .budget import CostLedger, ExecutionBudget, QueryBudget
from .capabilities import (
    BackendCapabilities,
    Capability,
    CapabilitySet,
    CapabilityStatus,
    MemoryCapabilities,
    ProbabilisticReasonerCapabilities,
    ScenarioCapabilities,
    SubstrateCapabilities,
)
from .coordinates import CoordinateSchema, horizon_schema, quantile_schema, series_schema
from .errors import (
    AccessPolicyViolation,
    BudgetExceeded,
    CausalityViolation,
    CheckpointError,
    CompositionError,
    ContractViolation,
    GradientBoundaryError,
    IncompatibleVersion,
    MemoryConflict,
    NessError,
    PluginDependencyMissing,
    PluginNotFound,
    UnsupportedCapability,
    ValidationError,
)
from .evidence import (
    ApproximationStatus,
    BindingEvidence,
    Completeness,
    ConstraintEvidence,
    Evidence,
    EvidenceFragment,
    EvidenceGraph,
    ExecutionTrace,
    FeatureEvidence,
    HypothesisEvidence,
    Knownness,
    PredictiveEvidence,
    QueryStatus,
    RetrievalEvidence,
    TruthStatus,
)
from .forecasts import (
    CategoricalForecast,
    Forecast,
    ForecastCapabilities,
    PointForecast,
    PredictionSpace,
    QuantileForecast,
    SampleForecast,
)
from .hashing import array_hash, canonical_json, content_hash, short_hash
from .manifest import ComponentRef, PredictorManifest
from .observations import AccessPolicy, Observation, ObservationBundle, ObservationField
from .ports import (
    MODULE_KINDS,
    SCHEME_FOR_KIND,
    Differentiability,
    ModuleDescriptor,
    ParameterGroupSpec,
    PortSchema,
    PortSpec,
)
from .requests import PredictionRequest, PredictionResult, ScoreResult, TaskQuery, TrainingOutcome
from .state import ModuleState, StateSnapshot, flatten_params, unflatten_params
from .versioning import SchemaVersion, SemVer, require_schema

__all__ = [n for n in dir() if not n.startswith("_")]
