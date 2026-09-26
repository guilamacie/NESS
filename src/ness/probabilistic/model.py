"""Finite-latent model specification shared by the reference reasoners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import ContractViolation, content_hash


@dataclass(frozen=True, slots=True)
class FiniteLatentModel:
    """Discrete latent with K states, categorical prior, diagonal-Gaussian emission
    over an F-dimensional feature vector. Missing features are marginalised out
    (dropped from the likelihood) under an explicit ``missing_policy``."""

    latent_name: str
    states: tuple[str, ...]
    prior: np.ndarray        # [K]
    means: np.ndarray        # [K, F]
    scales: np.ndarray       # [K, F]
    missing_policy: str = "marginalize"

    def __post_init__(self) -> None:
        K = len(self.states)
        p = np.asarray(self.prior, dtype=np.float64)
        m = np.asarray(self.means, dtype=np.float64)
        s = np.asarray(self.scales, dtype=np.float64)
        if K < 2:
            raise ContractViolation("a finite latent needs at least two states")
        if p.shape != (K,) or not np.isclose(p.sum(), 1.0) or np.any(p <= 0):
            raise ContractViolation("prior must be a strictly positive distribution over states")
        if m.ndim != 2 or m.shape[0] != K or s.shape != m.shape or np.any(s <= 0):
            raise ContractViolation("means [K,F] and positive scales [K,F] required")
        if self.missing_policy not in ("marginalize", "fail"):
            raise ContractViolation("missing_policy must be marginalize|fail")
        object.__setattr__(self, "prior", p)
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "scales", s)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def feature_dim(self) -> int:
        return int(self.means.shape[1])

    def log_likelihood(self, x: np.ndarray, known: np.ndarray) -> np.ndarray:
        """log p(x | z=k) for each k, using only known feature dimensions. Returns [K]."""
        x = np.asarray(x, dtype=np.float64)
        known = np.asarray(known, dtype=bool)
        if x.shape != (self.feature_dim,):
            raise ContractViolation(f"feature vector shape {x.shape} != ({self.feature_dim},)")
        if not known.any():
            if self.missing_policy == "fail":
                raise ContractViolation("all features missing")
            return np.zeros(self.n_states)
        if (~known).any() and self.missing_policy == "fail":
            raise ContractViolation("missing features with missing_policy=fail")
        xm, mu, sd = x[known], self.means[:, known], self.scales[:, known]
        z = (xm[None, :] - mu) / sd
        return np.sum(-0.5 * z**2 - np.log(sd) - 0.5 * np.log(2 * np.pi), axis=1)

    def canonical(self) -> dict:
        return {"latent": self.latent_name, "states": list(self.states), "prior": self.prior.tolist(),
                "means": self.means.tolist(), "scales": self.scales.tolist(), "missing_policy": self.missing_policy}

    @property
    def model_hash(self) -> str:
        return content_hash(self.canonical())

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FiniteLatentModel":
        states = tuple(cfg.get("states", ("regime_0", "regime_1")))
        K = len(states)
        prior = cfg.get("prior", [1.0 / K] * K)
        em = cfg.get("emission", {})
        if em.get("family", "gaussian_diag") != "gaussian_diag":
            raise ContractViolation(f"unsupported emission family {em.get('family')!r}")
        return cls(cfg.get("latent_name", "regime"), states, np.asarray(prior), np.asarray(em["means"]), np.asarray(em["scales"]),
                   cfg.get("missing_policy", "marginalize"))
