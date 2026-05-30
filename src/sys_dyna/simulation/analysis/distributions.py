from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..engine import SimulationError


if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


# Sampling distributions for Monte Carlo parameter sweeps.
DistKind = Literal["normal", "uniform", "triangular", "lognormal", "fixed"]
_DIST_KINDS: tuple[DistKind, ...] = ("normal", "uniform", "triangular", "lognormal", "fixed")


@dataclass(frozen=True)
class ParamDistribution:
    """A probability distribution to draw one model parameter from.

    Only the fields relevant to ``kind`` are required:

    - ``normal`` / ``lognormal``: ``mean`` + ``std``
    - ``uniform``: ``low`` + ``high``
    - ``triangular``: ``low`` + ``high`` (+ optional ``mode``, defaults to midpoint)
    - ``fixed``: ``mean`` (a constant; useful to pin a parameter)
    """

    name: str
    kind: DistKind = "normal"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    mode: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise SimulationError("invalid_distribution", "distribution.name is required")
        if self.kind not in _DIST_KINDS:
            raise SimulationError(
                "invalid_distribution",
                f"unknown distribution kind '{self.kind}' (expected one of {_DIST_KINDS})",
            )
        if self.kind in ("normal", "lognormal"):
            self._require("mean")
            std = self._require("std")
            if std < 0:
                raise SimulationError("invalid_distribution", f"{self.name}: std must be >= 0")
        elif self.kind in ("uniform", "triangular"):
            lo = self._require("low")
            hi = self._require("high")
            if hi < lo:
                raise SimulationError(
                    "invalid_distribution", f"{self.name}: high ({hi}) must be >= low ({lo})"
                )
        elif self.kind == "fixed":
            if self.mean is None and self.low is None:
                raise SimulationError(
                    "invalid_distribution", f"{self.name}: fixed distribution needs a value"
                )

    def _require(self, field: str) -> float:
        value = getattr(self, field)
        if value is None:
            raise SimulationError(
                "invalid_distribution",
                f"{self.name}: '{field}' is required for a {self.kind} distribution",
            )
        return float(value)

    def sample(self, rng: "np.random.Generator") -> float:
        """Draw a single value from the distribution using ``rng``."""
        if self.kind == "fixed":
            value = self.mean if self.mean is not None else self.low
            return float(value)  # type: ignore[arg-type]
        if self.kind == "normal":
            return float(rng.normal(self._require("mean"), self._require("std")))
        if self.kind == "lognormal":
            return float(rng.lognormal(self._require("mean"), self._require("std")))
        if self.kind == "uniform":
            return float(rng.uniform(self._require("low"), self._require("high")))
        # triangular
        lo, hi = self._require("low"), self._require("high")
        mode = self.mode if self.mode is not None else (lo + hi) / 2.0
        mode = min(max(float(mode), lo), hi)
        return float(rng.triangular(lo, mode, hi))

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind}
        for field in ("low", "high", "mean", "std", "mode"):
            value = getattr(self, field)
            if value is not None:
                out[field] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParamDistribution":
        if not isinstance(data, dict):
            raise SimulationError("invalid_distribution", "distribution entry must be an object")
        return cls(
            name=str(data.get("name", "")),
            kind=data.get("kind", "normal"),
            low=_opt_float(data.get("low")),
            high=_opt_float(data.get("high")),
            mean=_opt_float(data.get("mean")),
            std=_opt_float(data.get("std")),
            mode=_opt_float(data.get("mode")),
        )


@dataclass(frozen=True)
class ParamRange:
    """A bounded search range for one parameter in Bayesian optimisation."""

    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise SimulationError("invalid_range", "range.name is required")
        if self.high <= self.low:
            raise SimulationError(
                "invalid_range", f"{self.name}: high ({self.high}) must be > low ({self.low})"
            )
        if self.log and self.low <= 0:
            raise SimulationError(
                "invalid_range", f"{self.name}: log scale requires low > 0 (got {self.low})"
            )

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "low": self.low, "high": self.high, "log": self.log}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParamRange":
        if not isinstance(data, dict):
            raise SimulationError("invalid_range", "range entry must be an object")
        try:
            low = float(data["low"])
            high = float(data["high"])
        except (KeyError, TypeError, ValueError) as e:
            raise SimulationError("invalid_range", f"range needs numeric low/high: {e}") from e
        return cls(
            name=str(data.get("name", "")),
            low=low,
            high=high,
            log=bool(data.get("log", False)),
        )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
