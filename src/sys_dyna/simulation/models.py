from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ModelSource = Literal["catalog", "upload"]


@dataclass(frozen=True)
class ParamSpec:
    """A single tunable parameter exposed by a model.

    ``name`` must match the variable name PySD exposes for the model
    (whitespace in XMILE/Vensim names is normalised to underscores by PySD).
    """

    name: str
    label: str
    default: float
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    description: str | None = None


@dataclass(frozen=True)
class ModelRef:
    """Identifies which model a run targets and where it came from."""

    model_id: str
    source: ModelSource
    # For catalog models this is the packaged file path; for uploads it is the
    # Supabase Storage object path resolved to a local temp file at run time.
    path: str


@dataclass(frozen=True)
class ModelSpec:
    """Catalog metadata surfaced to the LLM and the parameter-confirm UI."""

    model_id: str
    name: str
    description: str
    ref: ModelRef
    params: tuple[ParamSpec, ...]
    output_variables: tuple[str, ...] = ()

    def default_params(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}

    def param(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


@dataclass(frozen=True)
class Scenario:
    """One parameter set to simulate. Multiple scenarios enable comparison."""

    name: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TimePoint:
    t: float
    v: float


@dataclass
class ScenarioResult:
    """Normalised result for a single scenario.

    ``variables`` maps each returned variable to its time series as a list of
    ``{"t": ..., "v": ...}`` dicts — the same shape v1.0 stored in
    ``simulation_results`` so the persistence layer stays compatible.
    """

    scenario: str
    params: dict[str, float]
    variables: dict[str, list[dict[str, float]]]

    def series(self, variable: str) -> list[dict[str, float]]:
        return self.variables.get(variable, [])


@dataclass
class SimulationRun:
    """The full result of running one or more scenarios against a model."""

    model_id: str
    scenarios: list[ScenarioResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "scenarios": [
                {
                    "scenario": s.scenario,
                    "params": s.params,
                    "variables": s.variables,
                }
                for s in self.scenarios
            ],
            "warnings": list(self.warnings),
        }
