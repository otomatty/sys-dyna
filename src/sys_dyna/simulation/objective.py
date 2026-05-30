from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .engine import SimulationError
from .models import ScenarioResult


# How a variable's time series is reduced to the single scalar that Monte Carlo
# aggregates and Bayesian optimization optimises.
Aggregate = Literal["final", "initial", "mean", "min", "max", "sum"]
Direction = Literal["maximize", "minimize"]

_AGGREGATES: tuple[Aggregate, ...] = ("final", "initial", "mean", "min", "max", "sum")
_DIRECTIONS: tuple[Direction, ...] = ("maximize", "minimize")


@dataclass(frozen=True)
class Objective:
    """A configurable objective extracted from a simulation result.

    ``variable`` names the output variable to look at (e.g. ``"Sales"``),
    ``aggregate`` reduces its time series to one number, and ``direction``
    states whether larger or smaller is better. The same spec drives both the
    Monte Carlo summary (distribution of the objective scalar) and the Bayesian
    optimisation target (value to maximise/minimise).
    """

    variable: str
    aggregate: Aggregate = "final"
    direction: Direction = "maximize"

    def __post_init__(self) -> None:
        if not self.variable or not str(self.variable).strip():
            raise SimulationError("invalid_objective", "objective.variable is required")
        if self.aggregate not in _AGGREGATES:
            raise SimulationError(
                "invalid_objective",
                f"unknown aggregate '{self.aggregate}' (expected one of {_AGGREGATES})",
            )
        if self.direction not in _DIRECTIONS:
            raise SimulationError(
                "invalid_objective",
                f"unknown direction '{self.direction}' (expected one of {_DIRECTIONS})",
            )

    def scalar(self, result: ScenarioResult) -> float:
        """Reduce ``result``'s ``variable`` series to the objective scalar."""
        series = result.series(self.variable)
        if not series:
            raise SimulationError(
                "unknown_variable",
                f"simulation produced no series for objective variable '{self.variable}'",
            )
        values = [p["v"] for p in series]
        if self.aggregate == "final":
            return float(values[-1])
        if self.aggregate == "initial":
            return float(values[0])
        if self.aggregate == "mean":
            return float(sum(values) / len(values))
        if self.aggregate == "min":
            return float(min(values))
        if self.aggregate == "max":
            return float(max(values))
        # "sum"
        return float(sum(values))

    @property
    def sign(self) -> float:
        """+1 when maximising, -1 when minimising (handy for ranking)."""
        return 1.0 if self.direction == "maximize" else -1.0

    def to_payload(self) -> dict[str, str]:
        return {
            "variable": self.variable,
            "aggregate": self.aggregate,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, default_variable: str | None = None) -> "Objective":
        """Build from a loosely-typed dict (LLM / tool arguments).

        ``default_variable`` is used when the dict omits ``variable`` (e.g. the
        model's first output variable), so callers can stay terse.
        """
        data = data or {}
        variable = data.get("variable") or default_variable
        if not variable:
            raise SimulationError(
                "invalid_objective",
                "objective.variable is required (no default available)",
            )
        return cls(
            variable=str(variable),
            aggregate=data.get("aggregate", "final"),
            direction=data.get("direction", "maximize"),
        )
