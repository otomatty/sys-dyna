from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..engine import PySDEngine, SimulationError
from ..models import ModelRef, Scenario
from ..objective import Objective
from .distributions import ParamDistribution


# Optional per-parameter clamp (typically ParamSpec.clamp) applied to each draw
# so sampled values never leave a model's valid range.
Clamp = Callable[[str, float], float]


@dataclass
class VariableStats:
    """Summary statistics for a sampled scalar (the objective across runs)."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "percentiles": dict(self.percentiles),
        }


@dataclass
class MonteCarloResult:
    """The outcome of a Monte Carlo parameter sweep over one model."""

    model_id: str
    iterations: int
    objective: dict[str, str]
    stats: VariableStats
    # Pearson correlation between each sampled parameter and the objective value;
    # a rough one-at-a-time sensitivity ranking of which inputs drive the output.
    sensitivities: dict[str, float] = field(default_factory=dict)
    histogram: dict[str, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "montecarlo",
            "model_id": self.model_id,
            "iterations": self.iterations,
            "objective": dict(self.objective),
            "stats": self.stats.to_payload(),
            "sensitivities": dict(self.sensitivities),
            "histogram": dict(self.histogram),
            "warnings": list(self.warnings),
            "summary": self.summary_ja(),
        }

    def summary_ja(self) -> str:
        s = self.stats
        var = self.objective.get("variable", "目的変数")
        agg = self.objective.get("aggregate", "final")
        lines = [
            f"モンテカルロ分析（{self.iterations} 回試行）の結果、"
            f"{var} の{_AGG_LABEL.get(agg, agg)}は",
            f"平均 {s.mean:.2f}（標準偏差 {s.std:.2f}、"
            f"範囲 {s.min:.2f}〜{s.max:.2f}）でした。",
        ]
        p = s.percentiles
        if {"p5", "p50", "p95"} <= set(p):
            lines.append(
                f"90% 信頼区間はおよそ {p['p5']:.2f}〜{p['p95']:.2f}"
                f"（中央値 {p['p50']:.2f}）です。"
            )
        if self.sensitivities:
            top = max(self.sensitivities.items(), key=lambda kv: abs(kv[1]))
            lines.append(
                f"目的値への影響が最も大きい入力は「{top[0]}」"
                f"（相関 {top[1]:+.2f}）と推定されます。"
            )
        return "".join(lines)


_AGG_LABEL = {
    "final": "最終値",
    "initial": "初期値",
    "mean": "平均",
    "min": "最小値",
    "max": "最大値",
    "sum": "累積",
}


class MonteCarloAnalysis:
    """Runs many simulations with parameters drawn from distributions.

    Engine-agnostic: anything exposing ``run_scenarios(ref, scenarios,
    return_columns=...)`` works (the production ``PySDEngine`` or a test fake).
    """

    def __init__(
        self,
        engine: PySDEngine,
        *,
        default_iterations: int = 200,
        max_iterations: int = 2000,
    ) -> None:
        if default_iterations <= 0 or max_iterations <= 0:
            raise ValueError("iteration counts must be > 0")
        self._engine = engine
        self._default_iterations = default_iterations
        self._max_iterations = max_iterations

    def run(
        self,
        ref: ModelRef,
        base_params: dict[str, float],
        distributions: list[ParamDistribution],
        objective: Objective,
        *,
        iterations: int | None = None,
        seed: int | None = None,
        clamp: Clamp | None = None,
    ) -> MonteCarloResult:
        import numpy as np  # local: numpy arrives via the simulation stack

        if not distributions:
            raise SimulationError(
                "no_distributions",
                "Monte Carlo analysis needs at least one parameter distribution",
            )
        n = self._default_iterations if iterations is None else int(iterations)
        if n <= 0:
            raise SimulationError("invalid_iterations", "iterations must be > 0")
        warnings: list[str] = []
        if n > self._max_iterations:
            warnings.append(
                f"iterations capped from {n} to {self._max_iterations}"
            )
            n = self._max_iterations

        rng = np.random.default_rng(seed)
        scenarios: list[Scenario] = []
        # sampled[name] -> list of drawn values, aligned with the objective values.
        sampled: dict[str, list[float]] = {d.name: [] for d in distributions}
        for i in range(n):
            params = dict(base_params)
            for dist in distributions:
                value = dist.sample(rng)
                if clamp is not None:
                    value = clamp(dist.name, value)
                params[dist.name] = value
                sampled[dist.name].append(value)
            scenarios.append(Scenario(name=f"mc_{i}", params=params))

        run = self._engine.run_scenarios(
            ref, scenarios, return_columns=[objective.variable]
        )
        values = np.array([objective.scalar(s) for s in run.scenarios], dtype=float)

        stats = VariableStats(
            count=int(values.size),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            percentiles={
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
            },
        )
        sensitivities = _sensitivities(np, sampled, values)
        counts, edges = np.histogram(values, bins=min(20, max(1, values.size)))
        histogram = {
            "bin_edges": [float(x) for x in edges],
            "counts": [int(c) for c in counts],
        }
        return MonteCarloResult(
            model_id=ref.model_id,
            iterations=int(values.size),
            objective=objective.to_payload(),
            stats=stats,
            sensitivities=sensitivities,
            histogram=histogram,
            warnings=warnings,
        )


def _sensitivities(
    np: Any, sampled: dict[str, list[float]], values: Any
) -> dict[str, float]:
    """Pearson correlation of each sampled input with the objective value.

    Zero-variance inputs (e.g. a fixed distribution) correlate to 0.0 rather
    than NaN so the payload stays JSON-safe and the ranking is meaningful.
    """
    out: dict[str, float] = {}
    if values.size < 2 or float(np.std(values)) == 0.0:
        return {name: 0.0 for name in sampled}
    for name, drawn in sampled.items():
        column = np.array(drawn, dtype=float)
        if column.size != values.size or float(np.std(column)) == 0.0:
            out[name] = 0.0
            continue
        corr = float(np.corrcoef(column, values)[0, 1])
        out[name] = corr if corr == corr else 0.0  # NaN guard
    return out
