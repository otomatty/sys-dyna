from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ..engine import PySDEngine, SimulationError
from ..models import ModelRef, Scenario
from ..objective import Objective
from .distributions import ParamRange


logger = logging.getLogger(__name__)

Clamp = Callable[[str, float], float]


@dataclass
class OptimizationResult:
    """The outcome of a Bayesian optimisation over a model's parameters."""

    model_id: str
    n_trials: int
    objective: dict[str, str]
    best_params: dict[str, float]
    best_value: float
    # Each completed trial as {"trial", "params", "value"} in run order, so the
    # UI can plot convergence and the analysis can reason about the search.
    history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "optimize",
            "model_id": self.model_id,
            "n_trials": self.n_trials,
            "objective": dict(self.objective),
            "best_params": dict(self.best_params),
            "best_value": self.best_value,
            "history": list(self.history),
            "warnings": list(self.warnings),
            "summary": self.summary_ja(),
        }

    def summary_ja(self) -> str:
        var = self.objective.get("variable", "目的変数")
        direction = "最大化" if self.objective.get("direction") == "maximize" else "最小化"
        params = "、".join(f"{k}={v:.4g}" for k, v in self.best_params.items())
        return (
            f"ベイズ最適化（{self.n_trials} 試行）の結果、{var} を{direction}する"
            f"最適なパラメータは {params} で、そのときの目的値は {self.best_value:.2f} でした。"
        )


class BayesianOptimization:
    """Optimises model parameters against an objective using Optuna (TPE).

    Optuna is imported lazily (like PySD) so importing this module never
    requires the dependency; a clear ``SimulationError`` is raised only when an
    optimisation is actually attempted without it installed.
    """

    def __init__(
        self,
        engine: PySDEngine,
        *,
        default_trials: int = 30,
        max_trials: int = 300,
    ) -> None:
        if default_trials <= 0 or max_trials <= 0:
            raise ValueError("trial counts must be > 0")
        self._engine = engine
        self._default_trials = default_trials
        self._max_trials = max_trials

    def optimize(
        self,
        ref: ModelRef,
        base_params: dict[str, float],
        search_space: list[ParamRange],
        objective: Objective,
        *,
        n_trials: int | None = None,
        seed: int | None = None,
        clamp: Clamp | None = None,
    ) -> OptimizationResult:
        try:
            import optuna
        except ImportError as e:  # pragma: no cover - depends on environment
            raise SimulationError(
                "optuna_unavailable",
                "Bayesian optimization requires the 'optuna' package "
                "(pip install optuna or sys-dyna[analysis])",
            ) from e

        if not search_space:
            raise SimulationError(
                "no_search_space",
                "Bayesian optimization needs at least one parameter range",
            )
        n = self._default_trials if n_trials is None else int(n_trials)
        if n <= 0:
            raise SimulationError("invalid_trials", "n_trials must be > 0")
        warnings: list[str] = []
        if n > self._max_trials:
            warnings.append(f"n_trials capped from {n} to {self._max_trials}")
            n = self._max_trials

        # Quiet Optuna's per-trial INFO logging; the caller surfaces results.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=objective.direction, sampler=sampler)

        # Clamp each search range to the model's valid bounds *up front* so
        # Optuna only ever proposes runnable values. Clamping a suggestion after
        # the fact (the previous approach) recorded a value Optuna never actually
        # evaluated: study.best_params / history would disagree with best_value,
        # and the TPE surrogate would be fed contradictory (input, output) pairs.
        bounds: list[tuple[str, float, float, bool]] = []
        for rng in search_space:
            low = clamp(rng.name, rng.low) if clamp is not None else rng.low
            high = clamp(rng.name, rng.high) if clamp is not None else rng.high
            if high <= low:
                high = low + 1e-9
            # A log scale needs a strictly positive lower bound; if clamping
            # pushed it to <= 0, fall back to a linear scale for that parameter.
            log = bool(rng.log and low > 0)
            bounds.append((rng.name, low, high, log))

        history: list[dict[str, Any]] = []

        def _trial_objective(trial: Any) -> float:
            params = dict(base_params)
            suggested: dict[str, float] = {}
            for name, low, high, log in bounds:
                value = trial.suggest_float(name, low, high, log=log)
                suggested[name] = value
                params[name] = value
            run = self._engine.run_scenarios(
                ref,
                [Scenario(name=f"trial_{trial.number}", params=params)],
                return_columns=[objective.variable],
            )
            value = objective.scalar(run.scenarios[0])
            history.append(
                {"trial": trial.number, "params": suggested, "value": value}
            )
            return value

        study.optimize(_trial_objective, n_trials=n)

        return OptimizationResult(
            model_id=ref.model_id,
            n_trials=len(study.trials),
            objective=objective.to_payload(),
            best_params={k: float(v) for k, v in study.best_params.items()},
            best_value=float(study.best_value),
            history=history,
            warnings=warnings,
        )
