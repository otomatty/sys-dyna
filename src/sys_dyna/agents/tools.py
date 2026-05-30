from __future__ import annotations

from typing import Any, Callable

from ..simulation import ModelSpec, get_model
from ..simulation.analysis import (
    BayesianOptimization,
    MonteCarloAnalysis,
    ParamDistribution,
    ParamRange,
)
from ..simulation.engine import SimulationError
from ..simulation.objective import Objective
from ..tools.base import Tool, ToolDefinition, ToolError, ToolResult


ModelLookup = Callable[[str], "ModelSpec | None"]


def _resolve_model(model_lookup: ModelLookup, arguments: dict[str, Any]) -> ModelSpec:
    model_id = arguments.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ToolError("invalid_argument", "model_id must be a non-empty string")
    spec = model_lookup(model_id)
    if spec is None:
        raise ToolError("not_found", f"unknown model_id: {model_id}")
    return spec


def _resolve_base_params(spec: ModelSpec, arguments: dict[str, Any]) -> dict[str, float]:
    """Model defaults overlaid with caller-supplied base_params (clamped)."""
    base = spec.default_params()
    provided = arguments.get("base_params") or {}
    if not isinstance(provided, dict):
        raise ToolError("invalid_argument", "base_params must be an object")
    for name, value in provided.items():
        pspec = spec.param(name)
        if pspec is None:
            continue
        try:
            base[name] = pspec.clamp(float(value))
        except (TypeError, ValueError):
            continue
    return base


def _make_clamp(spec: ModelSpec) -> Callable[[str, float], float]:
    def _clamp(name: str, value: float) -> float:
        pspec = spec.param(name)
        return pspec.clamp(value) if pspec is not None else value

    return _clamp


def _parse_objective(spec: ModelSpec, arguments: dict[str, Any]) -> Objective:
    default_variable = spec.output_variables[0] if spec.output_variables else None
    try:
        return Objective.from_dict(
            arguments.get("objective"), default_variable=default_variable
        )
    except SimulationError as e:
        raise ToolError(e.code, e.message) from e


def _opt_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ToolError("invalid_argument", f"{key} must be an integer") from e


def _opt_seed(arguments: dict[str, Any]) -> int | None:
    return _opt_int(arguments, "seed")


_OBJECTIVE_SCHEMA = {
    "type": "object",
    "description": "What to measure: a model output variable reduced to one number.",
    "properties": {
        "variable": {"type": "string", "description": "Output variable name (e.g. 'Sales')."},
        "aggregate": {
            "type": "string",
            "enum": ["final", "initial", "mean", "min", "max", "sum"],
            "description": "How to reduce the time series. Default 'final'.",
        },
        "direction": {
            "type": "string",
            "enum": ["maximize", "minimize"],
            "description": "Optimisation direction / what 'better' means. Default 'maximize'.",
        },
    },
}


class MonteCarloTool(Tool):
    """Monte Carlo simulation: sample parameters from distributions, run many
    simulations, and summarise the resulting distribution of the objective."""

    definition = ToolDefinition(
        name="monte_carlo",
        description=(
            "Run a Monte Carlo analysis of a catalog model: draw each listed "
            "parameter from a probability distribution, simulate many times, and "
            "report the mean/percentiles of the objective plus per-input "
            "sensitivities. Use for uncertainty / risk analysis."
        ),
        parameters={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Catalog model id."},
                "base_params": {
                    "type": "object",
                    "description": "Fixed parameter overrides (defaults used otherwise).",
                },
                "distributions": {
                    "type": "array",
                    "description": "Parameters to randomise and how.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": ["normal", "uniform", "triangular", "lognormal", "fixed"],
                            },
                            "low": {"type": "number"},
                            "high": {"type": "number"},
                            "mean": {"type": "number"},
                            "std": {"type": "number"},
                            "mode": {"type": "number"},
                        },
                        "required": ["name", "kind"],
                    },
                },
                "objective": _OBJECTIVE_SCHEMA,
                "iterations": {"type": "integer", "description": "Number of samples."},
                "seed": {"type": "integer", "description": "RNG seed for reproducibility."},
            },
            "required": ["model_id", "distributions"],
        },
    )

    def __init__(self, analysis: MonteCarloAnalysis, model_lookup: ModelLookup = get_model) -> None:
        self._analysis = analysis
        self._model_lookup = model_lookup

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        spec = _resolve_model(self._model_lookup, arguments)
        base = _resolve_base_params(spec, arguments)
        objective = _parse_objective(spec, arguments)

        raw = arguments.get("distributions")
        if not isinstance(raw, list) or not raw:
            raise ToolError("invalid_argument", "distributions must be a non-empty array")
        try:
            distributions = [ParamDistribution.from_dict(d) for d in raw]
            result = self._analysis.run(
                spec.ref,
                base,
                distributions,
                objective,
                iterations=_opt_int(arguments, "iterations"),
                seed=_opt_seed(arguments),
                clamp=_make_clamp(spec),
            )
        except SimulationError as e:
            raise ToolError(e.code, e.message) from e
        return ToolResult(payload=result.to_payload())


class BayesianOptimizationTool(Tool):
    """Bayesian optimisation (Optuna/TPE): search parameter ranges to maximise
    or minimise the objective with few simulations."""

    definition = ToolDefinition(
        name="bayesian_optimization",
        description=(
            "Run Bayesian optimization over a catalog model's parameters to find "
            "the values that maximise (or minimise) the objective. Provide a "
            "search range per tunable parameter. Use for 'what is the best ...' "
            "questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Catalog model id."},
                "base_params": {
                    "type": "object",
                    "description": "Fixed parameter overrides for non-searched params.",
                },
                "search_space": {
                    "type": "array",
                    "description": "Parameters to optimise and their bounds.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "low": {"type": "number"},
                            "high": {"type": "number"},
                            "log": {"type": "boolean", "description": "Search on a log scale."},
                        },
                        "required": ["name", "low", "high"],
                    },
                },
                "objective": _OBJECTIVE_SCHEMA,
                "n_trials": {"type": "integer", "description": "Number of optimisation trials."},
                "seed": {"type": "integer", "description": "Sampler seed for reproducibility."},
            },
            "required": ["model_id", "search_space"],
        },
    )

    def __init__(self, analysis: BayesianOptimization, model_lookup: ModelLookup = get_model) -> None:
        self._analysis = analysis
        self._model_lookup = model_lookup

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        spec = _resolve_model(self._model_lookup, arguments)
        base = _resolve_base_params(spec, arguments)
        objective = _parse_objective(spec, arguments)

        raw = arguments.get("search_space")
        if not isinstance(raw, list) or not raw:
            raise ToolError("invalid_argument", "search_space must be a non-empty array")
        try:
            search_space = [ParamRange.from_dict(r) for r in raw]
            result = self._analysis.optimize(
                spec.ref,
                base,
                search_space,
                objective,
                n_trials=_opt_int(arguments, "n_trials"),
                seed=_opt_seed(arguments),
                clamp=_make_clamp(spec),
            )
        except SimulationError as e:
            raise ToolError(e.code, e.message) from e
        return ToolResult(payload=result.to_payload())
