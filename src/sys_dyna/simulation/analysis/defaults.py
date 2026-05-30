from __future__ import annotations

import re
from typing import Any, Literal

from ..models import ModelSpec


AnalysisKind = Literal["montecarlo", "optimize"]

# Cues that flip the optimisation/objective direction to minimisation.
_MINIMIZE_CUES = ("最小", "最小化", "減ら", "下げ", "削減", "minimize", "lower", "reduce")


def default_objective(model: ModelSpec, user_text: str = "") -> dict[str, str]:
    """A sensible objective for a model: maximise its first output's final value.

    Direction flips to ``minimize`` when the request mentions reducing/lowering,
    so "コストを最小化" optimises downward without the user spelling it out.
    """
    variable = model.output_variables[0] if model.output_variables else "Sales"
    direction = "minimize" if any(c in (user_text or "") for c in _MINIMIZE_CUES) else "maximize"
    return {"variable": variable, "aggregate": "final", "direction": direction}


def _spread(spec: Any, base: float) -> tuple[float, float]:
    """A reasonable [low, high] search/spread band for one parameter.

    Uses the parameter's declared min/max when available, otherwise ±50% of the
    base value (and a small positive band when the base is ~0).
    """
    low = spec.min if spec is not None and spec.min is not None else None
    high = spec.max if spec is not None and spec.max is not None else None
    if low is None:
        low = base * 0.5 if base > 0 else (base - 1.0)
    if high is None:
        high = base * 1.5 if base > 0 else (base + 1.0)
    if high <= low:
        high = low + max(abs(low), 1.0)
    return float(low), float(high)


def build_default_analysis_request(
    user_text: str,
    model: ModelSpec,
    kind: AnalysisKind,
    base_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Heuristic, LLM-free analysis spec — also the graph's fallback.

    Produces the same dict shape the Monte Carlo / Bayesian-optimization tools
    accept: ``model_id``, ``base_params``, ``objective``, and either
    ``distributions`` (Monte Carlo) or ``search_space`` (optimisation).
    """
    base = dict(model.default_params())
    if base_params:
        for key, value in base_params.items():
            if model.param(key) is not None:
                base[key] = float(value)

    objective = default_objective(model, user_text)
    request: dict[str, Any] = {
        "model_id": model.model_id,
        "base_params": base,
        "objective": objective,
    }

    # Drivers: parameters explicitly named in the request, else every parameter.
    named = [p for p in model.params if p.name in (user_text or "") or (p.label and p.label in (user_text or ""))]
    drivers = named or list(model.params)

    if kind == "montecarlo":
        distributions = []
        for p in drivers:
            value = base.get(p.name, p.default)
            std = abs(value) * 0.2 if value else 1.0  # ~20% uncertainty by default
            distributions.append({"name": p.name, "kind": "normal", "mean": value, "std": std})
        request["distributions"] = distributions
        iterations = _first_int(user_text)
        if iterations:
            request["iterations"] = iterations
    else:  # optimize
        search_space = []
        for p in drivers:
            value = base.get(p.name, p.default)
            low, high = _spread(p, value)
            search_space.append({"name": p.name, "low": low, "high": high})
        request["search_space"] = search_space
        trials = _first_int(user_text)
        if trials:
            request["n_trials"] = trials
    return request


def _first_int(text: str) -> int | None:
    """Pull an explicit run count like "1000回" / "50 trials" from the request."""
    if not text:
        return None
    m = re.search(r"([0-9]{2,6})\s*(?:回|trials?|回数|試行|samples?|iter)", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None
