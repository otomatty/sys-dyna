from __future__ import annotations

import logging
import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .models import ModelRef, Scenario, ScenarioResult, SimulationRun


logger = logging.getLogger(__name__)

# A cached PySD model is a single stateful instance; ``model.run`` mutates it
# and is not safe to call concurrently. Engines are shared across sessions via
# Streamlit's @st.cache_resource, so serialise runs with a process-wide lock.
_MODEL_RUN_LOCK = threading.Lock()

# Bookkeeping columns PySD always emits; never returned as model variables.
_PYSD_INTERNAL_COLUMNS = {
    "INITIAL TIME",
    "FINAL TIME",
    "TIME STEP",
    "SAVEPER",
    "TIME",
}


class SimulationError(Exception):
    """Raised when a model cannot be loaded or a run fails."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _load_pysd_model(path: str) -> Any:
    """Load a PySD model from a Vensim ``.mdl`` or XMILE ``.xmile`` file.

    Imported lazily so the rest of the package (and its tests) does not pay the
    PySD import cost unless a simulation is actually run.
    """
    import pysd  # local import: heavy optional dependency

    p = Path(path)
    if not p.exists():
        raise SimulationError("model_not_found", f"model file not found: {path}")
    suffix = p.suffix.lower()
    try:
        # Pre-compiled Python models load directly (pysd.load) — no translation
        # step, so they work on read-only filesystems where read_xmile/
        # read_vensim would fail trying to write a .py next to the source.
        if suffix == ".py":
            return pysd.load(str(p))
        if suffix == ".xmile" or suffix == ".xml":
            return pysd.read_xmile(str(p))
        if suffix == ".mdl":
            return pysd.read_vensim(str(p))
    except Exception as e:  # pragma: no cover - depends on external parser
        raise SimulationError("model_load_failed", f"could not load model: {e}") from e
    raise SimulationError(
        "unsupported_model_format",
        f"unsupported model format '{suffix}' (expected .py, .xmile, or .mdl)",
    )


@lru_cache(maxsize=32)
def _cached_model(path: str, mtime: float) -> Any:
    """Cache parsed models by (path, mtime) so repeated runs skip re-parsing.

    ``mtime`` is part of the key so an edited catalog file is re-read.
    """
    return _load_pysd_model(path)


class PySDEngine:
    """Loads SD models with PySD and runs scenarios, returning normalised output."""

    def __init__(self, default_return_columns: Iterable[str] | None = None) -> None:
        self._default_return_columns = (
            list(default_return_columns) if default_return_columns else None
        )

    def run_scenarios(
        self,
        ref: ModelRef,
        scenarios: list[Scenario],
        return_columns: Iterable[str] | None = None,
    ) -> SimulationRun:
        if not scenarios:
            raise SimulationError("no_scenarios", "at least one scenario is required")

        try:
            mtime = Path(ref.path).stat().st_mtime
        except OSError as e:
            raise SimulationError("model_not_found", f"model file not found: {ref.path}") from e

        model = _cached_model(ref.path, mtime)
        cols = list(return_columns) if return_columns else self._default_return_columns

        run = SimulationRun(model_id=ref.model_id)
        for scenario in scenarios:
            result = self._run_one(model, scenario, cols)
            run.scenarios.append(result)
        return run

    def _run_one(
        self,
        model: Any,
        scenario: Scenario,
        return_columns: list[str] | None,
    ) -> ScenarioResult:
        params = dict(scenario.params)
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = params
        if return_columns:
            kwargs["return_columns"] = return_columns

        try:
            # ``reload`` resets stocks to their initial values so scenarios run
            # independently even though they share one cached model object.
            # The lock guards that shared, mutable instance against concurrent
            # runs from other sessions.
            with _MODEL_RUN_LOCK:
                frame = model.run(reload=True, **kwargs)
        except KeyError as e:
            raise SimulationError(
                "unknown_parameter",
                f"model has no such variable/parameter: {e}",
            ) from e
        except Exception as e:
            raise SimulationError("run_failed", f"simulation failed: {e}") from e

        variables = _frame_to_series(frame)
        return ScenarioResult(
            scenario=scenario.name,
            params=params,
            variables=variables,
        )


def _frame_to_series(frame: Any) -> dict[str, list[dict[str, float]]]:
    """Convert a PySD result DataFrame to ``{var: [{"t","v"}, ...]}``.

    Non-finite values (NaN/inf) are dropped so the payload is JSON-safe.
    """
    out: dict[str, list[dict[str, float]]] = {}
    index = list(frame.index)
    for column in frame.columns:
        if column in _PYSD_INTERNAL_COLUMNS:
            continue
        series: list[dict[str, float]] = []
        values = frame[column]
        for t, v in zip(index, values):
            fv = float(v)
            if not math.isfinite(fv):
                continue
            series.append({"t": float(t), "v": fv})
        out[column] = series
    return out
