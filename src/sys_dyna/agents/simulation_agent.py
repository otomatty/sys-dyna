from __future__ import annotations

import logging
from typing import Any, Callable

from ..simulation import ModelSpec, PySDEngine, get_model
from ..simulation.analysis import BayesianOptimization, MonteCarloAnalysis
from ..tools.base import ToolDefinition, ToolError
from .tools import BayesianOptimizationTool, ModelLookup, MonteCarloTool


logger = logging.getLogger(__name__)


# Map the graph/planner intent labels onto the agent's tool names.
_INTENT_TO_TOOL = {
    "montecarlo": "monte_carlo",
    "monte_carlo": "monte_carlo",
    "optimize": "bayesian_optimization",
    "optimization": "bayesian_optimization",
    "bayesian_optimization": "bayesian_optimization",
}


class AnalysisError(Exception):
    """Raised when the agent cannot run the requested analysis."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SimulationAgent:
    """A self-contained sub-agent that runs advanced simulation analyses.

    It owns two tools — Monte Carlo analysis and Bayesian optimization — both
    backed by the shared ``PySDEngine``. It is usable standalone (call
    ``run(kind, arguments)``) and is wired into the LangGraph orchestration as
    the handler for the ``montecarlo`` / ``optimize`` intents.
    """

    def __init__(
        self,
        engine: PySDEngine | None = None,
        *,
        model_lookup: ModelLookup = get_model,
        default_iterations: int = 200,
        max_iterations: int = 2000,
        default_trials: int = 30,
        max_trials: int = 300,
    ) -> None:
        engine = engine or PySDEngine()
        self._monte_carlo = MonteCarloTool(
            MonteCarloAnalysis(
                engine,
                default_iterations=default_iterations,
                max_iterations=max_iterations,
            ),
            model_lookup,
        )
        self._bayesian = BayesianOptimizationTool(
            BayesianOptimization(
                engine, default_trials=default_trials, max_trials=max_trials
            ),
            model_lookup,
        )
        self.tools = {
            self._monte_carlo.definition.name: self._monte_carlo,
            self._bayesian.definition.name: self._bayesian,
        }

    def tool_definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self.tools.values()]

    def supports(self, kind: str) -> bool:
        return _INTENT_TO_TOOL.get(kind, kind) in self.tools

    def run(self, kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run the analysis named by ``kind`` and return its result payload.

        ``kind`` accepts either a tool name (``monte_carlo`` /
        ``bayesian_optimization``) or a graph intent (``montecarlo`` /
        ``optimize``). Raises :class:`AnalysisError` on any failure so callers
        get a structured, JSON-safe error instead of an opaque exception.
        """
        tool_name = _INTENT_TO_TOOL.get(kind, kind)
        tool = self.tools.get(tool_name)
        if tool is None:
            raise AnalysisError(
                "unknown_analysis",
                f"no analysis tool for '{kind}' (have: {sorted(self.tools)})",
            )
        try:
            result = tool.run(arguments)
        except ToolError as e:
            raise AnalysisError(e.code, e.message) from e
        return result.payload
