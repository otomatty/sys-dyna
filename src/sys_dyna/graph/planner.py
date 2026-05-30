from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..simulation.models import ModelSpec, Scenario


@runtime_checkable
class Planner(Protocol):
    """The LLM-driven decision surface used by the graph nodes.

    Concrete implementations:
    - ``GeminiPlanner`` (production, gemini-3.5-flash via langchain-google-genai)
    - a deterministic fake in tests

    Keeping every LLM call behind this interface lets the LangGraph wiring be
    exercised end-to-end without any external API.
    """

    def classify_intent(self, user_text: str, history: list[dict[str, Any]]) -> str:
        """Return one of: 'simulate', 'past_reference', 'general'."""
        ...

    def select_model(
        self,
        user_text: str,
        catalog: list[dict[str, str]],
        history: list[dict[str, Any]],
    ) -> str | None:
        """Pick a catalog model_id for the request, or None if none fits."""
        ...

    def extract_scenarios(
        self,
        user_text: str,
        model: ModelSpec,
        history: list[dict[str, Any]],
        base_params: dict[str, float] | None = None,
    ) -> list[Scenario]:
        """Turn the natural-language request into one or more parameter sets.

        Implementations start from ``base_params`` (a prior turn's values, when
        present) or ``model.default_params()`` and only override what the user
        asked to change, so follow-up edits ("then make churn 0.1") preserve the
        previous run's unchanged parameters.
        """
        ...

    def analyze(
        self,
        user_text: str,
        model: ModelSpec | None,
        simulation: dict[str, Any] | None,
        past_references: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> str:
        """Produce the final natural-language analysis for the user."""
        ...
