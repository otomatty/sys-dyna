from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langgraph.types import Command

from ..simulation import PySDEngine
from .builder import GraphDeps, Persistence, PastLookup, build_graph
from .planner import Planner


TurnStatus = Literal["awaiting_confirmation", "completed"]


@dataclass
class TurnOutcome:
    """Result of a (possibly paused) graph turn, consumed by the UI."""

    status: TurnStatus
    session_id: str
    intent: str | None = None
    # Populated when status == "awaiting_confirmation": the confirm_params payload.
    confirm: dict[str, Any] | None = None
    # Populated when status == "completed".
    analysis: str | None = None
    simulation: dict[str, Any] | None = None
    selected_model_id: str | None = None


class TurnRunner:
    """Thin façade over the compiled LangGraph for the Streamlit app.

    ``thread_id`` is the session id, so the checkpointer can pause at the HITL
    interrupt and resume the same conversation later.
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def start(
        self,
        session_id: str,
        user_text: str,
        user_id: str = "",
        history: list[dict[str, Any]] | None = None,
        base_params: dict[str, float] | None = None,
    ) -> TurnOutcome:
        result = self._graph.invoke(
            {
                "user_text": user_text,
                "session_id": session_id,
                "user_id": user_id,
                "messages": list(history or []),
                # A prior turn's params (from the last simulation) so follow-up
                # edits build on them instead of reverting to defaults.
                "base_params": dict(base_params or {}),
                # The checkpointer reuses one thread per session and merges this
                # input into the existing state, so explicitly reset per-turn
                # fields — otherwise a prior turn's simulation/model leaks into a
                # later general or past-reference question.
                "intent": None,
                "selected_model_id": None,
                "scenarios": [],
                "confirmed": False,
                "simulation": None,
                "past_references": [],
                "analysis": None,
                "error": None,
            },
            self._config(session_id),
        )
        return self._to_outcome(session_id, result)

    def resume(self, session_id: str, decision: Any) -> TurnOutcome:
        result = self._graph.invoke(
            Command(resume=decision), self._config(session_id)
        )
        return self._to_outcome(session_id, result)

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    @staticmethod
    def _to_outcome(session_id: str, result: dict[str, Any]) -> TurnOutcome:
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            return TurnOutcome(
                status="awaiting_confirmation",
                session_id=session_id,
                intent=result.get("intent"),
                confirm=payload,
                selected_model_id=payload.get("model_id"),
            )
        return TurnOutcome(
            status="completed",
            session_id=session_id,
            intent=result.get("intent"),
            analysis=result.get("analysis"),
            simulation=result.get("simulation"),
            selected_model_id=result.get("selected_model_id"),
        )


def build_planner(
    gemini_api_key: str | None,
    gemini_model: str = "gemini-3.5-flash",
    temperature: float = 0.2,
    max_scenarios: int = 5,
) -> Planner:
    """Pick the production Gemini planner when a key is set, else the offline one."""
    if gemini_api_key:
        from .gemini_planner import GeminiPlanner

        return GeminiPlanner(
            model=gemini_model,
            api_key=gemini_api_key,
            temperature=temperature,
            max_scenarios=max_scenarios,
        )
    from .heuristic_planner import HeuristicPlanner

    return HeuristicPlanner()


def build_runner(
    planner: Planner,
    checkpointer: Any,
    engine: PySDEngine | None = None,
    persistence: Persistence | None = None,
    past_lookup: PastLookup | None = None,
) -> TurnRunner:
    deps = GraphDeps(
        planner=planner,
        engine=engine or PySDEngine(),
        **({"persistence": persistence} if persistence else {}),
        **({"past_lookup": past_lookup} if past_lookup else {}),
    )
    graph = build_graph(deps, checkpointer=checkpointer)
    return TurnRunner(graph)
