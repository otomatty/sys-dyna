from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..simulation import PySDEngine, get_model
from ..simulation.models import Scenario
from .planner import Planner
from .state import AgentState


logger = logging.getLogger(__name__)


class Persistence(Protocol):
    """Sink for completed runs (Supabase repository in production)."""

    def save_run(self, state: AgentState) -> None: ...


class _NoOpPersistence:
    def save_run(self, state: AgentState) -> None:  # pragma: no cover - trivial
        return None


# A past-session lookup returns reference dicts for the analysis context.
PastLookup = Callable[[str], list[dict[str, Any]]]


def _no_past(_user_text: str) -> list[dict[str, Any]]:
    return []


@dataclass
class GraphDeps:
    planner: Planner
    engine: PySDEngine
    persistence: Persistence = _NoOpPersistence()
    past_lookup: PastLookup = _no_past
    model_lookup: Callable[[str], Any] = get_model


class _Nodes:
    def __init__(self, deps: GraphDeps) -> None:
        self.d = deps

    # -- routing -----------------------------------------------------------
    def classify_intent(self, state: AgentState) -> dict[str, Any]:
        intent = self.d.planner.classify_intent(
            state["user_text"], state.get("past_references") or []
        )
        if intent not in ("simulate", "past_reference", "general"):
            intent = "general"
        return {"intent": intent}

    def _route_intent(self, state: AgentState) -> str:
        return state.get("intent", "general")

    # -- past reference ----------------------------------------------------
    def retrieve_past(self, state: AgentState) -> dict[str, Any]:
        refs = self.d.past_lookup(state["user_text"])
        return {"past_references": refs}

    # -- simulation branch -------------------------------------------------
    def select_model(self, state: AgentState) -> dict[str, Any]:
        from ..simulation import catalog_summary

        model_id = self.d.planner.select_model(state["user_text"], catalog_summary())
        if model_id is None or self.d.model_lookup(model_id) is None:
            return {"selected_model_id": None, "error": "no_matching_model"}
        return {"selected_model_id": model_id, "error": None}

    def _route_after_select(self, state: AgentState) -> str:
        return "extract_params" if state.get("selected_model_id") else "analyze"

    def extract_params(self, state: AgentState) -> dict[str, Any]:
        spec = self.d.model_lookup(state["selected_model_id"])
        scenarios = self.d.planner.extract_scenarios(state["user_text"], spec)
        if not scenarios:
            scenarios = [Scenario(name="base", params=spec.default_params())]
        return {
            "scenarios": [
                {"name": s.name, "params": dict(s.params)} for s in scenarios
            ]
        }

    def confirm_params(self, state: AgentState) -> dict[str, Any]:
        """HITL gate. Pauses for the user to confirm/adjust parameters.

        ``interrupt`` surfaces the proposed scenarios to the caller; the value
        passed to ``Command(resume=...)`` comes back as ``decision``. A dict
        with a ``scenarios`` key overrides the proposal; anything else (e.g.
        ``"approve"``) accepts it as-is.
        """
        decision = interrupt(
            {
                "type": "confirm_params",
                "model_id": state.get("selected_model_id"),
                "scenarios": state.get("scenarios", []),
            }
        )
        scenarios = state.get("scenarios", [])
        # ``"scenarios" in decision`` (not a truthiness check) so an explicit
        # empty list — the cancellation signal — propagates instead of silently
        # falling back to the proposed scenarios.
        if isinstance(decision, dict) and "scenarios" in decision:
            scenarios = decision["scenarios"]
        return {"scenarios": scenarios, "confirmed": True}

    def _route_after_confirm(self, state: AgentState) -> str:
        # Cancellation (empty scenarios) skips simulation; PySDEngine requires
        # at least one scenario and would otherwise raise.
        return "run_simulation" if state.get("scenarios") else "analyze"

    def run_simulation(self, state: AgentState) -> dict[str, Any]:
        spec = self.d.model_lookup(state["selected_model_id"])
        scenarios = [
            Scenario(name=s["name"], params=dict(s["params"]))
            for s in state.get("scenarios", [])
        ]
        run = self.d.engine.run_scenarios(spec.ref, scenarios)
        return {"simulation": run.to_payload()}

    def persist(self, state: AgentState) -> dict[str, Any]:
        try:
            self.d.persistence.save_run(state)
        except Exception:  # pragma: no cover - persistence must not break the turn
            logger.exception("save_run failed")
        return {}

    def analyze(self, state: AgentState) -> dict[str, Any]:
        spec = (
            self.d.model_lookup(state["selected_model_id"])
            if state.get("selected_model_id")
            else None
        )
        text = self.d.planner.analyze(
            state["user_text"],
            spec,
            state.get("simulation"),
            state.get("past_references") or [],
        )
        return {"analysis": text}


def build_graph(deps: GraphDeps, checkpointer: Any | None = None) -> Any:
    """Construct and compile the orchestration StateGraph.

    A ``checkpointer`` is required for the HITL interrupt to resume across
    turns (MemorySaver in tests, PostgresSaver/Supabase in production).
    """
    nodes = _Nodes(deps)
    g: StateGraph = StateGraph(AgentState)

    g.add_node("classify_intent", nodes.classify_intent)
    g.add_node("retrieve_past", nodes.retrieve_past)
    g.add_node("select_model", nodes.select_model)
    g.add_node("extract_params", nodes.extract_params)
    g.add_node("confirm_params", nodes.confirm_params)
    g.add_node("run_simulation", nodes.run_simulation)
    g.add_node("persist", nodes.persist)
    g.add_node("analyze", nodes.analyze)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        nodes._route_intent,
        {
            "simulate": "select_model",
            "past_reference": "retrieve_past",
            "general": "analyze",
        },
    )
    g.add_edge("retrieve_past", "analyze")
    g.add_conditional_edges(
        "select_model",
        nodes._route_after_select,
        {"extract_params": "extract_params", "analyze": "analyze"},
    )
    g.add_edge("extract_params", "confirm_params")
    g.add_conditional_edges(
        "confirm_params",
        nodes._route_after_confirm,
        {"run_simulation": "run_simulation", "analyze": "analyze"},
    )
    g.add_edge("run_simulation", "persist")
    g.add_edge("persist", "analyze")
    g.add_edge("analyze", END)

    return g.compile(checkpointer=checkpointer)
