from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..agents import AnalysisError, SimulationAgent
from ..simulation import PySDEngine, get_model
from ..simulation.analysis import build_default_analysis_request
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
    # The advanced-analysis sub-agent (Monte Carlo / Bayesian optimization). When
    # omitted it is built from ``engine`` so existing callers need no changes.
    agent: SimulationAgent | None = None


class _Nodes:
    def __init__(self, deps: GraphDeps) -> None:
        self.d = deps
        self._agent = deps.agent or SimulationAgent(
            deps.engine, model_lookup=deps.model_lookup
        )

    # -- routing -----------------------------------------------------------
    def classify_intent(self, state: AgentState) -> dict[str, Any]:
        intent = self.d.planner.classify_intent(
            state["user_text"], state.get("messages") or []
        )
        if intent not in ("simulate", "past_reference", "montecarlo", "optimize", "general"):
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

        model_id = self.d.planner.select_model(
            state["user_text"], catalog_summary(), state.get("messages") or []
        )
        if model_id is None or self.d.model_lookup(model_id) is None:
            return {"selected_model_id": None, "error": "no_matching_model"}
        return {"selected_model_id": model_id, "error": None}

    def _route_after_select(self, state: AgentState) -> str:
        if not state.get("selected_model_id"):
            return "analyze"
        # The advanced-analysis intents branch to the sub-agent path; a plain
        # simulate request goes through scenario extraction + HITL confirm.
        if state.get("intent") in ("montecarlo", "optimize"):
            return "prepare_analysis"
        return "extract_params"

    def extract_params(self, state: AgentState) -> dict[str, Any]:
        spec = self.d.model_lookup(state["selected_model_id"])
        scenarios = self.d.planner.extract_scenarios(
            state["user_text"],
            spec,
            state.get("messages") or [],
            state.get("base_params") or None,
        )
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
            Scenario(name=s["name"], params=self._clamp_params(spec, s["params"]))
            for s in state.get("scenarios", [])
        ]
        run = self.d.engine.run_scenarios(spec.ref, scenarios)
        return {"simulation": run.to_payload()}

    @staticmethod
    def _clamp_params(spec: Any, params: dict[str, Any]) -> dict[str, float]:
        # Authoritative bound enforcement: manual HITL edits (or a programmatic
        # resume) reach here unclamped, unlike planner output. Keep simulation
        # inputs within each ParamSpec's range so we never run an invalid model.
        clamped: dict[str, float] = {}
        for name, value in params.items():
            pspec = spec.param(name)
            clamped[name] = pspec.clamp(float(value)) if pspec else float(value)
        return clamped

    # -- advanced-analysis branch (Monte Carlo / Bayesian optimization) ----
    def prepare_analysis(self, state: AgentState) -> dict[str, Any]:
        """Build the analysis arguments, then HITL-confirm them like params.

        Uses the planner's ``build_analysis_request`` when available, falling
        back to the deterministic heuristic so planners that predate this method
        (and the offline path) still work.
        """
        spec = self.d.model_lookup(state["selected_model_id"])
        kind = state.get("intent") or "montecarlo"
        builder = getattr(self.d.planner, "build_analysis_request", None)
        if callable(builder):
            request = builder(
                state["user_text"],
                spec,
                kind,
                state.get("messages") or [],
                state.get("base_params") or None,
            )
        else:
            request = build_default_analysis_request(
                state["user_text"], spec, kind, state.get("base_params") or None
            )
        return {"analysis_kind": kind, "analysis_spec": request}

    def confirm_analysis(self, state: AgentState) -> dict[str, Any]:
        """HITL gate for the analysis spec (distributions / search space).

        Mirrors ``confirm_params``: ``Command(resume=...)`` may pass a dict with
        a ``spec`` key to override the proposal, or anything else to accept it.
        """
        decision = interrupt(
            {
                "type": "confirm_analysis",
                "analysis_kind": state.get("analysis_kind"),
                "model_id": state.get("selected_model_id"),
                "spec": state.get("analysis_spec", {}),
            }
        )
        spec = state.get("analysis_spec", {})
        if isinstance(decision, dict) and "spec" in decision:
            spec = decision["spec"]
        return {"analysis_spec": spec, "confirmed": True}

    def _route_after_confirm_analysis(self, state: AgentState) -> str:
        return "run_analysis" if state.get("analysis_spec") else "analyze"

    def run_analysis(self, state: AgentState) -> dict[str, Any]:
        kind = state.get("analysis_kind") or "montecarlo"
        spec = dict(state.get("analysis_spec") or {})
        spec.setdefault("model_id", state.get("selected_model_id"))
        try:
            payload = self._agent.run(kind, spec)
        except AnalysisError as e:
            logger.warning("analysis failed: %s (%s)", e.message, e.code)
            return {
                "simulation_analysis": {"error": e.code, "message": e.message},
                "analysis": f"解析を実行できませんでした（{e.message}）。",
                "error": e.code,
            }
        return {
            "simulation_analysis": payload,
            "analysis": payload.get("summary", "解析が完了しました。"),
        }

    def persist(self, state: AgentState) -> dict[str, Any]:
        try:
            self.d.persistence.save_run(state)
        except Exception:  # pragma: no cover - persistence must not break the turn
            logger.exception("save_run failed")
        return {}

    def _route_after_persist(self, state: AgentState) -> str:
        # The analysis branch fills ``analysis`` in run_analysis, so persist is
        # terminal there; the simulate branch still needs the analyze node.
        return "end" if state.get("analysis") else "analyze"

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
            state.get("messages") or [],
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
    g.add_node("prepare_analysis", nodes.prepare_analysis)
    g.add_node("confirm_analysis", nodes.confirm_analysis)
    g.add_node("run_analysis", nodes.run_analysis)
    g.add_node("persist", nodes.persist)
    g.add_node("analyze", nodes.analyze)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        nodes._route_intent,
        {
            "simulate": "select_model",
            "montecarlo": "select_model",
            "optimize": "select_model",
            "past_reference": "retrieve_past",
            "general": "analyze",
        },
    )
    g.add_edge("retrieve_past", "analyze")
    g.add_conditional_edges(
        "select_model",
        nodes._route_after_select,
        {
            "extract_params": "extract_params",
            "prepare_analysis": "prepare_analysis",
            "analyze": "analyze",
        },
    )
    g.add_edge("extract_params", "confirm_params")
    g.add_conditional_edges(
        "confirm_params",
        nodes._route_after_confirm,
        {"run_simulation": "run_simulation", "analyze": "analyze"},
    )
    g.add_edge("run_simulation", "persist")
    # Advanced-analysis branch: prepare -> HITL confirm -> run -> persist.
    g.add_edge("prepare_analysis", "confirm_analysis")
    g.add_conditional_edges(
        "confirm_analysis",
        nodes._route_after_confirm_analysis,
        {"run_analysis": "run_analysis", "analyze": "analyze"},
    )
    g.add_edge("run_analysis", "persist")
    g.add_conditional_edges(
        "persist",
        nodes._route_after_persist,
        {"analyze": "analyze", "end": END},
    )
    g.add_edge("analyze", END)

    return g.compile(checkpointer=checkpointer)
