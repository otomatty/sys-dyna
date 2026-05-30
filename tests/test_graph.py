from __future__ import annotations

from typing import Any

import pytest

from sys_dyna.graph import GraphDeps, build_graph
from sys_dyna.simulation import PySDEngine, get_model
from sys_dyna.simulation.models import ModelSpec, Scenario


pytest.importorskip("pysd")
pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402


class FakePlanner:
    """Deterministic stand-in for GeminiPlanner used to drive the graph."""

    def __init__(self, intent: str = "simulate") -> None:
        self._intent = intent
        self.analyze_calls: list[dict[str, Any]] = []

    def classify_intent(self, user_text: str, history: list[dict[str, Any]]) -> str:
        return self._intent

    def select_model(self, user_text: str, catalog: list[dict[str, str]]) -> str | None:
        return "sales_growth"

    def extract_scenarios(self, user_text: str, model: ModelSpec) -> list[Scenario]:
        base = model.default_params()
        return [Scenario(name="ad_x1.5", params={**base, "ad_spend": 150.0})]

    def analyze(self, user_text, model, simulation, past_references) -> str:
        self.analyze_calls.append(
            {"simulation": simulation, "past": past_references}
        )
        if simulation is None:
            return "no simulation"
        finals = []
        for sc in simulation["scenarios"]:
            series = sc["variables"]["Sales"]
            finals.append((sc["scenario"], series[-1]["v"]))
        return "analysis: " + ", ".join(f"{n}={v:.0f}" for n, v in finals)


def _deps(planner: FakePlanner, **kw: Any) -> GraphDeps:
    return GraphDeps(planner=planner, engine=PySDEngine(), **kw)


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": "t-1"}}


def test_simulate_flow_pauses_at_confirm_then_resumes() -> None:
    planner = FakePlanner(intent="simulate")
    graph = build_graph(_deps(planner), checkpointer=MemorySaver())
    cfg = _config()

    out = graph.invoke(
        {"user_text": "広告費を1.5倍にしたら売上は?", "session_id": "s", "user_id": "u"},
        cfg,
    )
    # The graph must be interrupted at the HITL gate, not finished.
    assert "__interrupt__" in out
    interrupt_payload = out["__interrupt__"][0].value
    assert interrupt_payload["type"] == "confirm_params"
    assert interrupt_payload["model_id"] == "sales_growth"
    assert interrupt_payload["scenarios"][0]["params"]["ad_spend"] == 150.0

    # Resume with approval -> simulation runs and analysis is produced.
    final = graph.invoke(Command(resume="approve"), cfg)
    assert final["confirmed"] is True
    assert final["simulation"] is not None
    assert final["analysis"].startswith("analysis:")


def test_human_can_override_parameters_at_confirm() -> None:
    planner = FakePlanner(intent="simulate")
    graph = build_graph(_deps(planner), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-override"}}

    graph.invoke({"user_text": "売上は?", "session_id": "s", "user_id": "u"}, cfg)
    override = {"scenarios": [{"name": "manual", "params": {"ad_spend": 300.0}}]}
    final = graph.invoke(Command(resume=override), cfg)

    sim = final["simulation"]
    assert sim["scenarios"][0]["scenario"] == "manual"
    assert sim["scenarios"][0]["params"]["ad_spend"] == 300.0


def test_general_intent_skips_simulation() -> None:
    planner = FakePlanner(intent="general")
    graph = build_graph(_deps(planner), checkpointer=MemorySaver())
    out = graph.invoke(
        {"user_text": "こんにちは", "session_id": "s", "user_id": "u"},
        {"configurable": {"thread_id": "t-general"}},
    )
    assert "__interrupt__" not in out
    assert out.get("simulation") is None
    assert out["analysis"] == "no simulation"


def test_past_reference_intent_runs_lookup_then_analyze() -> None:
    planner = FakePlanner(intent="past_reference")
    seen: dict[str, Any] = {}

    def past_lookup(text: str) -> list[dict[str, Any]]:
        seen["text"] = text
        return [{"session_id": "prev-1", "summary": "似た分析"}]

    graph = build_graph(
        _deps(planner, past_lookup=past_lookup), checkpointer=MemorySaver()
    )
    out = graph.invoke(
        {"user_text": "過去に似た分析あった?", "session_id": "s", "user_id": "u"},
        {"configurable": {"thread_id": "t-past"}},
    )
    assert seen["text"] == "過去に似た分析あった?"
    assert out["past_references"][0]["session_id"] == "prev-1"
    assert planner.analyze_calls[-1]["past"][0]["session_id"] == "prev-1"


def test_persistence_is_invoked_on_simulate() -> None:
    planner = FakePlanner(intent="simulate")
    saved: list[Any] = []

    class Rec:
        def save_run(self, state: Any) -> None:
            saved.append(state.get("simulation"))

    graph = build_graph(
        _deps(planner, persistence=Rec()), checkpointer=MemorySaver()
    )
    cfg = {"configurable": {"thread_id": "t-persist"}}
    graph.invoke({"user_text": "売上は?", "session_id": "s", "user_id": "u"}, cfg)
    graph.invoke(Command(resume="approve"), cfg)
    assert saved and saved[0] is not None
