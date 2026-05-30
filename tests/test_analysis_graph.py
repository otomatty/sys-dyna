from __future__ import annotations

from typing import Any, Callable

import pytest

from sys_dyna.agents import SimulationAgent
from sys_dyna.graph import GraphDeps, build_graph
from sys_dyna.simulation.analysis import build_default_analysis_request
from sys_dyna.simulation.models import (
    ModelRef,
    ModelSpec,
    ParamSpec,
    ScenarioResult,
    SimulationRun,
)


pytest.importorskip("numpy")
pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402


class FakeEngine:
    def __init__(self, fn: Callable[[dict], float]) -> None:
        self.fn = fn

    def run_scenarios(self, ref, scenarios, return_columns=None):  # type: ignore[no-untyped-def]
        run = SimulationRun(model_id=ref.model_id)
        for sc in scenarios:
            v = self.fn(sc.params)
            run.scenarios.append(
                ScenarioResult(
                    scenario=sc.name,
                    params=dict(sc.params),
                    variables={"Sales": [{"t": 0.0, "v": 0.0}, {"t": 1.0, "v": v}]},
                )
            )
        return run


_SPEC = ModelSpec(
    model_id="sales_growth",
    name="fake",
    description="d",
    ref=ModelRef(model_id="sales_growth", source="catalog", path="/dev/null"),
    params=(ParamSpec(name="ad_spend", label="広告費", default=100.0, min=0.0),),
    output_variables=("Sales",),
)


def _lookup(model_id: str) -> ModelSpec | None:
    return _SPEC if model_id == "sales_growth" else None


class FakePlanner:
    def __init__(self, intent: str, with_builder: bool = True) -> None:
        self._intent = intent
        if not with_builder:
            # Simulate an older planner so the graph must fall back to the
            # deterministic default analysis-request builder.
            self.build_analysis_request = None  # type: ignore[assignment]

    def classify_intent(self, user_text: str, history: list[dict[str, Any]]) -> str:
        return self._intent

    def select_model(self, user_text, catalog, history) -> str | None:
        return "sales_growth"

    def extract_scenarios(self, user_text, model, history, base_params=None):  # pragma: no cover
        return []

    def build_analysis_request(self, user_text, model, kind, history, base_params=None):
        return build_default_analysis_request(user_text, model, kind, base_params)

    def analyze(self, user_text, model, simulation, past_references, history) -> str:  # pragma: no cover
        return "analyze"


def _deps(intent: str, fn: Callable[[dict], float], with_builder: bool = True) -> GraphDeps:
    engine = FakeEngine(fn)
    return GraphDeps(
        planner=FakePlanner(intent, with_builder),
        engine=engine,
        model_lookup=_lookup,
        agent=SimulationAgent(engine, model_lookup=_lookup),
    )


def test_montecarlo_flow_confirms_then_runs() -> None:
    graph = build_graph(_deps("montecarlo", lambda p: p["ad_spend"]), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "mc"}}

    out = graph.invoke(
        {"user_text": "広告費のばらつきリスクを見たい", "session_id": "s", "user_id": "u"}, cfg
    )
    # Pauses at the analysis-specific HITL gate.
    assert "__interrupt__" in out
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "confirm_analysis"
    assert payload["analysis_kind"] == "montecarlo"
    assert payload["spec"]["model_id"] == "sales_growth"
    assert payload["spec"]["distributions"]

    final = graph.invoke(Command(resume="approve"), cfg)
    assert final["confirmed"] is True
    assert final["simulation_analysis"]["kind"] == "montecarlo"
    assert "モンテカルロ" in final["analysis"]


def test_optimize_flow_runs_and_skips_analyze_node() -> None:
    pytest.importorskip("optuna")
    graph = build_graph(
        _deps("optimize", lambda p: -((p["ad_spend"] - 80.0) ** 2)), checkpointer=MemorySaver()
    )
    cfg = {"configurable": {"thread_id": "opt"}}
    out = graph.invoke({"user_text": "広告費を最適化して", "session_id": "s", "user_id": "u"}, cfg)
    # Pin the TPE sampler seed (the default analysis spec omits it) so the
    # best-params assertion below is deterministic and not CI-flaky.
    spec = dict(out["__interrupt__"][0].value["spec"])
    spec["seed"] = 0
    spec["n_trials"] = 40
    final = graph.invoke(Command(resume={"spec": spec}), cfg)

    assert final["simulation_analysis"]["kind"] == "optimize"
    assert final["simulation_analysis"]["best_params"]["ad_spend"] == pytest.approx(80.0, abs=15.0)
    assert "ベイズ最適化" in final["analysis"]


def test_user_can_override_analysis_spec_at_confirm() -> None:
    graph = build_graph(_deps("montecarlo", lambda p: p["ad_spend"]), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "mc-override"}}
    graph.invoke({"user_text": "ばらつきを見たい", "session_id": "s", "user_id": "u"}, cfg)

    override = {
        "spec": {
            "model_id": "sales_growth",
            "distributions": [{"name": "ad_spend", "kind": "fixed", "mean": 200.0}],
            "objective": {"variable": "Sales"},
            "iterations": 50,
            "seed": 1,
        }
    }
    final = graph.invoke(Command(resume=override), cfg)
    stats = final["simulation_analysis"]["stats"]
    # Every sample pinned to 200 -> mean 200, zero spread.
    assert stats["mean"] == pytest.approx(200.0)
    assert stats["std"] == pytest.approx(0.0)


def test_fallback_when_planner_lacks_builder() -> None:
    graph = build_graph(
        _deps("montecarlo", lambda p: p["ad_spend"], with_builder=False),
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "mc-fallback"}}
    out = graph.invoke({"user_text": "ばらつき", "session_id": "s", "user_id": "u"}, cfg)
    payload = out["__interrupt__"][0].value
    assert payload["spec"]["distributions"]  # default builder produced a spec
    final = graph.invoke(Command(resume="approve"), cfg)
    assert final["simulation_analysis"]["kind"] == "montecarlo"
