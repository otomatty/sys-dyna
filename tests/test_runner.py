from __future__ import annotations

import pytest

from sys_dyna.graph import build_planner, build_runner
from sys_dyna.graph.heuristic_planner import HeuristicPlanner
from sys_dyna.simulation import get_model


pytest.importorskip("pysd")
pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402


SPEC = get_model("sales_growth")


def _runner(planner=None):
    return build_runner(
        planner or HeuristicPlanner(), checkpointer=MemorySaver()
    )


def test_build_planner_selects_offline_without_key() -> None:
    assert isinstance(build_planner(None), HeuristicPlanner)


def test_build_planner_selects_gemini_with_key() -> None:
    p = build_planner("fake-key")
    assert p.__class__.__name__ == "GeminiPlanner"


def test_start_pauses_for_confirmation_on_simulate() -> None:
    runner = _runner()
    out = runner.start("sess-1", "広告費を1.5倍にしたら売上はどうなる?")
    assert out.status == "awaiting_confirmation"
    assert out.confirm is not None
    scenarios = out.confirm["scenarios"]
    # "1.5倍" -> ad_spend scaled 100 -> 150.
    assert scenarios[0]["params"]["ad_spend"] == 150.0


def test_resume_completes_with_analysis_and_simulation() -> None:
    runner = _runner()
    runner.start("sess-2", "広告費を2倍にしたら売上は?")
    out = runner.resume("sess-2", "approve")
    assert out.status == "completed"
    assert out.simulation is not None
    assert out.analysis and "シミュレーション結果" in out.analysis


def test_general_intent_completes_without_pause() -> None:
    runner = _runner()
    out = runner.start("sess-3", "こんにちは")
    assert out.status == "completed"
    assert out.simulation is None


def test_resume_with_edited_parameters() -> None:
    runner = _runner()
    runner.start("sess-4", "広告費を1.5倍にしたら?")
    edited = {"scenarios": [{"name": "manual", "params": {"ad_spend": 250.0}}]}
    out = runner.resume("sess-4", edited)
    assert out.simulation["scenarios"][0]["params"]["ad_spend"] == 250.0


def test_cancel_resumes_cleanly_without_simulation() -> None:
    """Empty scenarios (cancellation) must complete, not run or hang the thread."""
    runner = _runner()
    runner.start("sess-cancel", "広告費を1.5倍にしたら?")
    out = runner.resume("sess-cancel", {"scenarios": []})
    assert out.status == "completed"
    assert out.simulation is None
    # The thread is no longer interrupted: a fresh turn on the same session works.
    again = runner.start("sess-cancel", "こんにちは")
    assert again.status == "completed"


def test_heuristic_multiple_multipliers_make_scenarios() -> None:
    planner = HeuristicPlanner()
    scenarios = planner.extract_scenarios("広告費を1.2倍と1.5倍で比較", SPEC, [])
    names = [s.params["ad_spend"] for s in scenarios]
    assert 120.0 in names and 150.0 in names


def test_heuristic_followup_param_override_uses_history() -> None:
    planner = HeuristicPlanner()
    history = [
        {"role": "user", "content": "広告費を1.5倍にしたら売上は?"},
        {"role": "assistant", "content": "シミュレーション結果（Sales）..."},
    ]
    # A bare numeric tweak after a prior exchange is treated as simulate.
    assert planner.classify_intent("churn_rate を 0.1 に", history) == "simulate"
    scenarios = planner.extract_scenarios("churn_rate を 0.1 に", SPEC, history)
    assert scenarios[0].params["churn_rate"] == 0.1
    # Untouched params keep defaults.
    assert scenarios[0].params["ad_spend"] == 100.0


def test_runner_threads_history_into_turn() -> None:
    runner = _runner()
    history = [
        {"role": "user", "content": "広告費を1.5倍にしたら?"},
        {"role": "assistant", "content": "売上は増加します。"},
    ]
    out = runner.start("sess-hist", "解約率を0.1にしたら?", history=history)
    # Follow-up with history -> classified as simulate -> awaits confirmation.
    assert out.status == "awaiting_confirmation"
    assert out.confirm["scenarios"][0]["params"]["churn_rate"] == 0.1


def test_heuristic_classify_intent() -> None:
    p = HeuristicPlanner()
    assert p.classify_intent("広告を増やしたらどうなる?", []) == "simulate"
    assert p.classify_intent("過去に似た事例ある?", []) == "past_reference"
    assert p.classify_intent("ありがとう", []) == "general"
