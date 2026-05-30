from __future__ import annotations

from dataclasses import dataclass

import pytest

from sys_dyna.graph.gemini_planner import (
    GeminiPlanner,
    parse_intent,
    parse_model_id,
    parse_scenarios,
)
from sys_dyna.simulation import get_model


SPEC = get_model("sales_growth")
assert SPEC is not None


def test_parse_intent_variants() -> None:
    assert parse_intent("simulate") == "simulate"
    assert parse_intent("  Past_Reference  ") == "past_reference"
    assert parse_intent("これは general です") == "general"
    assert parse_intent("???") == "general"  # unknown -> safe default


def test_parse_model_id() -> None:
    valid = {"sales_growth", "inventory"}
    assert parse_model_id("sales_growth", valid) == "sales_growth"
    assert parse_model_id("model is sales_growth.", valid) == "sales_growth"
    assert parse_model_id("none", valid) is None
    assert parse_model_id("unknown_model", valid) is None


def test_parse_scenarios_overrides_and_fills_defaults() -> None:
    raw = '{"scenarios": [{"name": "x1.5", "params": {"ad_spend": 150}}]}'
    scenarios = parse_scenarios(raw, SPEC)
    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.name == "x1.5"
    assert s.params["ad_spend"] == 150.0
    # Untouched params fall back to defaults.
    assert s.params["conversion"] == 0.5
    assert s.params["churn_rate"] == 0.05


def test_parse_scenarios_drops_unknown_and_clamps() -> None:
    raw = (
        '{"scenarios": [{"name": "c", "params": '
        '{"ad_spend": 150, "bogus": 9, "churn_rate": 5.0}}]}'
    )
    s = parse_scenarios(raw, SPEC)[0]
    assert "bogus" not in s.params
    # churn_rate has max=1.0 -> clamped.
    assert s.params["churn_rate"] == 1.0


def test_parse_scenarios_handles_fenced_json() -> None:
    raw = "```json\n{\"scenarios\": [{\"name\": \"a\", \"params\": {}}]}\n```"
    s = parse_scenarios(raw, SPEC)
    assert s[0].name == "a"
    assert s[0].params == SPEC.default_params()


def test_parse_scenarios_malformed_falls_back_to_default() -> None:
    s = parse_scenarios("not json at all", SPEC)
    assert len(s) == 1
    assert s[0].params == SPEC.default_params()


def test_parse_scenarios_respects_max() -> None:
    items = ",".join(
        f'{{"name":"s{i}","params":{{}}}}' for i in range(10)
    )
    raw = f'{{"scenarios": [{items}]}}'
    s = parse_scenarios(raw, SPEC, max_scenarios=3)
    assert len(s) == 3


@dataclass
class _Reply:
    content: str


class _ScriptedChat:
    """Minimal stand-in for ChatGoogleGenerativeAI: returns queued replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> _Reply:
        self.prompts.append(prompt)
        return _Reply(content=self._replies.pop(0))


def test_planner_with_injected_chat_model() -> None:
    chat = _ScriptedChat(
        [
            "simulate",
            "sales_growth",
            '{"scenarios": [{"name": "x2", "params": {"ad_spend": 200}}]}',
            "売上は増加傾向です。",
        ]
    )
    planner = GeminiPlanner(chat_model=chat)

    assert planner.classify_intent("広告を増やしたら?", []) == "simulate"
    assert planner.select_model(
        "...", [{"model_id": "sales_growth", "name": "", "description": ""}], []
    ) == "sales_growth"
    scenarios = planner.extract_scenarios("広告2倍", SPEC, [])
    assert scenarios[0].params["ad_spend"] == 200.0
    text = planner.analyze("?", SPEC, {"scenarios": []}, [], [])
    assert "売上" in text
