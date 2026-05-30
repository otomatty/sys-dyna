from __future__ import annotations

import pytest


pytest.importorskip("pysd")
pytest.importorskip("langgraph")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


_SIM = {
    "model_id": "sales_growth",
    "warnings": [],
    "scenarios": [
        {
            "scenario": "base",
            "params": {},
            "variables": {"Sales": [{"t": 0.0, "v": 1000.0}, {"t": 1.0, "v": 1010.0}]},
        }
    ],
}


def test_app_starts_and_renders_title() -> None:
    at = AppTest.from_file("app.py", default_timeout=40)
    at.run()
    assert not at.exception
    assert any("システムダイナミクス" in t.value for t in at.title)


def test_single_simulation_turn_produces_analysis() -> None:
    at = AppTest.from_file("app.py", default_timeout=40)
    at.run()
    at.chat_input[0].set_value("広告費を1.5倍にしたら売上は?").run()
    # HITL form appears.
    assert any(b.label == "この内容で実行" for b in at.button)
    [b for b in at.button if b.label == "この内容で実行"][0].click().run()
    assert not at.exception
    assert any("シミュレーション結果" in m.value for m in at.markdown)


def test_multiple_simulations_in_history_have_unique_widget_keys() -> None:
    """Regression: several simulations in one run must not collide on widget IDs."""
    at = AppTest.from_file("app.py", default_timeout=40)
    at.run()
    from app import ChatTurn

    at.session_state["history"] = [
        ChatTurn("user", "q1"),
        ChatTurn("assistant", "a1", _SIM),
        ChatTurn("user", "q2"),
        ChatTurn("assistant", "a2", _SIM),
    ]
    at.run()
    assert not at.exception
    keys = [m.key for m in at.multiselect]
    assert len(keys) == 2
    assert len(set(keys)) == 2  # distinct
