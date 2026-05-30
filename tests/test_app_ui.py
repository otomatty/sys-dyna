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


def test_monte_carlo_turn_confirms_then_renders_result() -> None:
    """The analysis interrupt must use its own confirm form and surface the
    result — not fall through the scenario form and be dropped as cancelled."""
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.chat_input[0].set_value("広告費のばらつきをモンテカルロで30回見たい").run()
    # The dedicated analysis-confirm form appears (distinct run-button label).
    assert any(b.label == "この設定で実行" for b in at.button)
    [b for b in at.button if b.label == "この設定で実行"][0].click().run()
    assert not at.exception
    # The analysis summary is shown, and it is NOT treated as a cancellation.
    assert any("モンテカルロ" in m.value for m in at.markdown)
    assert not any("キャンセル" in m.value for m in at.markdown)


def test_optimize_confirm_blocks_invalid_bounds() -> None:
    """An invalid range (low >= high) must surface an error and not submit."""
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.chat_input[0].set_value("広告費をベイズ最適化して").run()
    assert any(b.label == "この設定で実行" for b in at.button)
    lows = [n for n in at.number_input if n.key and n.key.endswith("_r0_low")]
    highs = [n for n in at.number_input if n.key and n.key.endswith("_r0_high")]
    assert lows and highs
    lows[0].set_value(500.0)
    highs[0].set_value(100.0)
    [b for b in at.button if b.label == "この設定で実行"][0].click().run()
    assert not at.exception
    # The error is shown; nothing is run and the turn is not falsely cancelled.
    assert any("high は low より大きく" in e.value for e in at.error)
    assert not any("キャンセル" in m.value for m in at.markdown)


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
