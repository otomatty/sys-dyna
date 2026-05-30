from __future__ import annotations

import pytest

from sys_dyna.ui.charts import scenario_variables, to_long_frame


pytest.importorskip("pandas")


_SIM = {
    "model_id": "sales_growth",
    "scenarios": [
        {
            "scenario": "base",
            "params": {"ad_spend": 100.0},
            "variables": {
                "Sales": [{"t": 0.0, "v": 1000.0}, {"t": 1.0, "v": 1000.0}],
                "acquisition": [{"t": 0.0, "v": 50.0}, {"t": 1.0, "v": 50.0}],
            },
        },
        {
            "scenario": "x1.5",
            "params": {"ad_spend": 150.0},
            "variables": {
                "Sales": [{"t": 0.0, "v": 1000.0}, {"t": 1.0, "v": 1025.0}],
            },
        },
    ],
    "warnings": [],
}


def test_scenario_variables_union_order_preserving() -> None:
    assert scenario_variables(_SIM) == ["Sales", "acquisition"]


def test_to_long_frame_shape_and_values() -> None:
    frame = to_long_frame(_SIM, "Sales")
    assert list(frame.columns) == ["t", "value", "scenario"]
    # 2 scenarios x 2 timepoints = 4 rows.
    assert len(frame) == 4
    assert set(frame["scenario"]) == {"base", "x1.5"}
    last = frame[(frame["scenario"] == "x1.5") & (frame["t"] == 1.0)]["value"].iloc[0]
    assert last == 1025.0


def test_to_long_frame_missing_variable_is_empty() -> None:
    frame = to_long_frame(_SIM, "DoesNotExist")
    assert frame.empty
