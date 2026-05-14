from __future__ import annotations

import pytest

from sys_dyna.tools.base import ToolError
from sys_dyna.tools.get_session_full import GetSessionFullTool
from sys_dyna.tools.get_simulation_results import GetSimulationResultsTool
from sys_dyna.tools.query_sessions import MAX_LIMIT, QuerySessionsTool


def test_query_sessions_keyword_match(connection_factory):
    tool = QuerySessionsTool(connection_factory)
    result = tool.run({"keywords": ["広告費"]})
    payload = result.payload
    assert payload["count"] == 1
    assert payload["sessions"][0]["session_id"] == "sess-2025-12-ad-revenue"
    # Excerpt should look human readable, not raw JSON
    assert "user" in payload["sessions"][0]["chat_excerpt"]


def test_query_sessions_keyword_and(connection_factory):
    tool = QuerySessionsTool(connection_factory)
    result = tool.run({"keywords": ["広告費", "在庫"]})
    assert result.payload["count"] == 0


def test_query_sessions_model_filter(connection_factory):
    tool = QuerySessionsTool(connection_factory)
    result = tool.run(
        {"keywords": ["リードタイム"], "model_name": "InventoryTurnover_v2"}
    )
    assert result.payload["count"] == 1


def test_query_sessions_limit_clamped(connection_factory):
    tool = QuerySessionsTool(connection_factory)
    result = tool.run({"keywords": ["シミュレーション"], "limit": 9999})
    # Clamp must apply: returned count cannot exceed MAX_LIMIT regardless of input.
    assert result.payload["count"] <= MAX_LIMIT
    assert len(result.payload["sessions"]) == result.payload["count"]


def test_query_sessions_invalid_since(connection_factory):
    tool = QuerySessionsTool(connection_factory)
    with pytest.raises(ToolError) as exc:
        tool.run({"keywords": ["広告費"], "since": "not-a-date"})
    assert exc.value.code == "invalid_argument"


def test_query_sessions_invalid_args(connection_factory):
    tool = QuerySessionsTool(connection_factory)
    with pytest.raises(ToolError):
        tool.run({"keywords": []})
    with pytest.raises(ToolError):
        tool.run({"keywords": [""]})


def test_get_session_full(connection_factory):
    tool = GetSessionFullTool(connection_factory)
    result = tool.run({"session_id": "sess-2025-12-ad-revenue"})
    payload = result.payload
    assert payload["user_id"] == "yamada.taro"
    assert len(payload["chat_log"]) == 4
    assert payload["simulation_results"][0]["result_id"] == "res-2025-12-ad-revenue"


def test_get_session_full_not_found(connection_factory):
    tool = GetSessionFullTool(connection_factory)
    with pytest.raises(ToolError) as exc:
        tool.run({"session_id": "does-not-exist"})
    assert exc.value.code == "not_found"


def test_get_simulation_results_all_variables(connection_factory):
    tool = GetSimulationResultsTool(connection_factory)
    result = tool.run({"session_id": "sess-2025-12-ad-revenue"})
    assert "revenue" in result.payload["variables"]
    assert "margin_pct" in result.payload["variables"]


def test_get_simulation_results_filtered(connection_factory):
    tool = GetSimulationResultsTool(connection_factory)
    result = tool.run(
        {"session_id": "sess-2025-12-ad-revenue", "variable_names": ["revenue", "missing"]}
    )
    payload = result.payload
    assert set(payload["variables"].keys()) == {"revenue"}
    assert payload["missing_variables"] == ["missing"]


def test_get_simulation_results_empty_list_returns_no_variables(connection_factory):
    tool = GetSimulationResultsTool(connection_factory)
    result = tool.run(
        {"session_id": "sess-2025-12-ad-revenue", "variable_names": []}
    )
    payload = result.payload
    # An empty (but provided) list means "no variables requested", not "all variables".
    assert payload["variables"] == {}
    assert payload["missing_variables"] == []
