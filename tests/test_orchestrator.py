from __future__ import annotations

import time

from sys_dyna.db.connection import _connect
from sys_dyna.llm.client import LLMMessage, LLMResponse, LLMToolCall
from sys_dyna.llm.mock_client import MockGeminiClient
from sys_dyna.orchestrator import AgenticSearchOrchestrator
from sys_dyna.repository import sessions as sessions_repo
from sys_dyna.repository import tool_call_logs as logs_repo
from sys_dyna.tools import build_default_tools
from sys_dyna.tools.base import Tool, ToolDefinition, ToolError, ToolResult


def _ensure_session(db_path, session_id: str = "test-session") -> str:
    conn = _connect(db_path)
    try:
        sessions_repo.create_empty(conn, session_id, "yamada.taro", "MarketingMix_v3")
    finally:
        conn.close()
    return session_id


def test_full_loop_progression(seeded_db_path):
    session_id = _ensure_session(seeded_db_path)
    tools = build_default_tools(seeded_db_path)
    orch = AgenticSearchOrchestrator(
        llm=MockGeminiClient(),
        tools=tools,
        connection_factory=lambda: _connect(seeded_db_path),
    )
    result = orch.run_turn(
        session_id=session_id,
        user_text="過去に広告費を増やしたら売上がどうなったか数値で比較したい",
    )
    names = [inv.name for inv in result.invocations]
    assert names == ["query_sessions", "get_session_full", "get_simulation_results"]
    assert result.text
    assert not result.hit_loop_limit
    assert not result.hit_turn_timeout

    conn = _connect(seeded_db_path)
    try:
        logs = logs_repo.list_for_session(conn, session_id)
    finally:
        conn.close()
    assert [log.tool_name for log in logs] == names
    assert all(log.duration_ms is not None for log in logs)


def test_no_tool_calls_when_unrelated_question(seeded_db_path):
    session_id = _ensure_session(seeded_db_path, "no-tool-session")
    tools = build_default_tools(seeded_db_path)
    orch = AgenticSearchOrchestrator(
        llm=MockGeminiClient(),
        tools=tools,
        connection_factory=lambda: _connect(seeded_db_path),
    )
    result = orch.run_turn(
        session_id=session_id,
        user_text="モデルの基本的な解説をお願いします",
    )
    assert result.invocations == []


class _LoopingLLM:
    """Always asks to call query_sessions until the loop cap kicks in."""

    def __init__(self) -> None:
        self.calls = 0
        self.final_call = False

    def generate(self, messages, tools, thinking="medium", timeout_sec=None):
        # When the orchestrator asks for a final answer it passes tools=[].
        if not tools:
            self.final_call = True
            return LLMResponse(text="LOOP-CAP-FINAL")
        self.calls += 1
        return LLMResponse(
            text="",
            tool_calls=[
                LLMToolCall(name="query_sessions", arguments={"keywords": ["a"]})
            ],
        )


def test_loop_limit_triggers_finalisation(seeded_db_path):
    session_id = _ensure_session(seeded_db_path, "loop-cap-session")
    tools = build_default_tools(seeded_db_path)
    llm = _LoopingLLM()
    orch = AgenticSearchOrchestrator(
        llm=llm,
        tools=tools,
        connection_factory=lambda: _connect(seeded_db_path),
        max_tool_calls=3,
    )
    result = orch.run_turn(session_id=session_id, user_text="ループしてください")
    assert result.hit_loop_limit
    assert len(result.invocations) == 3
    assert llm.final_call
    assert result.text == "LOOP-CAP-FINAL"


class _BatchLLM:
    """Returns a batch of `batch_size` tool calls in a single response, then a
    final answer once tools is empty.
    """

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def generate(self, messages, tools, thinking="medium", timeout_sec=None):
        if not tools:
            return LLMResponse(text="BATCH-CAPPED")
        return LLMResponse(
            text="",
            tool_calls=[
                LLMToolCall(name="query_sessions", arguments={"keywords": ["a"]})
                for _ in range(self.batch_size)
            ],
        )


def test_cap_truncates_oversized_batch(seeded_db_path):
    """When a single LLM response asks for more tool calls than the cap allows,
    the orchestrator must truncate the batch instead of running every call.
    """
    session_id = _ensure_session(seeded_db_path, "batch-cap-session")
    tools = build_default_tools(seeded_db_path)
    orch = AgenticSearchOrchestrator(
        llm=_BatchLLM(batch_size=5),
        tools=tools,
        connection_factory=lambda: _connect(seeded_db_path),
        max_tool_calls=2,
    )
    result = orch.run_turn(session_id=session_id, user_text="batched")
    assert result.hit_loop_limit
    assert len(result.invocations) == 2
    assert result.text == "BATCH-CAPPED"


class _SlowTool(Tool):
    definition = ToolDefinition(
        name="slow_tool",
        description="hangs forever",
        parameters={"type": "object", "properties": {}},
    )

    def run(self, arguments):
        time.sleep(2.0)
        return ToolResult(payload={"ok": True})


class _SingleSlowCallLLM:
    def __init__(self) -> None:
        self.called = 0

    def generate(self, messages, tools, thinking="medium", timeout_sec=None):
        if not tools:
            return LLMResponse(text="timeout-final")
        self.called += 1
        if self.called == 1:
            return LLMResponse(
                text="",
                tool_calls=[LLMToolCall(name="slow_tool", arguments={})],
            )
        return LLMResponse(text="ok")


def test_per_tool_timeout(seeded_db_path):
    session_id = _ensure_session(seeded_db_path, "timeout-session")
    orch = AgenticSearchOrchestrator(
        llm=_SingleSlowCallLLM(),
        tools={"slow_tool": _SlowTool()},
        connection_factory=lambda: _connect(seeded_db_path),
        per_tool_timeout_sec=0.2,
        turn_timeout_sec=5.0,
    )
    started = time.monotonic()
    result = orch.run_turn(session_id=session_id, user_text="run slow tool")
    elapsed = time.monotonic() - started
    assert len(result.invocations) == 1
    inv = result.invocations[0]
    assert inv.error is not None
    assert "timeout" in inv.error.lower()
    # The orchestrator must not block waiting for the hung worker to finish:
    # a 2s sleep with a 0.2s timeout should return well under 1s.
    assert elapsed < 1.0, f"timeout did not abort early: elapsed={elapsed:.2f}s"


class _ErrorTool(Tool):
    definition = ToolDefinition(
        name="error_tool",
        description="always errors",
        parameters={"type": "object", "properties": {}},
    )

    def run(self, arguments):
        raise ToolError("bad", "broken")


class _OneCallThenAnswerLLM:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.called = 0

    def generate(self, messages, tools, thinking="medium", timeout_sec=None):
        self.called += 1
        if self.called == 1 and tools:
            return LLMResponse(
                text="",
                tool_calls=[LLMToolCall(name=self.tool_name, arguments={})],
            )
        return LLMResponse(text="recovered")


def test_tool_error_passed_to_llm(seeded_db_path):
    session_id = _ensure_session(seeded_db_path, "error-session")
    orch = AgenticSearchOrchestrator(
        llm=_OneCallThenAnswerLLM("error_tool"),
        tools={"error_tool": _ErrorTool()},
        connection_factory=lambda: _connect(seeded_db_path),
    )
    result = orch.run_turn(session_id=session_id, user_text="trigger error")
    assert result.text == "recovered"
    assert len(result.invocations) == 1
    assert result.invocations[0].error == "broken"

    conn = _connect(seeded_db_path)
    try:
        logs = logs_repo.list_for_session(conn, session_id)
    finally:
        conn.close()
    assert len(logs) == 1
    assert logs[0].tool_output == {"error": "bad", "message": "broken"}
