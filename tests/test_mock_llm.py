from __future__ import annotations

import json

from sys_dyna.llm.client import LLMMessage, LLMToolCall
from sys_dyna.llm.mock_client import MockGeminiClient
from sys_dyna.tools.base import ToolDefinition


def _tool_defs() -> list[ToolDefinition]:
    return [
        ToolDefinition(name="query_sessions", description="", parameters={}),
        ToolDefinition(name="get_session_full", description="", parameters={}),
        ToolDefinition(name="get_simulation_results", description="", parameters={}),
    ]


def test_first_turn_with_past_keyword_invokes_query_sessions():
    client = MockGeminiClient()
    messages = [LLMMessage(role="user", content="過去に広告費の分析事例ありましたか?")]
    response = client.generate(messages, _tool_defs())
    assert response.tool_calls
    assert response.tool_calls[0].name == "query_sessions"


def test_progression_to_get_session_full():
    client = MockGeminiClient()
    messages = [
        LLMMessage(role="user", content="過去に似た広告費の分析あった?"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(name="query_sessions", arguments={"keywords": ["広告"]})],
        ),
        LLMMessage(
            role="tool",
            content=json.dumps(
                {
                    "count": 1,
                    "sessions": [
                        {
                            "session_id": "sess-x",
                            "created_at": "2025-12-04T00:00:00Z",
                            "user_id": "u",
                            "model_name": "M",
                            "chat_excerpt": "...",
                        }
                    ],
                }
            ),
            tool_name="query_sessions",
        ),
    ]
    response = client.generate(messages, _tool_defs())
    assert response.tool_calls
    call = response.tool_calls[0]
    assert call.name == "get_session_full"
    assert call.arguments == {"session_id": "sess-x"}


def test_skip_numeric_when_user_did_not_request_numbers():
    client = MockGeminiClient()
    messages = [
        LLMMessage(role="user", content="過去事例の概要だけ知りたい"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(name="query_sessions", arguments={"keywords": ["事例"]})],
        ),
        LLMMessage(
            role="tool",
            content=json.dumps({"count": 1, "sessions": [{"session_id": "sess-x"}]}),
            tool_name="query_sessions",
        ),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(name="get_session_full", arguments={"session_id": "sess-x"})],
        ),
        LLMMessage(
            role="tool",
            content=json.dumps({"session_id": "sess-x", "chat_log": [], "user_id": "u", "created_at": "x"}),
            tool_name="get_session_full",
        ),
    ]
    response = client.generate(messages, _tool_defs())
    assert response.tool_calls == []
    assert response.text


def test_numeric_keyword_triggers_simulation_results():
    client = MockGeminiClient()
    messages = [
        LLMMessage(role="user", content="過去事例の数値を比較したい"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(name="query_sessions", arguments={"keywords": ["事例"]})],
        ),
        LLMMessage(
            role="tool",
            content=json.dumps({"count": 1, "sessions": [{"session_id": "sess-y"}]}),
            tool_name="query_sessions",
        ),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(name="get_session_full", arguments={"session_id": "sess-y"})],
        ),
        LLMMessage(
            role="tool",
            content=json.dumps({"session_id": "sess-y", "chat_log": [], "user_id": "u", "created_at": "x"}),
            tool_name="get_session_full",
        ),
    ]
    response = client.generate(messages, _tool_defs())
    assert response.tool_calls
    assert response.tool_calls[0].name == "get_simulation_results"
    assert response.tool_calls[0].arguments == {"session_id": "sess-y"}


def test_no_tool_when_no_past_keyword():
    client = MockGeminiClient()
    messages = [LLMMessage(role="user", content="広告費を1.5倍にした影響だけ知りたい")]
    response = client.generate(messages, _tool_defs())
    assert response.tool_calls == []
    assert response.text
