from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..tools.base import ToolDefinition
from .client import LLMMessage, LLMResponse, LLMToolCall, ThinkingLevel


PAST_KEYWORDS = ("過去", "以前", "前回", "似た", "事例", "ありました", "あった", "履歴")
NUMERIC_KEYWORDS = ("数値", "比較", "結果", "推移", "時系列", "グラフ")


@dataclass
class MockGeminiClient:
    """Keyword-driven stand-in for the real Gemini 3.1 Pro function-calling endpoint.

    Behaviour mirrors the design doc sequence (query_sessions ->
    get_session_full -> get_simulation_results -> final answer) so the
    orchestrator and UI can be exercised end-to-end without external services.
    """

    model_name: str = "mock-gemini-3.1-pro"

    # Allows tests to force a "drain" mode that always requests tool calls until
    # the orchestrator's loop cap kicks in.
    force_tool_loop: bool = False

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
        thinking: ThinkingLevel = "medium",
        timeout_sec: float | None = None,
    ) -> LLMResponse:
        tool_names = {t.name for t in tools}

        latest_user = _last_user_message(messages)
        recent_tool_outputs = _recent_tool_outputs(messages)

        if self.force_tool_loop and "query_sessions" in tool_names:
            return LLMResponse(
                text="",
                tool_calls=[
                    LLMToolCall(
                        name="query_sessions",
                        arguments={"keywords": ["loop"], "limit": 5},
                    )
                ],
            )

        if not _has_called(messages, "query_sessions") and _looks_like_past_question(
            latest_user
        ):
            keywords = _extract_keywords(latest_user)
            if "query_sessions" in tool_names and keywords:
                return LLMResponse(
                    text="",
                    tool_calls=[
                        LLMToolCall(
                            name="query_sessions",
                            arguments={"keywords": keywords, "limit": 5},
                        )
                    ],
                )

        last_qs = recent_tool_outputs.get("query_sessions")
        if (
            last_qs is not None
            and "get_session_full" in tool_names
            and not _has_called(messages, "get_session_full")
        ):
            session_id = _first_session_id(last_qs)
            if session_id:
                return LLMResponse(
                    text="",
                    tool_calls=[
                        LLMToolCall(
                            name="get_session_full",
                            arguments={"session_id": session_id},
                        )
                    ],
                )

        last_full = recent_tool_outputs.get("get_session_full")
        if (
            last_full is not None
            and "get_simulation_results" in tool_names
            and not _has_called(messages, "get_simulation_results")
            and _wants_numbers(latest_user)
        ):
            session_id = _session_id_from_full(last_full) or _first_session_id(
                last_qs or {}
            )
            if session_id:
                return LLMResponse(
                    text="",
                    tool_calls=[
                        LLMToolCall(
                            name="get_simulation_results",
                            arguments={"session_id": session_id},
                        )
                    ],
                )

        return LLMResponse(text=_compose_final_answer(latest_user, recent_tool_outputs))


def _last_user_message(messages: Iterable[LLMMessage]) -> str:
    text = ""
    for m in messages:
        if m.role == "user":
            text = m.content
    return text


def _has_called(messages: Iterable[LLMMessage], name: str) -> bool:
    for m in messages:
        for tc in m.tool_calls:
            if tc.name == name:
                return True
    return False


def _recent_tool_outputs(messages: Iterable[LLMMessage]) -> dict[str, Any]:
    """Return the latest payload for each tool, parsed back into Python."""
    out: dict[str, Any] = {}
    for m in messages:
        if m.role == "tool" and m.tool_name:
            try:
                payload = json.loads(m.content)
            except json.JSONDecodeError:
                payload = m.content
            out[m.tool_name] = payload
    return out


def _looks_like_past_question(text: str) -> bool:
    if not text:
        return False
    return any(k in text for k in PAST_KEYWORDS)


def _wants_numbers(text: str) -> bool:
    if not text:
        return False
    return any(k in text for k in NUMERIC_KEYWORDS)


# Pick out runs of kanji or katakana (length >= 2) plus latin words.
# Hiragana is excluded so we don't grab particle-laden phrases as a single token.
_TOKEN_RE = re.compile(r"[一-龥々]{2,}|[ァ-ンー]{2,}|[A-Za-z]{3,}")
_STOPWORDS = {
    "過去",
    "以前",
    "前回",
    "事例",
    "履歴",
    "数値",
    "比較",
    "結果",
    "推移",
    "時系列",
    "グラフ",
    "分析",
    "今回",
}


def _extract_keywords(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    seen: list[str] = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if t in seen:
            continue
        seen.append(t)
        if len(seen) >= 2:
            break
    return seen


def _first_session_id(query_payload: Any) -> str | None:
    if not isinstance(query_payload, dict):
        return None
    sessions = query_payload.get("sessions") or []
    if not sessions:
        return None
    first = sessions[0]
    if isinstance(first, dict):
        sid = first.get("session_id")
        return sid if isinstance(sid, str) else None
    return None


def _session_id_from_full(full_payload: Any) -> str | None:
    if isinstance(full_payload, dict):
        sid = full_payload.get("session_id")
        return sid if isinstance(sid, str) else None
    return None


def _compose_final_answer(user_text: str, tool_outputs: dict[str, Any]) -> str:
    if not tool_outputs:
        if not user_text:
            return "ご質問の内容をもう少し具体的に教えてください。"
        return (
            "ご質問ありがとうございます。今回は過去セッションを参照する必要はないと判断しました。"
            "シミュレーションのご要望があればモデル名や入力変数を教えてください。"
        )

    parts: list[str] = []
    qs = tool_outputs.get("query_sessions")
    if isinstance(qs, dict):
        count = qs.get("count", 0)
        parts.append(f"過去セッションを {count} 件見つけました。")

    full = tool_outputs.get("get_session_full")
    if isinstance(full, dict):
        sid = full.get("session_id", "?")
        user_id = full.get("user_id", "?")
        created_at = full.get("created_at", "?")
        parts.append(f"代表セッション {sid}(担当: {user_id} / {created_at})の概要を確認しました。")
        log = full.get("chat_log") or []
        if isinstance(log, list) and log:
            first = log[0]
            if isinstance(first, dict):
                snippet = str(first.get("content", "")).strip()
                if snippet:
                    parts.append(f"このセッションは「{snippet[:80]}」から始まっています。")

    sim = tool_outputs.get("get_simulation_results")
    if isinstance(sim, dict):
        variables = sim.get("variables") or {}
        if isinstance(variables, dict) and variables:
            sample_var = next(iter(variables))
            series = variables.get(sample_var) or []
            tail = series[-1] if isinstance(series, list) and series else None
            if isinstance(tail, dict):
                parts.append(
                    f"数値結果(例: {sample_var})の最終値は t={tail.get('t')} の時点で {tail.get('v')} でした。"
                )

    parts.append("以上を踏まえると、今回のご質問にも同様の傾向が当てはまる可能性があります。")
    return "\n".join(parts)
