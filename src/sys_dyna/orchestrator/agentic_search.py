from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import get_settings
from ..db.connection import _connect  # type: ignore[attr-defined]
from ..llm.client import LLMClient, LLMMessage, LLMToolCall
from ..llm.prompts import SYSTEM_PROMPT
from ..repository import sessions as sessions_repo
from ..repository import tool_call_logs as logs_repo
from ..tools.base import Tool, ToolError, ToolResult


logger = logging.getLogger(__name__)


@dataclass
class ToolInvocation:
    """A single tool call that occurred within a turn, surfaced to UI / tests."""

    name: str
    arguments: dict[str, Any]
    output: Any
    duration_ms: int
    error: str | None = None


@dataclass
class AgentTurnResult:
    text: str
    invocations: list[ToolInvocation] = field(default_factory=list)
    hit_loop_limit: bool = False
    hit_turn_timeout: bool = False


class AgenticSearchOrchestrator:
    """Implements F-01 / F-05.

    Drives the LLM <-> tool loop with the limits from section 5.2:
    - up to ``max_tool_calls`` tool invocations per turn
    - 10 second per-tool timeout
    - 60 second total turn budget
    - persists tool call telemetry to ``tool_call_logs``
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: dict[str, Tool],
        connection_factory: Callable[[], sqlite3.Connection] | None = None,
        max_tool_calls: int | None = None,
        per_tool_timeout_sec: float | None = None,
        turn_timeout_sec: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        settings = get_settings()
        self.llm = llm
        self.tools = tools
        self.max_tool_calls = (
            settings.max_tool_calls if max_tool_calls is None else max_tool_calls
        )
        self.per_tool_timeout = (
            settings.per_tool_timeout_sec
            if per_tool_timeout_sec is None
            else per_tool_timeout_sec
        )
        self.turn_timeout = (
            settings.turn_timeout_sec if turn_timeout_sec is None else turn_timeout_sec
        )
        # Apply the same fail-fast checks to caller-supplied overrides so the
        # settings-level validation in get_settings() cannot be bypassed.
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be > 0")
        if self.per_tool_timeout <= 0:
            raise ValueError("per_tool_timeout_sec must be > 0")
        if self.turn_timeout <= 0:
            raise ValueError("turn_timeout_sec must be > 0")
        self._clock = clock

        if connection_factory is None:
            db_path: Path = settings.db_path
            self._connect = lambda: _connect(db_path)
        else:
            self._connect = connection_factory

    def run_turn(
        self,
        session_id: str,
        user_text: str,
        history: list[LLMMessage] | None = None,
    ) -> AgentTurnResult:
        history = list(history or [])
        messages: list[LLMMessage] = [LLMMessage(role="system", content=SYSTEM_PROMPT)]
        messages.extend(history)
        messages.append(LLMMessage(role="user", content=user_text))

        deadline = self._clock() + self.turn_timeout
        invocations: list[ToolInvocation] = []
        tool_definitions = [t.definition for t in self.tools.values()]

        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return self._finalise_timeout(messages, invocations)

            try:
                response = self.llm.generate(
                    messages=messages,
                    tools=tool_definitions,
                    thinking="medium",
                    timeout_sec=remaining,
                )
            except Exception:
                logger.exception("llm.generate failed in main loop")
                return AgentTurnResult(
                    text="(応答生成中にエラーが発生しました。時間をおいて再試行してください。)",
                    invocations=invocations,
                )

            if not response.tool_calls:
                return AgentTurnResult(
                    text=response.text,
                    invocations=invocations,
                    hit_loop_limit=False,
                )

            if len(invocations) >= self.max_tool_calls:
                # The cap is on cumulative tool invocations, not on LLM iterations.
                return self._finalise_loop_limit(messages, invocations, deadline)

            # Truncate this batch so we never exceed the per-turn invocation cap.
            permitted = self.max_tool_calls - len(invocations)
            calls_to_run = list(response.tool_calls[:permitted])

            messages.append(
                LLMMessage(role="assistant", content=response.text, tool_calls=calls_to_run)
            )

            for call in calls_to_run:
                invocation = self._invoke_one(session_id, call, deadline)
                invocations.append(invocation)

                tool_message = LLMMessage(
                    role="tool",
                    content=json.dumps(
                        invocation.output if invocation.error is None
                        else {"error": invocation.error, "payload": invocation.output},
                        ensure_ascii=False,
                        default=str,
                    ),
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                )
                messages.append(tool_message)

                if self._clock() >= deadline:
                    return self._finalise_timeout(messages, invocations)

            if len(invocations) >= self.max_tool_calls:
                return self._finalise_loop_limit(messages, invocations, deadline)

    def _invoke_one(
        self,
        session_id: str,
        call: LLMToolCall,
        deadline: float,
    ) -> ToolInvocation:
        tool = self.tools.get(call.name)
        started = self._clock()
        called_at = _now_iso()
        if tool is None:
            err = f"unknown tool: {call.name}"
            duration_ms = int((self._clock() - started) * 1000)
            self._log_tool_call(
                session_id=session_id,
                tool_name=call.name,
                arguments=call.arguments,
                output={"error": "unknown_tool", "message": err},
                called_at=called_at,
                duration_ms=duration_ms,
            )
            return ToolInvocation(
                name=call.name,
                arguments=call.arguments,
                output=None,
                duration_ms=duration_ms,
                error=err,
            )

        budget = min(self.per_tool_timeout, max(0.0, deadline - self._clock()))
        # Use a daemon thread (not ThreadPoolExecutor): a daemon thread cannot
        # block interpreter shutdown, so even if a tool hangs forever the
        # process can still exit cleanly. ThreadPoolExecutor workers are
        # non-daemon and accumulate over repeated timeouts.
        timed_out, payload = _run_with_timeout(_safe_run, (tool, call.arguments), budget)
        if timed_out:
            duration_ms = int((self._clock() - started) * 1000)
            err = f"tool '{call.name}' exceeded per-call timeout of {self.per_tool_timeout}s"
            output: Any = {"error": "timeout", "message": err}
            self._log_tool_call(
                session_id=session_id,
                tool_name=call.name,
                arguments=call.arguments,
                output=output,
                called_at=called_at,
                duration_ms=duration_ms,
            )
            return ToolInvocation(
                name=call.name,
                arguments=call.arguments,
                output=output,
                duration_ms=duration_ms,
                error=err,
            )
        result = payload

        duration_ms = int((self._clock() - started) * 1000)
        if isinstance(result, ToolError):
            output_payload = result.to_payload()
            self._log_tool_call(
                session_id=session_id,
                tool_name=call.name,
                arguments=call.arguments,
                output=output_payload,
                called_at=called_at,
                duration_ms=duration_ms,
            )
            return ToolInvocation(
                name=call.name,
                arguments=call.arguments,
                output=output_payload,
                duration_ms=duration_ms,
                error=result.message,
            )
        if isinstance(result, Exception):
            output_payload = {"error": "tool_exception", "message": str(result)}
            self._log_tool_call(
                session_id=session_id,
                tool_name=call.name,
                arguments=call.arguments,
                output=output_payload,
                called_at=called_at,
                duration_ms=duration_ms,
            )
            return ToolInvocation(
                name=call.name,
                arguments=call.arguments,
                output=output_payload,
                duration_ms=duration_ms,
                error=str(result),
            )

        assert isinstance(result, ToolResult)
        self._log_tool_call(
            session_id=session_id,
            tool_name=call.name,
            arguments=call.arguments,
            output=result.payload,
            called_at=called_at,
            duration_ms=duration_ms,
        )
        return ToolInvocation(
            name=call.name,
            arguments=call.arguments,
            output=result.payload,
            duration_ms=duration_ms,
        )

    def _finalise_loop_limit(
        self,
        messages: list[LLMMessage],
        invocations: list[ToolInvocation],
        deadline: float,
    ) -> AgentTurnResult:
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "ツール呼び出し回数の上限に達しました。これ以上ツールは利用できません。"
                    "これまでに収集した情報のみで、ユーザーへの最終回答を日本語で生成してください。"
                ),
            )
        )
        remaining = max(0.0, deadline - self._clock())
        try:
            final = self.llm.generate(
                messages=messages,
                tools=[],
                thinking="medium",
                timeout_sec=remaining,
            )
            text = final.text or "(情報が不十分なため十分な回答ができませんでした。)"
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("final-answer generation failed: %s", e)
            text = "(最終応答の生成に失敗しました。)"
        return AgentTurnResult(text=text, invocations=invocations, hit_loop_limit=True)

    def _finalise_timeout(
        self,
        messages: list[LLMMessage],
        invocations: list[ToolInvocation],
    ) -> AgentTurnResult:
        return AgentTurnResult(
            text="(処理時間が上限に達したため、ここまでで応答します。)",
            invocations=invocations,
            hit_turn_timeout=True,
        )

    def _log_tool_call(
        self,
        *,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
        called_at: str,
        duration_ms: int,
    ) -> None:
        try:
            conn = self._connect()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("could not open DB for tool log: %s", e)
            return
        try:
            # The session row must exist before we can FK-reference it.
            if not sessions_repo.get(conn, session_id):
                logger.debug("skipping tool log: session %s not present", session_id)
                return
            logs_repo.record(
                conn,
                logs_repo.ToolCallLog(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_input=arguments,
                    tool_output=output,
                    called_at=called_at,
                    duration_ms=duration_ms,
                ),
            )
        finally:
            conn.close()


def _safe_run(tool: Tool, arguments: dict[str, Any]) -> Any:
    try:
        return tool.run(arguments)
    except ToolError as e:
        return e
    except Exception as e:  # pragma: no cover - defensive
        return e


def _run_with_timeout(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    timeout: float,
) -> tuple[bool, Any]:
    """Run ``fn(*args)`` in a daemon thread, returning (timed_out, result).

    Using a daemon thread (instead of a ThreadPoolExecutor) ensures that a
    hung tool cannot prevent interpreter shutdown — leaked threads die with
    the process. The thread cannot actually be killed if it hangs, but
    `daemon=True` keeps the process exitable.
    """
    holder: list[Any] = []

    def target() -> None:
        holder.append(fn(*args))

    t = threading.Thread(target=target, daemon=True, name="sys-dyna-tool")
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return True, None
    if not holder:
        # Should not happen, but treat as timeout to be safe.
        return True, None
    return False, holder[0]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
