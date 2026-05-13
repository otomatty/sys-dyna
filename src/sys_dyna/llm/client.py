from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..tools.base import ToolDefinition


Role = Literal["system", "user", "assistant", "tool"]
ThinkingLevel = Literal["low", "medium", "high"]


@dataclass
class LLMToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class LLMMessage:
    role: Role
    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # for role == "tool"
    tool_name: str | None = None     # for role == "tool"


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
        thinking: ThinkingLevel = "medium",
        timeout_sec: float | None = None,
    ) -> LLMResponse:
        ...
