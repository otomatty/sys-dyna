from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """JSON-Schema-like definition surfaced to the LLM as a function declaration."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolResult:
    payload: Any
    truncated: bool = False


class ToolError(Exception):
    """Raised when a tool fails. The orchestrator turns this into a structured
    error response that the LLM can react to (per design doc section 8.2)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_payload(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message}


class Tool(ABC):
    definition: ToolDefinition

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError
