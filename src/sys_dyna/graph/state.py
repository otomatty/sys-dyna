from __future__ import annotations

from typing import Any, Literal, TypedDict


Intent = Literal["simulate", "past_reference", "general"]


class ScenarioDict(TypedDict):
    """Scenario in JSON-serialisable form so it survives checkpointing."""

    name: str
    params: dict[str, float]


class AgentState(TypedDict, total=False):
    """State threaded through the LangGraph StateGraph.

    Kept JSON-serialisable end-to-end so the Postgres checkpointer can persist
    and resume a turn (notably across the HITL ``interrupt`` in confirm_params).
    """

    session_id: str
    user_id: str
    user_text: str
    # Prior conversation as [{"role": "user"|"assistant", "content": str}], passed
    # in at the start of each turn so the LLM has multi-turn (follow-up) context.
    # Plain dicts (not LangChain BaseMessage) keep the state checkpoint JSON-safe.
    messages: list[dict[str, Any]]

    intent: Intent
    selected_model_id: str | None
    scenarios: list[ScenarioDict]
    confirmed: bool

    simulation: dict[str, Any] | None  # SimulationRun.to_payload()
    past_references: list[dict[str, Any]]
    analysis: str | None
    error: str | None
