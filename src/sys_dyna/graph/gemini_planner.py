from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..simulation.analysis import build_default_analysis_request
from ..simulation.models import ModelSpec, ParamSpec, Scenario
from .planner import Planner


logger = logging.getLogger(__name__)

_VALID_INTENTS = ("simulate", "past_reference", "montecarlo", "optimize", "general")

_INTENT_PROMPT = """\
あなたはシステムダイナミクス分析ツールのルータです。
ユーザーの発言を次のいずれかに分類し、ラベルのみを1語で出力してください。

- montecarlo: パラメータのばらつき・不確実性・リスク・確率分布・感度を調べたい
- optimize: パラメータを最適化したい(最大化/最小化・「最適な値は?」「ベイズ最適化」)
- simulate: 上記以外で、シミュレーションの実行や「もし〜したら」の予測を求めている
  (直前にシミュレーションを行った後の「ではXを〜に変えたら」等の追問も含む)
- past_reference: 過去・以前の分析事例の参照だけを求めている
- general: 上記以外の一般的な質問・雑談
{history}
ユーザー発言: {user_text}
ラベル:"""

_ANALYSIS_PROMPT = """\
ユーザーの要求を、高度なシミュレーション解析の設定(JSON)に変換します。
解析種別: {kind}  (montecarlo=モンテカルロ分析 / optimize=ベイズ最適化)
モデル: {model_name}
調整可能なパラメータ(name: 既定値, 説明):
{param_lines}
出力変数の候補: {outputs}
{current}{history}
ユーザー発言: {user_text}

次の JSON のみを出力してください(該当しないキーは省略可、既定が使われます)。
- objective: {{"variable": 出力変数名, "aggregate": "final|mean|min|max|sum", "direction": "maximize|minimize"}}
- montecarlo の場合: "distributions": [{{"name": パラメータ名, "kind": "normal|uniform|triangular|lognormal", "mean":, "std":, "low":, "high":}}], "iterations": 整数
- optimize の場合: "search_space": [{{"name": パラメータ名, "low": 下限, "high": 上限}}], "n_trials": 整数
JSON:"""

_SELECT_PROMPT = """\
ユーザーの要求に最も適したシミュレーションモデルを1つ選びます。
候補モデル(JSON): {catalog}
{history}
ユーザー発言: {user_text}

最も適切な model_id だけを出力してください。直前の会話で使ったモデルを継続する場合は
そのモデルを選びます。該当が無ければ none と出力してください。
model_id:"""

_EXTRACT_PROMPT = """\
ユーザーの要求を、モデルのパラメータ設定(シナリオ)に変換します。
モデル: {model_name}
調整可能なパラメータ(name: 既定値, 説明):
{param_lines}
{current}{history}
ユーザー発言: {user_text}

次の JSON のみを出力してください。複数シナリオの比較要求があれば複数要素にします。
追問の場合は「現在の設定値」を起点に、変更点だけ反映してください。
変更しないパラメータは省略可(起点の値が使われます)。
{{"scenarios": [{{"name": "シナリオ名", "params": {{"パラメータ名": 数値}}}}]}}
JSON:"""

_ANALYZE_PROMPT = """\
あなたはシステムダイナミクス分析の専門家です。以下の数値シミュレーション結果を解釈し、
日本語で簡潔に説明してください。要因・ボトルネック・示唆に触れ、断定しすぎないこと。
{history}
ユーザーの質問: {user_text}
モデル: {model_name}
シミュレーション結果(JSON): {simulation}
過去の参考分析(JSON): {past}

回答:"""


def format_history(history: list[dict[str, Any]], max_turns: int = 6) -> str:
    """Render recent conversation as a compact block for prompt injection.

    Returns "" when there is no history so single-turn prompts are unchanged.
    """
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        role = "ユーザー" if m.get("role") == "user" else "アシスタント"
        content = str(m.get("content", "")).strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:300]}")
    if not lines:
        return ""
    return "\n直近の会話:\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Pure parsing/normalisation helpers (unit-tested without any LLM call).
# --------------------------------------------------------------------------
def parse_intent(raw: str) -> str:
    """Map a free-form model reply to a valid intent label."""
    text = (raw or "").strip().lower()
    for intent in _VALID_INTENTS:
        if intent in text:
            return intent
    return "general"


def parse_model_id(raw: str, valid_ids: set[str]) -> str | None:
    """Extract a known model_id from a free-form reply, else None."""
    text = (raw or "").strip()
    if text.lower() in ("none", "なし", ""):
        return None
    # Exact match first, then substring (model ids are word-like).
    if text in valid_ids:
        return text
    for mid in valid_ids:
        if re.search(rf"\b{re.escape(mid)}\b", text):
            return mid
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, spec: ParamSpec) -> float:
    return spec.clamp(value)


def resolve_base_params(
    model: ModelSpec, base_params: dict[str, Any] | None
) -> dict[str, float]:
    """Starting point for a scenario: model defaults, overlaid with carried-over
    params from a prior turn (clamped; unknown/non-numeric entries dropped).

    This is what makes follow-up edits ("then set churn_rate to 0.1") preserve
    the previous run's unchanged values instead of reverting to defaults.
    """
    base = model.default_params()
    if base_params:
        for key, val in base_params.items():
            spec = model.param(key)
            if spec is None:
                continue
            fv = _coerce_float(val)
            if fv is None:
                continue
            base[key] = _clamp(fv, spec)
    return base


def parse_scenarios(
    raw: str,
    model: ModelSpec,
    max_scenarios: int = 5,
    base_params: dict[str, Any] | None = None,
) -> list[Scenario]:
    """Parse the extract-scenarios JSON into validated ``Scenario`` objects.

    - unknown parameter names are dropped
    - non-numeric values are ignored
    - values are clamped to each ParamSpec's [min, max]
    - omitted parameters fall back to ``base_params`` (a prior turn's values)
      or the model defaults
    Malformed output yields a single base scenario rather than raising.
    """
    base = resolve_base_params(model, base_params)
    data = _extract_json(raw)
    raw_scenarios = []
    if isinstance(data, dict):
        raw_scenarios = data.get("scenarios") or []
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        return [Scenario(name="base", params=dict(base))]

    scenarios: list[Scenario] = []
    for i, item in enumerate(raw_scenarios[:max_scenarios]):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"scenario_{i + 1}")
        params = dict(base)
        raw_params = item.get("params")
        if isinstance(raw_params, dict):
            for key, val in raw_params.items():
                spec = model.param(key)
                if spec is None:
                    continue
                fv = _coerce_float(val)
                if fv is None:
                    continue
                params[key] = _clamp(fv, spec)
        scenarios.append(Scenario(name=name, params=params))
    return scenarios or [Scenario(name="base", params=dict(base))]


def _merge_analysis_request(
    default: dict[str, Any],
    data: Any,
    kind: str,
    model: ModelSpec,
) -> dict[str, Any]:
    """Overlay an LLM-produced analysis spec onto the heuristic default.

    Only well-formed, model-relevant pieces are taken from ``data`` so a partial
    or noisy LLM reply degrades to the always-runnable default rather than
    producing an invalid request. Unknown parameter names are dropped.
    """
    if not isinstance(data, dict):
        return default
    valid_names = {p.name for p in model.params}
    merged = dict(default)

    obj = data.get("objective")
    if isinstance(obj, dict):
        base_obj = dict(default["objective"])
        if isinstance(obj.get("variable"), str) and obj["variable"]:
            base_obj["variable"] = obj["variable"]
        if obj.get("aggregate") in ("final", "initial", "mean", "min", "max", "sum"):
            base_obj["aggregate"] = obj["aggregate"]
        if obj.get("direction") in ("maximize", "minimize"):
            base_obj["direction"] = obj["direction"]
        merged["objective"] = base_obj

    if kind == "montecarlo":
        dists = [
            d
            for d in (data.get("distributions") or [])
            if isinstance(d, dict) and d.get("name") in valid_names
        ]
        if dists:
            merged["distributions"] = dists
        it = _coerce_int(data.get("iterations"))
        if it and it > 0:
            merged["iterations"] = it
    else:  # optimize
        space = [
            r
            for r in (data.get("search_space") or [])
            if isinstance(r, dict) and r.get("name") in valid_names
        ]
        if space:
            merged["search_space"] = space
        nt = _coerce_int(data.get("n_trials"))
        if nt and nt > 0:
            merged["n_trials"] = nt
    return merged


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_json(raw: str) -> Any:
    """Best-effort JSON extraction, tolerating ```json fences and prose."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _param_lines(model: ModelSpec) -> str:
    lines = []
    for p in model.params:
        desc = p.description or p.label
        lines.append(f"- {p.name}: {p.default} ({desc})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
class GeminiPlanner(Planner):
    """Production planner backed by Gemini (gemini-3.5-flash) via LangChain.

    The chat model is created lazily so importing this module never requires an
    API key or network access.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_scenarios: int = 5,
        chat_model: Any | None = None,
    ) -> None:
        self._model_name = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_scenarios = max_scenarios
        self._chat = chat_model  # injectable for testing

    def _llm(self) -> Any:
        if self._chat is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._chat = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=self._api_key,
                temperature=self._temperature,
            )
        return self._chat

    def _ask(self, prompt: str) -> str:
        resp = self._llm().invoke(prompt)
        content = getattr(resp, "content", resp)
        return content if isinstance(content, str) else str(content)

    def classify_intent(self, user_text: str, history: list[dict[str, Any]]) -> str:
        return parse_intent(
            self._ask(
                _INTENT_PROMPT.format(
                    user_text=user_text, history=format_history(history)
                )
            )
        )

    def select_model(
        self,
        user_text: str,
        catalog: list[dict[str, str]],
        history: list[dict[str, Any]],
    ) -> str | None:
        valid = {c["model_id"] for c in catalog}
        raw = self._ask(
            _SELECT_PROMPT.format(
                catalog=json.dumps(catalog, ensure_ascii=False),
                user_text=user_text,
                history=format_history(history),
            )
        )
        return parse_model_id(raw, valid)

    def extract_scenarios(
        self,
        user_text: str,
        model: ModelSpec,
        history: list[dict[str, Any]],
        base_params: dict[str, float] | None = None,
    ) -> list[Scenario]:
        current = ""
        if base_params:
            base = resolve_base_params(model, base_params)
            current = "現在の設定値: " + json.dumps(base, ensure_ascii=False) + "\n"
        raw = self._ask(
            _EXTRACT_PROMPT.format(
                model_name=model.name,
                param_lines=_param_lines(model),
                current=current,
                user_text=user_text,
                history=format_history(history),
            )
        )
        return parse_scenarios(raw, model, self._max_scenarios, base_params)

    def build_analysis_request(
        self,
        user_text: str,
        model: ModelSpec,
        kind: str,
        history: list[dict[str, Any]],
        base_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        normalized = "montecarlo" if kind not in ("montecarlo", "optimize") else kind
        # Heuristic default is the floor: it always yields a runnable spec, and
        # we overlay only the parts the LLM returns well-formed.
        request = build_default_analysis_request(user_text, model, normalized, base_params)
        current = ""
        if base_params:
            base = resolve_base_params(model, base_params)
            current = "現在の設定値: " + json.dumps(base, ensure_ascii=False) + "\n"
        try:
            raw = self._ask(
                _ANALYSIS_PROMPT.format(
                    kind=normalized,
                    model_name=model.name,
                    param_lines=_param_lines(model),
                    outputs=", ".join(model.output_variables) or "(モデル既定)",
                    current=current,
                    user_text=user_text,
                    history=format_history(history),
                )
            )
            data = _extract_json(raw)
        except Exception:  # pragma: no cover - defensive: fall back to default
            logger.exception("build_analysis_request LLM call failed; using default")
            data = None
        return _merge_analysis_request(request, data, normalized, model)

    def analyze(
        self,
        user_text: str,
        model: ModelSpec | None,
        simulation: dict[str, Any] | None,
        past_references: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> str:
        return self._ask(
            _ANALYZE_PROMPT.format(
                user_text=user_text,
                model_name=model.name if model else "(なし)",
                simulation=json.dumps(simulation, ensure_ascii=False, default=str),
                past=json.dumps(past_references, ensure_ascii=False, default=str),
                history=format_history(history),
            )
        ).strip()
