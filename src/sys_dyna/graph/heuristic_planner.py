from __future__ import annotations

import re
from typing import Any

from ..simulation.analysis import build_default_analysis_request
from ..simulation.models import ModelSpec, Scenario
from .planner import Planner


# Keyword cues (mirrors the spirit of the v1.0 MockGeminiClient).
_SIMULATE_CUES = ("倍", "したら", "増や" , "減ら" , "上げ", "下げ", "シミュ", "予測", "どうなる", "なったら")
_PAST_CUES = ("過去", "以前", "前回", "似た", "事例", "履歴", "あった")
# Advanced-analysis cues. Checked before the simulate cues so e.g. "広告費を
# 最適化したら" routes to optimisation rather than a plain scenario run.
_MONTECARLO_CUES = ("モンテカルロ", "monte", "ばらつき", "不確実", "リスク", "確率", "分布", "感度")
_OPTIMIZE_CUES = ("最適化", "最適な", "ベイズ", "optimi", "最大化", "最小化", "チューニング", "探索")

# "1.5倍" / "２倍" style multipliers.
_MULT_RE = re.compile(r"([0-9０-９]+(?:[.．][0-9０-９]+)?)\s*倍")


def _normalize_digits(s: str) -> str:
    table = str.maketrans("０１２３４５６７８９．", "0123456789.")
    return s.translate(table)


class HeuristicPlanner(Planner):
    """API-free planner for offline/demo use and as a Gemini fallback.

    Deterministic keyword/regex heuristics — no external calls. The production
    path uses ``GeminiPlanner`` instead; both satisfy the ``Planner`` protocol.
    """

    def classify_intent(self, user_text: str, history: list[dict[str, Any]]) -> str:
        text = user_text or ""
        lowered = text.lower()
        # Advanced analyses take precedence over a plain simulate request.
        if any(c in text or c in lowered for c in _MONTECARLO_CUES):
            return "montecarlo"
        if any(c in text or c in lowered for c in _OPTIMIZE_CUES):
            return "optimize"
        if any(c in text for c in _SIMULATE_CUES):
            return "simulate"
        if any(c in text for c in _PAST_CUES):
            return "past_reference"
        # Follow-up: after a prior exchange, a bare numeric tweak ("churn を 0.1 に")
        # is treated as a continued simulation request.
        if history and re.search(r"[0-9０-９]", _normalize_digits(text)):
            return "simulate"
        return "general"

    def select_model(
        self,
        user_text: str,
        catalog: list[dict[str, str]],
        history: list[dict[str, Any]],
    ) -> str | None:
        if not catalog:
            return None
        # Prefer a model whose name/description shares a keyword with the query.
        for entry in catalog:
            blob = f"{entry.get('name','')}{entry.get('description','')}"
            if any(tok and tok in blob for tok in re.findall(r"[一-龥ァ-ンA-Za-z]{2,}", user_text or "")):
                return entry["model_id"]
        return catalog[0]["model_id"]

    def extract_scenarios(
        self,
        user_text: str,
        model: ModelSpec,
        history: list[dict[str, Any]],
        base_params: dict[str, float] | None = None,
    ) -> list[Scenario]:
        # Start from a prior turn's values when present so follow-up tweaks
        # preserve unchanged parameters (and multipliers compose on the base).
        base = dict(model.default_params())
        if base_params:
            for key, val in base_params.items():
                spec = model.param(key)
                if spec is not None:
                    base[key] = spec.clamp(float(val))
        text = _normalize_digits(user_text or "")

        # Parse each named parameter's number and whether it is a "N倍" multiplier
        # ("広告費を1.5倍" -> mult) or an absolute value ("churn_rate を 0.1 に").
        named: dict[str, tuple[float, bool]] = {}
        for p in model.params:
            for token in (p.name, p.label):
                if not token:
                    continue
                m = re.search(
                    re.escape(token) + r"[^0-9.-]{0,6}([0-9]+(?:\.[0-9]+)?)\s*(倍)?", text
                )
                if m:
                    named[p.name] = (float(m.group(1)), m.group(2) == "倍")
                    break

        overrides = {
            name: model.param(name).clamp(num)
            for name, (num, is_mult) in named.items()
            if not is_mult
        }
        # Multipliers target the parameter the user named (e.g. 解約率を2倍 ->
        # churn_rate), or the first driver param when no parameter is named.
        mult_param = next((n for n, (_, is_mult) in named.items() if is_mult), None)
        mults = [float(x) for x in _MULT_RE.findall(text)]

        if mults:
            target = mult_param or (model.params[0].name if model.params else None)
            if target:
                tspec = model.param(target)
                scenarios: list[Scenario] = []
                for mult in mults[:5]:
                    params = dict(base)
                    params.update(overrides)
                    params[target] = tspec.clamp(base[target] * mult)
                    scenarios.append(Scenario(name=f"{target}_x{mult:g}", params=params))
                return scenarios
        if overrides:
            params = dict(base)
            params.update(overrides)
            return [Scenario(name="custom", params=params)]
        return [Scenario(name="base", params=dict(base))]

    def build_analysis_request(
        self,
        user_text: str,
        model: ModelSpec,
        kind: str,
        history: list[dict[str, Any]],
        base_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        normalized = "montecarlo" if kind not in ("montecarlo", "optimize") else kind
        return build_default_analysis_request(
            _normalize_digits(user_text or ""), model, normalized, base_params
        )

    def analyze(
        self,
        user_text: str,
        model: ModelSpec | None,
        simulation: dict[str, Any] | None,
        past_references: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        if simulation and simulation.get("scenarios"):
            target = model.output_variables[0] if (model and model.output_variables) else "Sales"
            parts.append(f"シミュレーション結果（{target}）の最終値:")
            for sc in simulation["scenarios"]:
                series = sc["variables"].get(target) or next(iter(sc["variables"].values()), [])
                if series:
                    last = series[-1]
                    parts.append(f"・{sc['scenario']}: t={last['t']:g} で {last['v']:.1f}")
            base = None
            best = None
            for sc in simulation["scenarios"]:
                series = sc["variables"].get(target) or []
                if not series:
                    continue
                final = series[-1]["v"]
                if base is None:
                    base = final
                if best is None or final > best[1]:
                    best = (sc["scenario"], final)
            if best and base and best[1] > base:
                parts.append(
                    f"最も {target} が高いのは「{best[0]}」で、基準シナリオ比 "
                    f"{(best[1] / base - 1) * 100:.0f}% の増加です。"
                )
        if past_references:
            parts.append(f"過去の類似分析を {len(past_references)} 件参照しました。")
        if not parts:
            return (
                "ご質問ありがとうございます。シミュレーションが必要な場合は、"
                "モデルや変更したい変数（例: 広告費を1.5倍）を教えてください。"
            )
        return "\n".join(parts)
