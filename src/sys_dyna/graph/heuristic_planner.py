from __future__ import annotations

import re
from typing import Any

from ..simulation.models import ModelSpec, Scenario
from .planner import Planner


# Keyword cues (mirrors the spirit of the v1.0 MockGeminiClient).
_SIMULATE_CUES = ("倍", "したら", "増や" , "減ら" , "上げ", "下げ", "シミュ", "予測", "どうなる", "なったら")
_PAST_CUES = ("過去", "以前", "前回", "似た", "事例", "履歴", "あった")

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
        if any(c in text for c in _SIMULATE_CUES):
            return "simulate"
        if any(c in text for c in _PAST_CUES):
            return "past_reference"
        return "general"

    def select_model(self, user_text: str, catalog: list[dict[str, str]]) -> str | None:
        if not catalog:
            return None
        # Prefer a model whose name/description shares a keyword with the query.
        for entry in catalog:
            blob = f"{entry.get('name','')}{entry.get('description','')}"
            if any(tok and tok in blob for tok in re.findall(r"[一-龥ァ-ンA-Za-z]{2,}", user_text or "")):
                return entry["model_id"]
        return catalog[0]["model_id"]

    def extract_scenarios(self, user_text: str, model: ModelSpec) -> list[Scenario]:
        defaults = model.default_params()
        text = _normalize_digits(user_text or "")
        mults = [float(m) for m in _MULT_RE.findall(text)]

        # The primary driver parameter to scale (first param, e.g. ad_spend).
        driver = model.params[0].name if model.params else None
        if driver is None or not mults:
            return [Scenario(name="base", params=dict(defaults))]

        scenarios: list[Scenario] = []
        for mult in mults[:5]:
            params = dict(defaults)
            params[driver] = defaults[driver] * mult
            scenarios.append(Scenario(name=f"{driver}_x{mult:g}", params=params))
        return scenarios

    def analyze(
        self,
        user_text: str,
        model: ModelSpec | None,
        simulation: dict[str, Any] | None,
        past_references: list[dict[str, Any]],
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
