from __future__ import annotations

from pathlib import Path

from .models import ModelRef, ModelSpec, ParamSpec


_CATALOG_DIR = Path(__file__).resolve().parent / "catalog_models"


def _catalog_ref(model_id: str, filename: str) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        source="catalog",
        path=str(_CATALOG_DIR / filename),
    )


# Starter catalog. The initial line-up is provisional (design v2.0 §11 / §7.1):
# a single advertising -> sales growth model so the end-to-end flow is
# exercisable. Add ModelSpec entries here as real models are curated.
_STARTER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="sales_growth",
        name="広告効果による売上成長モデル",
        description=(
            "広告費 (ad_spend) が新規顧客獲得を通じて売上 (Sales) を押し上げ、"
            "解約率 (churn_rate) が売上を減少させる、基本的なストック・フローモデル。"
            "広告費を増減させたときの売上推移の比較に使う。"
        ),
        # Load the pre-compiled .py (committed) so catalog models work on
        # read-only filesystems; sales_growth.xmile is kept as the source.
        ref=_catalog_ref("sales_growth", "sales_growth.py"),
        params=(
            ParamSpec(
                name="ad_spend",
                label="広告費",
                default=100.0,
                unit="万円/月",
                min=0.0,
                description="毎月の広告支出。新規顧客獲得の主要ドライバ。",
            ),
            ParamSpec(
                name="conversion",
                label="広告→顧客 転換係数",
                default=0.5,
                min=0.0,
                description="広告費1単位あたりに獲得できる新規顧客数。",
            ),
            ParamSpec(
                name="churn_rate",
                label="解約率",
                default=0.05,
                min=0.0,
                max=1.0,
                description="毎月失われる売上の割合。",
            ),
        ),
        output_variables=("Sales", "acquisition", "churn_flow"),
    ),
)


_BY_ID: dict[str, ModelSpec] = {m.model_id: m for m in _STARTER_MODELS}


def list_models() -> list[ModelSpec]:
    """Return all catalog models."""
    return list(_STARTER_MODELS)

def get_model(model_id: str) -> ModelSpec | None:
    """Return a catalog model by id, or None if unknown."""
    return _BY_ID.get(model_id)


def catalog_summary() -> list[dict[str, str]]:
    """Compact listing for LLM model-selection prompts."""
    return [
        {"model_id": m.model_id, "name": m.name, "description": m.description}
        for m in _STARTER_MODELS
    ]
