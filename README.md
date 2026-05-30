# sys-dyna

システムダイナミクス × LLM 社内分析ツール。
**PySD でシミュレーションを実行し、その結果を Gemini が分析・説明する**機能を、
LangGraph による AI オーケストレーションとして実装する（基本設計書 v2.0: `docs/design_v2.md`）。

## コアフロー

```text
ユーザー入力
  → 意図分類 (Gemini)
  → モデル選択（カタログ / アップロード）
  → パラメータ抽出（自然言語 → 数値）
  → 【HITL】パラメータ確認・修正        ← LangGraph interrupt による人手確認
  → シミュレーション実行（PySD, シナリオ並列比較）
  → 結果を時系列グラフ表示 + Gemini が自然言語で分析
```

例: `広告費を1.2倍と1.5倍で比較したら売上は?` → パラメータ確認フォーム → 実行 →
シナリオ別の売上推移グラフと比較分析が表示されます。

## 高度な分析（シミュレーション・サブエージェント）

単一シナリオの実行に加え、`SimulationAgent`（`src/sys_dyna/agents/`）が 2 つの分析ツールを提供します。
意図分類が `montecarlo` / `optimize` を検出すると、グラフは確認(HITL)を挟んでサブエージェントに委譲します。

- **モンテカルロ分析** (`monte_carlo`): パラメータを確率分布（正規・一様・三角・対数正規）から多数サンプリングして実行し、
  目的指標の平均・パーセンタイル（90% 区間）・入力感度（相関）を集計。「広告費のばらつきリスクは?」等。
- **ベイズ最適化** (`bayesian_optimization`): Optuna (TPE) でパラメータ探索空間を探索し、目的指標を最大化/最小化する
  最適パラメータを少ない試行で発見。「売上を最大化する広告費は?」等。

目的関数は設定可能で、`{variable, aggregate(final/mean/min/max/sum/initial), direction(maximize/minimize)}` で指定します
（`src/sys_dyna/simulation/objective.py`）。プログラムからの利用例:

```python
from sys_dyna.agents import SimulationAgent

agent = SimulationAgent()  # 既定の PySDEngine + カタログ
agent.run("optimize", {
    "model_id": "sales_growth",
    "search_space": [{"name": "ad_spend", "low": 50, "high": 300}],
    "objective": {"variable": "Sales", "direction": "maximize"},
    "n_trials": 30,
})
```

## 構成

| レイヤー | 実装 |
|---|---|
| UI | Streamlit (`app.py`, `src/sys_dyna/ui/`) — チャット + HITL 確認フォーム + グラフ |
| オーケストレーション | LangGraph StateGraph (`src/sys_dyna/graph/`) |
| LLM | Gemini `gemini-3.5-flash` (`graph/gemini_planner.py`)。キー未設定時は `HeuristicPlanner` で動作 |
| シミュレーション | PySD (`src/sys_dyna/simulation/`) + モデルカタログ |
| シミュレーション・サブエージェント | `src/sys_dyna/agents/` — モンテカルロ分析 (numpy) / ベイズ最適化 (Optuna) を実行するツールを束ねた `SimulationAgent` |
| データ / 認証 | Supabase (Postgres / Auth / Storage)。スキーマ: `supabase/migrations/` |

## セットアップ

PySD / LangGraph 等の依存を含むため、クリーンな仮想環境を推奨します。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env   # Gemini / Supabase の値を設定（未設定でもオフライン動作）
```

## 起動

```bash
streamlit run app.py
```

`SYS_DYNA_GEMINI_API_KEY` を設定すると Gemini 分析が有効になります。
未設定の場合はヒューリスティックなオフラインモードで動作します。

## テスト

```bash
pytest
```

## 設計書との対応 (v2.0)

| 要件 | 実装 |
|---|---|
| シミュレーション実行 | `src/sys_dyna/simulation/engine.py` (PySD) |
| モンテカルロ分析 / ベイズ最適化 | `src/sys_dyna/agents/`（`SimulationAgent` + 2 ツール）+ `src/sys_dyna/simulation/analysis/` |
| モデルカタログ | `src/sys_dyna/simulation/catalog.py` + `catalog_models/*.xmile` |
| AI オーケストレーション | `src/sys_dyna/graph/builder.py` (LangGraph) |
| HITL パラメータ確認 | `graph` の `confirm_params`（`interrupt`）+ `ui/param_confirm.py` |
| Gemini 分析 | `src/sys_dyna/graph/gemini_planner.py` |
| シナリオ比較・可視化 | `src/sys_dyna/ui/charts.py` |
| 認証 / 永続化 (Supabase) | `supabase/migrations/0001_init.sql`（RLS）※リポジトリ層/SSO 配線は実装中 |

## 実装状況

- ✅ PySD シミュレーションエンジン + スターターカタログ
- ✅ シミュレーション・サブエージェント（モンテカルロ分析 + ベイズ最適化）と LangGraph 統合（HITL 付き）
- ✅ LangGraph オーケストレーション（HITL 含む）
- ✅ Gemini プランナ（オフラインフォールバック付き）
- ✅ Streamlit UI（HITL フォーム + シナリオ比較グラフ）
- ✅ Supabase Postgres スキーマ + RLS（マイグレーション）
- ⬜ Supabase リポジトリ層（Python）と Google SSO 認証の配線
- ⬜ Postgres チェックポインタ（本番 HITL 永続化）への切替

旧 v1.0（過去セッション参照 / Agentic Search）の SQLite・モック LLM・ツール群は
`src/sys_dyna/{orchestrator,tools,repository}` に残置しており、過去参照は LangGraph の
`past_lookup` 経由で統合予定です。
