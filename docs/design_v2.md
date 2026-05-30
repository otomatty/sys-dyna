# sys-dyna 基本設計書 v2.0

システムダイナミクス × LLM 社内分析ツール。
v1.0 (過去セッション参照 / Agentic Search) のスタブ実装を、**本番想定インフラ + シミュレーション実行 + Gemini 分析 + LangGraph オーケストレーション**へ拡張する。

---

## 1. 目的とコンセプト

ユーザーの自然言語の問い（例: 「広告費を1.5倍にしたら売上はどうなる?」）に対し、

1. システムダイナミクス（SD）モデルを **PySD** で数値シミュレーション実行し、
2. その時系列結果を **Gemini** が解釈・説明し、
3. 必要に応じて **過去の類似分析セッション**を参照して文脈を補強する、

という一連のフローを **LangGraph** によるステートフルな AI オーケストレーションとして実装する。

### v1.0 からの主な変更（確定事項）

| 項目 | v1.0 (スタブ) | v2.0 (本番想定) |
|---|---|---|
| データ基盤 | SQLite | **Supabase (Postgres)** |
| 認証 | 固定ユーザー | **Supabase Auth + Google SSO** |
| LLM | キーワード駆動モック | **Gemini `gemini-3.5-flash` (Google AI Studio API キー)** |
| オーケストレーション | 自前 while ループ | **LangGraph (StateGraph + checkpointer)** |
| 主機能 | 過去セッション参照のみ | **SD シミュレーション実行 + Gemini 分析**（過去参照はツールとして統合） |
| SD エンジン | なし | **PySD** |

---

## 2. 主要機能（コアフロー）

> **シミュレーション実行 → Gemini 分析・説明** をコア体験とする。過去参照は分析の文脈付けに使う補助ツール。

```text
ユーザー入力
  → 意図分類
  → (必要なら) 過去セッション参照
  → モデル選択（カタログ / アップロード / 指定）
  → パラメータ抽出（自然言語 → 数値）
  → 【HITL】パラメータ確認・修正                ← 人間の確認ステップ
  → シミュレーション実行（PySD, シナリオ並列）
  → 結果を Supabase に永続化
  → 時系列グラフ表示 + Gemini による自然言語分析
```

---

## 3. モデル提供の3経路（ユーザー操作の起点）

確定: 以下 3 経路すべてをサポートする。

1. **事前定義モデルカタログ** — 管理者が用意した SD モデル群（売上・在庫・人員 等）から選択し、パラメータ（初期値・係数）だけ調整。最も予測可能でガバナンスしやすい既定経路。
2. **ユーザーが自然言語でパラメータ指定** — 「広告費を1.5倍に」のような表現を Gemini がカタログモデルのパラメータ変更へ変換。経路1と組み合わせ。
3. **ユーザーがモデルファイルをアップロード** — Vensim (`.mdl`) / XMILE (`.xmile`) を PySD で読み込み実行。既存 SD 資産を持つユーザー向け。

> 注: 「LLM が方程式そのものを生成」する経路は **採用しない**（検証・安全性コストのため）。

---

## 4. システム構成

### 4.1 技術スタック

| レイヤー | 採用技術 |
|---|---|
| UI | Streamlit（v1.0 を踏襲・拡張） |
| AI オーケストレーション | **LangGraph** (`StateGraph` + Postgres checkpointer) |
| LLM | **Gemini `gemini-3.5-flash`** via `langchain-google-genai`（Google AI Studio API キー、`SYS_DYNA_GEMINI_MODEL` で切替可） |
| シミュレーション | **PySD**（`.mdl` / `.xmile` ロード、Python モデルも可） |
| DB / Auth / Storage | **Supabase**（Postgres / Auth / Storage） |
| DB アクセス | `supabase-py`（データ操作）+ `langgraph-checkpoint-postgres`（会話状態） |
| 可視化 | Altair / Streamlit native charts |

### 4.2 コンポーネント

```text
app.py                       # Streamlit エントリ（認証ガード + チャット UI）
src/sys_dyna/
  auth/                      # Supabase Auth (Google SSO) 連携
  config.py                  # 設定（Supabase URL/key, Gemini API key/model, 制限値）
  llm/
    gemini_client.py         # langchain-google-genai ラッパ（LLMClient Protocol 維持）
  graph/                     # ★ LangGraph オーケストレーション
    state.py                 # AgentState (TypedDict)
    builder.py               # StateGraph 構築・コンパイル
    nodes/                   # 各ノード実装
  simulation/                # ★ PySD エンジン
    engine.py                # モデルロード・実行・シナリオ並列
    catalog.py               # 事前定義モデルカタログ
    upload.py                # .mdl/.xmile アップロード処理（Supabase Storage）
  tools/                     # 過去参照ツール（v1.0 を LangChain Tool 化して統合）
    query_sessions.py
    get_session_full.py
    get_simulation_results.py
  repository/                # Supabase Postgres リポジトリ層
  ui/                        # chat / sidebar / charts / param_confirm
```

---

## 5. LangGraph オーケストレーション設計

### 5.1 State

```python
class AgentState(TypedDict):
    session_id: str
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Literal["simulate", "past_reference", "followup", "general"] | None
    selected_model: ModelRef | None          # カタログ / アップロード参照
    extracted_params: dict[str, float] | None
    scenarios: list[Scenario]                 # シナリオ比較用（1件以上）
    confirmed: bool                           # HITL 通過フラグ
    simulation_results: list[SimResult]
    past_references: list[dict]
    analysis: str | None
```

### 5.2 グラフ（ノードとエッジ）

| ノード | 役割 |
|---|---|
| `classify_intent` | Gemini でユーザー意図を分類（simulate / past_reference / followup / general） |
| `retrieve_past` | 過去参照ツール群（`query_sessions` → `get_session_full` → `get_simulation_results`）を呼び、文脈を収集 |
| `select_model` | カタログ照合 / アップロードモデル解決 / 直前モデル継続 |
| `extract_params` | 自然言語からパラメータ・シナリオ集合を抽出（モデル仕様をコンテキストに与える） |
| `confirm_params` | **`interrupt()` による HITL**。抽出パラメータ／シナリオをユーザーに提示し、確認・修正を受ける |
| `run_simulation` | PySD でシナリオごとに実行（複数シナリオは並列 / fan-out） |
| `persist` | 実行結果・パラメータを Supabase に保存 |
| `analyze` | Gemini が数値結果を解釈し、要因・ボトルネック・示唆を自然言語で説明 |

```text
START → classify_intent
classify_intent ─┬─(simulate)──────→ select_model → extract_params → confirm_params
                 ├─(past_reference)→ retrieve_past → analyze → END
                 ├─(followup)──────→ extract_params (既存モデル継続)
                 └─(general)───────→ analyze(回答) → END

confirm_params ─(interrupt: 修正)→ extract_params で再提示
confirm_params ─(承認)→ run_simulation → persist → analyze → END
```

- **HITL 実装**: `confirm_params` で LangGraph の `interrupt()` を使い、Streamlit 側でパラメータ確認 UI を出す。ユーザーが「実行」を押すと `Command(resume=...)` でグラフを再開。
- **チェックポインタ**: `PostgresSaver`（Supabase Postgres）を thread_id = session_id で利用し、会話状態と中断状態を永続化。ページ再読込・中断後も再開可能。
- **過去参照の統合**: v1.0 の 3 ツールは `@tool` 化し、`classify_intent`／`extract_params` から呼べるようにする（「過去に似た分析」要求時の文脈収集）。

### 5.3 制限値（v1.0 から継承）

- ツール呼び出し上限、各ツールタイムアウト、ターンタイムアウト（`config.py`）。
- シミュレーション実行のタイムアウト・最大シナリオ数を追加。

---

## 6. シミュレーションエンジン（PySD）

- **ロード**: カタログモデル（リポジトリ同梱の `.py`/`.mdl`/`.xmile`）、またはアップロードモデル（Supabase Storage 経由）を `pysd.read_vensim` / `pysd.read_xmile` でロード。
- **実行**: `model.run(params=..., return_columns=..., initial_condition=...)` で時系列 DataFrame を取得。
- **シナリオ比較**: パラメータ集合（例: 広告費 ×1.2 / ×1.5 / ×2.0）ごとに実行し、結果を束ねて比較表示。
- **出力**: `{variable: [{t, v}, ...]}` 形式に正規化し v1.0 の `simulation_results` スキーマと互換に保つ。

---

## 7. データモデル（Supabase / Postgres）

v1.0 の SQLite スキーマを Postgres へ移行し、SD モデルカタログとシミュレーション実行を拡張。

| テーブル | 内容 | 備考 |
|---|---|---|
| `auth.users` | Supabase Auth 管理 | Google SSO |
| `profiles` | 表示名・部署等のアプリ側ユーザー属性 | `auth.users` と 1:1 |
| `sd_models` | モデルカタログ（名前・種別・スキーマ・ファイル参照） | アップロードは Storage パス |
| `sessions` | 会話セッション | RLS: user_id 単位 |
| `messages` | チャットログ（v1.0 の chat_log を正規化） | |
| `simulation_runs` | 実行単位（モデル・パラメータ・シナリオ） | |
| `simulation_results` | 時系列結果（JSONB） | run に紐付け |
| `tool_call_logs` | ツール利用ログ | v1.0 踏襲 |

- **RLS**: 全テーブルで `user_id = auth.uid()` ベースのポリシーで絞る（`tool_call_logs` / `sessions` も自身のデータのみ参照可。v1.0 の「全社員公開」前提は廃止）。管理者ロールのみ全社横断参照を許可。
- **マイグレーション**: Supabase migration（`supabase/migrations/*.sql`）で管理。型は `generate_typescript_types` 相当で同期。

---

## 8. 認証基盤（提案 → 確定）

**Supabase Auth + Google SSO** を採用。

- **ログイン**: Google OAuth（Workspace アカウント）。`hd`/allowed-domain 制限で自社ドメインのみ許可。
- **セッション**: Supabase が発行する JWT を Streamlit セッションで保持。`auth.uid()` を全リポジトリ呼び出しのスコープに使用。
- **認可**: Postgres RLS でユーザー単位のデータ分離。管理者ロール（カタログ管理）は `profiles.role` で区別。
- **v1.0 互換**: `auth.get_current_user()` の Protocol を維持し、固定ユーザー実装を Supabase 実装へ差し替え。

---

## 9. UI（Streamlit）

- ログインゲート（未認証は Google ログインへ誘導）。
- チャット欄 + 履歴（v1.0 踏襲）。
- **パラメータ確認パネル**（HITL）: 抽出されたモデル・パラメータ・シナリオを編集可能なフォームで提示 → 「実行」で再開。
- **結果ビュー**: 時系列グラフ（シナリオ重ね描き）+ Gemini 分析テキスト。
- サイドバー: ツール/グラフノードの実行トレース（v1.0 のトレース表示を LangGraph イベントに接続）。

---

## 10. 移行・段階リリース

1. **Phase 1 — 基盤移行**: SQLite→Supabase（スキーマ/RLS/マイグレーション）、固定ユーザー→Supabase Auth、モック LLM→Gemini(AI Studio)。既存テストを Postgres 向けに調整。
2. **Phase 2 — シミュレーション**: PySD エンジン + カタログ + アップロード。`run_simulation` 単体動作。
3. **Phase 3 — LangGraph 化**: 自前ループを StateGraph へ置換、HITL（interrupt）導入、過去参照ツール統合。
4. **Phase 4 — 出力強化**: シナリオ比較・グラフ・Gemini 分析・結果永続化。

---

## 11. 未決事項（要確認）

- ~~**Gemini モデル名**~~: **確定** → `gemini-3.5-flash`（`SYS_DYNA_GEMINI_MODEL` で切替可）。
- ~~**ログ共有範囲**~~: **確定** → RLS で `auth.uid()` 単位に絞る。管理者ロールのみ横断参照。
- **カタログモデルの初期ラインナップ**: 未定。実装では下記スターターセットを暫定同梱し、後から差し替え可能にする（§7.1）。
- **アップロードモデルの検証**: `.mdl`/`.xmile` の安全性・サイズ・実行時間上限。
- **シナリオ最大数**・**シミュレーション実行タイムアウト**の既定値。
- **デプロイ先 / ネットワークポリシー**: Streamlit ホスティング先と Supabase / Google AI Studio への送信経路（社内データ外部送信の取り扱い）。
