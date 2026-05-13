# sys-dyna

システムダイナミクス × LLM 社内分析ツール。
基本設計書 v1.0 (`過去セッション参照機能 / Agentic Search`) の実装。

## 構成

- Streamlit チャット UI (`app.py`)
- Agentic Search オーケストレータ (`src/sys_dyna/orchestrator/`)
- 3 つのツール: `query_sessions`, `get_session_full`, `get_simulation_results` (`src/sys_dyna/tools/`)
- `LLMClient` 抽象 + キーワード駆動の `MockGeminiClient` (`src/sys_dyna/llm/`)
- リポジトリ層 (`src/sys_dyna/repository/`) と SQLite スキーマ (`src/sys_dyna/db/schema.sql`)

設計書の Snowflake / Gemini API / 共通認証基盤はそれぞれ SQLite / モック LLM / 固定ユーザーに置き換えてある。
インタフェースはそのまま維持しているので、後から実装を差し替えられる。

## セットアップ

```bash
pip install -e .[dev]
cp .env.example .env
python -m scripts.init_db
```

## 起動

```bash
streamlit run app.py
```

例: `広告費を1.5倍にしたら売上はどうなる? 過去に似た分析あった?`
と入力すると、サイドバーに 3 ツールの呼び出しトレースが表示される。

## テスト

```bash
pytest
```

## 設計書との対応

| 要件 | 実装 |
|---|---|
| F-01 Agentic Search エンジン | `src/sys_dyna/orchestrator/agentic_search.py` |
| F-02 `query_sessions` | `src/sys_dyna/tools/query_sessions.py` |
| F-03 `get_session_full` | `src/sys_dyna/tools/get_session_full.py` |
| F-04 `get_simulation_results` | `src/sys_dyna/tools/get_simulation_results.py` |
| F-05 ツール呼び出し制御 | オーケストレータの `max_tool_calls` / per-tool / turn timeout |
| F-06 セッション永続化 | `src/sys_dyna/repository/sessions.py` + `app.py` の毎ターン upsert |
| F-07 利用ログ収集 | `src/sys_dyna/repository/tool_call_logs.py` + オーケストレータでの記録 |

詳細は `/root/.claude/plans/root-claude-uploads-502f27c0-fb3b-4ad5-sparkling-map.md` を参照。
