# AGENTS.md

このリポジトリで作業するコーディングエージェント向けのガイダンス。
特定のツールに依存しない共通の記述をここに置く (`CLAUDE.md` は本ファイルへの symlink)。

## Coding Standards [WRITTEN BY HUMAN - DO NOT OVERWRITE]
- HH型マルチコンパートメントニューロンの一部ノードをSINDyで抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード
- 一時変数は同じ値を何度も使うような場合にのみ許可
  - NG: `x = obj.attr; f(x)`
  - OK: `f(obj.attr)`
- 大きな改装が終わったら、just test でエラーが出ないことを確認,tests/ 以下のテストは自由に追加して良い ただし、20s以下に抑えること
- Hooksで実行されるjust lint、just formatのエラーは都度対処すること
- 研究のまとめは、docs/poster、docs/slideディレクトリ以下にtypstとしてまとめる
- __init__.pyに__all__フィールドは定義しないこと　過剰な複雑さ
- `_` 始まりのモジュール名は「そのパッケージの外から import しない」印　外から使うものに `_` を付けない
[以下のセクションは永続メモリとして上書きしても構いません。
ただし、基本的なコマンドやディレクトリ構成などの目録のみを記述すること]

## Commands

```bash
uv sync                                                           # 初期セットアップ（依存導入）
uv run scripts/main.py                                            # 実行 (fit+MLflow log のみ。kernel は回さない)
uv run scripts/main.py surrogate=_hh_informed                     # Hydraプリセット切替 (素体 base/hh/traub/traub19、lib違いは _hh_informed/_hh_relaxation 等)
uv run scripts/main.py --multirun                                 # preset の hydra.sweeper.params 直積 sweep (例 hh: n_components{1,2} × preprocessor{pca,ae} = 4 run)
just test                  # pytest (tests/、Hydraプリセット読込→fit→置換シミュ→指標/描画) + main.py
just format && just lint   # ruff fix+format / ruff+mypy (strict、scripts/ 除外)
just mlflow                # MLflow UI (port 5100、backend: mlflow.db)
just marimo                # marimo notebook (port 2700。run選択+レポートボタン1つ (評価→描画)。CLIは持たず二重管理を避ける)
just marimo-mcp            # Claude Code MCP連携 (port 2701)
just traub                 # traub_* preset を順に --multirun 一括実行
just clean-cache / clean-log
just clean-run / clean-test # MLflow run 全削除 / smoke_test experiment のみ削除 (本番 run 不変)
```

## Architecture

依存の向き: `core ← neurons ← sim.{catalog,spec,result} ← surrogate ← sim.{run,artifacts} ← artifact.bundle`
(`core` は他ディレクトリを一切 import しない。詳細は `docs/architecture.md`)。
`neurosurrogate/` = ドメイン層 (marimo/MLflow 非依存)、`scripts/` = Hydra/MLflow/marimo の入口、
描画成果物も評価結果本体も MLflow (図はレポート run の artifact)。

公開範囲の綴りは機械検査される (`tests/test_conventions.py`): **module 直下の名前は他 module から
参照されるものだけが `_` 無し**、自分の module 内でしか使わないものは `_` 始まり。動的に呼ばれる
入口 (Hydra entry / marimo app / `vars()` ごと注入する `compartments/{hh,traub}.py`) はテスト側の
免除リストに明記する。

各ディレクトリの責務・ファイル単位の役割・設定ファイル (`scripts/conf/`, `scripts/catalog.py`) の規約は
**`docs/architecture.md`** に分離。コード配置や設定の詳細が要るときにそれを読む。

## Agent skills

### Issue tracker

GitHub Issues (`gh` CLI)。See `docs/agents/issue-tracker.md`.

### Triage labels

デフォルト5ラベル (`needs-triage`/`needs-info`/`ready-for-agent`/`ready-for-human`/`wontfix`)。See `docs/agents/triage-labels.md`.

### Domain docs

single-context (`CONTEXT.md` + `docs/adr/`、未作成)。See `docs/agents/domain.md`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
