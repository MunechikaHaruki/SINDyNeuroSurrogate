# AGENTS.md

このリポジトリで作業するコーディングエージェント向けのガイダンス。
特定のツールに依存しない共通の記述をここに置く (`CLAUDE.md` は本ファイルへの symlink)。
**この領域の入口はこのファイルだけとする。** `README.md` は対外向けの説明、`CONTEXT.md` は
この領域の用語、`docs/` は設計とエージェント運用の詳細であり、いずれも入口ではなく、
必要になった時点でここから参照する。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（ドメイン層） | `neurosurrogate/` |
| Hydra / MLflow / marimo の入口 | `scripts/` |
| 実験の結果と図 | MLflow (`just mlflow`) |
| 研究のまとめ、ポスター、スライド | `docs/poster/`、`docs/slide/` |
| この領域の用語 | `CONTEXT.md` |
| 設計の詳細と判断基準 | `docs/architecture.md`、`docs/agents/` |

## 持たないもの

- 概念の説明や教材。手法の背景を読みたいだけなら、ここのコードではなく教材を見る。
- 修論の計画、期日、週次の進捗、受けた指摘。
- 氏名、学歴、資格などの本人の事実と、進路の判断基準。

概念の説明や教材は、この領域からは参照しない。**教材と研究コードは互いを見ないまま
独立に保ち、同じ内容の二重実装を許す。**

## Coding Standards

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に
置換し、演算コスト削減と波形再現性を評価する研究コードである。

- 一時変数は同じ値を何度も使うときだけ許す（NG: `x = obj.attr; f(x)` / OK: `f(obj.attr)`）
- `__init__.py` に `__all__` を定義しない。過剰な複雑さになる
- `_` 始まりのモジュール名は「そのパッケージの外から import しない」印。外から使うものに付けない
- 大きな改装のあとは `just test` が通ることを確認する。テストは自由に足してよいが 20s 以下に抑える
- Hooks で走る `just lint`、`just format` のエラーは都度対処する
- 研究のまとめは `docs/poster/`、`docs/slide/` に Typst で置く

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

依存の向き: `core ← neurons ← sim.{_current_catalog,spec,result} ← surrogate ← sim.{run,artifacts} ← artifact.bundle`
(`core` は他ディレクトリを一切 import しない。詳細は `docs/architecture.md`)。
`neurosurrogate/` = ドメイン層 (marimo/MLflow 非依存)、`scripts/` = Hydra/MLflow/marimo の入口、
描画成果物も評価結果本体も MLflow (図はレポート run の artifact)。

これらは全部 `tests/test_conventions.py` で**機械検査される** — 依存の向き (層の表 `_LAYERS` が
そのまま実行される)、ドメイン層が marimo/MLflow/Hydra を import しないこと、`__all__` の不在、
そして公開範囲の綴り: **名前も module 名も、外から参照されるものだけが `_` 無し**。動的に呼ばれる
入口 (Hydra entry / marimo app / `vars()` ごと注入する `neurons/{hh,traub}.py`) はテスト側の
免除リストに明記する。落ちたらテストを緩めるのでなくコードを直す。

各ディレクトリの責務・ファイル単位の役割・設定ファイル (`scripts/conf/`, `scripts/catalog.py`) の規約は
**`docs/architecture.md`** に分離。コード配置や設定の詳細が要るときにそれを読む。

## Design principles

リファクタ・設計変更の**判断基準** (抽象を消すか / class を割るか / 中間の型を作らないか、および
リファクタの進め方と止まる条件) は **`docs/agents/design-principles.md`**。設計に手を入れる前に読む。
上の機械検査が「守れているか」を見るのに対し、あれは「どちらへ倒すか」を決める。

## Agent skills

- Issue tracker: GitHub Issues (`gh` CLI)。`docs/agents/issue-tracker.md`
- Triage labels: 5 ラベル (`needs-triage`/`needs-info`/`ready-for-agent`/`ready-for-human`/`wontfix`)。`docs/agents/triage-labels.md`
- Domain docs: single-context (`CONTEXT.md` + `docs/adr/`、未作成)。`docs/agents/domain.md`

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## RTK

シェルコマンドの `rtk` 前置は出力を縮めるだけで**終了コードは素通し** = exit≠0 は rtk 起因でない。
出力が変なときだけ `rtk proxy <cmd>` の素の出力と比べる (コマンド一覧はグローバル設定が持つ)。
