# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

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

- 概念の説明や教材。この領域からは参照しない。
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**教材と研究コードは互いを見ないまま独立に保ち、同じ内容の二重実装を許す。**

## Coding Standards

- 一時変数は同じ値を何度も使うときだけ許す（NG: `x = obj.attr; f(x)` / OK: `f(obj.attr)`）
- `__init__.py` に `__all__` を定義しない。過剰な複雑さになる
- `_` 始まりの名前とモジュール名は「外から参照しない」印。外から使うものに付けない
- 大きな改装のあとは `just test` が通ることを確認する。テストは自由に足してよいが 20s 以下に抑える
- Hooks で走る `just lint`、`just format` のエラーは都度対処する
- 研究のまとめは `docs/poster/`、`docs/slide/` に Typst で置く

## Commands

```bash
uv sync                                      # 依存導入
uv run scripts/main.py                       # fit + MLflow log のみ (kernel は回さない)
uv run scripts/main.py surrogate=_hh_informed  # Hydra プリセット切替
uv run scripts/main.py --multirun            # preset の hydra.sweeper.params 直積 sweep
just test                  # pytest + main.py
just format && just lint   # ruff / ruff + mypy (strict、scripts/ 除外)
just mlflow                # MLflow UI (port 5100)
just marimo                # marimo notebook (port 2700。CLI は持たず二重管理を避ける)
just marimo-mcp            # Claude Code MCP 連携 (port 2701)
just traub                 # traub_* preset を順に --multirun
just clean-cache / clean-log
just clean-run / clean-test # MLflow run 全削除 / smoke_test experiment のみ削除
```

## Architecture

- 依存の向き: `core ← neurons ← sim.{_current_catalog,spec,result} ← surrogate ← sim.{run,artifacts} ← artifact.bundle`
- `neurosurrogate/` はドメイン層で、marimo/MLflow/Hydra を import しない。
- 描画成果物も評価結果本体も MLflow に置く（図はレポート run の artifact）。
- 依存の向き、ドメイン層の import、`__all__` の不在、`_` の綴りは `tests/test_conventions.py` が機械検査する。
- 動的に呼ばれる入口（Hydra entry / marimo app / `neurons/{hh,traub}.py`）はテスト側の免除リストに明記する。
- 検査が落ちたら、テストを緩めずコードを直す。
- 各ディレクトリとファイルの責務、設定ファイルの規約は `docs/architecture.md` が持つ。
- 設計に手を入れる前に `docs/agents/design-principles.md`（どちらへ倒すかの判断基準）を読む。

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

`rtk` 前置は出力を縮めるだけで、終了コードは素通しする。出力が変なときだけ `rtk proxy <cmd>` と比べる。
