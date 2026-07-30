# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Standards [WRITTEN BY HUMAN - DO NOT OVERWRITE]
- HH型マルチコンパートメントニューロンの一部ノードをSINDyで抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード
- 一時変数は同じ値を何度も使うような場合にのみ許可
  - NG: `x = obj.attr; f(x)`
  - OK: `f(obj.attr)`
- 大きな改装が終わったら、just test でエラーが出ないことを確認,tests/ 以下のテストは自由に追加して良い ただし、20s以下に抑えること
- Hooksで実行されるjust lint、just formatのエラーは都度対処すること
- 研究のまとめは、docs/poster、docs/slideディレクトリ以下にtypstとしてまとめる
- __init__.pyに__all__フィールドは定義しないこと　過剰な複雑さ
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
just marimo                # marimo notebook (port 2700。run選択+評価ボタン+描画ボタン。CLIは持たず二重管理を避ける)
just marimo-mcp            # Claude Code MCP連携 (port 2701)
just traub                 # traub_* preset を順に --multirun 一括実行
just clean-cache / clean-log
just clean-run / clean-test # MLflow run 全削除 / smoke_test experiment のみ削除 (本番 run 不変)
```

## Directory

```
neurosurrogate/                  # ドメイン層 (marimo/MLflow 非依存)
  __init__.py                    # jax_enable_x64 を強制 ON
  core/  network.py              # Compartment/CompartmentType/NeuronGraph/DatasetConfig/chain
         simulator.py            # unified_simulator (JAX Euler + lax.scan)
         coords.py               # xarray 座標の write 側 + transform_gate
         access.py               # 同スキーマの read 規約 (生 sel はここ以外で使わない)
         opcost.py               # OpCost 代数 (演算コスト集計)
  neurons/  __init__.py          # MCMODELS (適用先モデル一覧)
            traub19.py           # 19-comp モデル (traub.c 代数的等価)
            compartments/        # hh.py / traub.py / common.py + COMPARTMENT_TEMPLATES
            currents.py          # 注入電流波形 + CURRENT_MAP
  surrogate/  meta.py            # SurrogateMeta (学習構造の単一源)
              bundle.py          # SurrogateBundle.setup/load/save + SURR_CLS/PREPROCESSOR_CLS
              replace.py         # 置換可否判定 + apply_surrogate
              ansatz/            # base.py + impl/{sindy,hybrid,hybrid_kernel,ude,sindy_fit}.py
              closure/           # base.py / ude.py / sindy/{__init__,roles,entry,catalog}.py
              preprocessor/      # base.py + impl/{pca,autoencoder}.py
  eval/  spec.py                 # SimSpec/SweepAxis + parse_evals (計算入力のみ)
         run.py                  # SimKey/SimResult + expand/simulate/run_results (spec → 結果。永続化は知らない)
  metrics/  select.py            # 結果 (SimKey→SimResult) からの群 (系列名/label/run_id) 選択
            declare.py           # 描画宣言の型 DrawSpec/ReportSpec/CompareSpec (draw.json のスキーマ)
            report.py            # model/eval グループの組立 + render_report (組立→保存まで一括の入口、marimo から呼ぶ)
            save.py              # SaveEntry/slug/save_entries (成果物ごとの由来 sources/draw を meta.json へ)
            artifact/__init__.py # 外部公開 API (Figure/DataFrame を返す集約関数のみ: cell_figs/closure_figs/preprocessor_figs/neuron_graph_figs/train_figs/wave_report + grid.py の re-export)
            artifact/cell.py     # 1 セルの個別図 (diff/simple/attractor) + 電流プレビュー
            artifact/grid.py     # 点軸メトリクス折れ線 + 波形格子 (行=run / 行=評価)
            artifact/model.py    # 静的図の個別生成 (neurograph/closure/preprocessor の中身)
            artifact/train.py    # 学習データの個別図
            artifact/wave_table.py # _internal/wave.py の計算値 → DataFrame 組立
            artifact/_internal/engine.py # 描画プリミティブ (Figure/DataFrame を返さない実装詳細)
            artifact/_internal/wave.py   # DynamicMetrics/diverged (eFEL 計算、Figure/DataFrame を返さない計算層)
scripts/  main.py                # Hydra エントリ
          mlflow_io.py           # MLflow I/O (import 時に tracking URI をリポジトリ直下へ固定)。学習 experiment (surrogate) と評価 experiment (1 run = 1 SimSpec、親=原系/子=置換系、波形 artifact) の両方
          marimo.py              # notebook 本体 (run 選択 + 評価/描画ボタン。組立は neurosurrogate 側の関数呼び出しのみ)
          poster_assets.py       # results/<dir> → docs/poster/result へ poster 使用分だけコピー
          conf/                  # 下記「設定ファイル」参照
tests/    conftest.py / test_surrogate.py / test_inits.py / test_eval_mlflow.py (評価 run の保存/読込。tracking 先は tmp へ差し替え)
docs/     poster/ slide/         # typst
results/  <保存名>/               # marimo 描画ボタンが書く図 + meta.json (評価結果そのものは MLflow の評価 experiment)
```

## 設定ファイル

- `scripts/conf/config.yaml` + `surrogate/<preset>.yaml` — 学習設定。`surrogate` 直下は
  `meta` / `preprocessor` / `ansatz` の 3 ブロックで `SurrogateBundle.setup` の宛先と 1 対 1。
  `_test_*.yaml` はテスト専用 preset (tests は preset 名を指すだけ)。
- `scripts/conf/eval.json` — シミュ入力のみ (entry の配列。1 entry = target +
  current_type + dt + current_params、掃引したいときだけ `sweep`:{param,start,stop,steps})。
  marimo が入口で `EvalSpec` へ落とし、以降 domain は型でしか受け取らない。
- `scripts/conf/draw.json` — 描画宣言のみ (計算入力と完全分離。artifact が入力仕様を
  自分で持つので描画側は `eval.json` を読まない)。`default` (グローバル設定は
  `plt_style` のみ。`eval_comp` 等は適用先ごとに違うので既定を持たせない) /
  `results` (空=手元の結果を全部描く既定。非空なら列挙した label だけへ絞り込み。
  1件 = `{"eval": label, ...DrawSpec のキー}`。override でなく label ごとに完結する
  宣言) / `compare` (既に回した結果を label 参照して 1 枚の格子に並べる。`eval_comp`
  は compare 自身が持つ) / `kinds` (保存する図/表の種類の絞り込み。省略時は全種類。
  種類名は `ReportSpec.ALL_KINDS`)。`ReportSpec.from_dict` が唯一の入口で、以降は型
  (`DrawSpec`/`ResultSpec`/`CompareSpec`) で渡す。
- `scripts/conf/style/*.mplstyle` — matplotlib スタイル (paper / presentation)。

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
