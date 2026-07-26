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
just marimo                # marimo notebook (port 2700。run選択+実行+artifact保存のみ、描画はしない)
just marimo-mcp            # Claude Code MCP連携 (port 2701)
just draw [dir]            # artifact (results/artifacts/) + conf/draw.json → results/<dir> へ図/表出力
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
  metrics/  spec.py              # EvalSpec/SweepAxis + parse_evals (計算入力のみ)
            eval.py              # EvalGrid (点軸 × run 軸) + evaluate/run_evals
            store.py             # 評価結果 artifact の save/load (results/artifacts/)
            wave.py              # eFEL スパイク + RMSE/MAE + METRIC_KEYS
  view/  engine.py               # 描画プリミティブ (new_figure/draw_engine/collect/error_fig)
         figs/cell.py            # 1 セル (点 × run) の詳細図 + 電流プレビュー
         figs/grid.py            # 点軸メトリクス折れ線 + 波形格子 (行=run / 行=評価)
         figs/model.py           # 静的図 (neurograph/closure/preprocessor)
         figs/train.py           # 学習データ図
         report.py               # 描画宣言 DrawSpec/ResultSpec/ReportSpec/CompareSpec + model/eval グループの組立
         save.py                 # SaveEntry/slug/save_entries (図と表の書き出し)
scripts/  main.py                # Hydra エントリ
          mlflow_io.py           # MLflow I/O (import 時に tracking URI をリポジトリ直下へ固定)
          draw.py                # artifact + conf/draw.json → 図/表の書き出し (CLI と marimo 保存ボタンの共通本体)
          marimo.py              # notebook セル (run 選択 + 実行 + 描画呼び出しのみ。描画自体は draw.py)
          widgets.py             # marimo widget 層 (計算も図の組立も持たない)
          conf/                  # 下記「設定ファイル」参照
tests/    conftest.py / test_surrogate.py / test_inits.py
docs/     poster/ slide/         # typst
results/  <保存名>/               # draw.py が書く図 + meta.json / artifacts/ に評価結果 (計算)
```

## 設定ファイル

- `scripts/conf/config.yaml` + `surrogate/<preset>.yaml` — 学習設定。`surrogate` 直下は
  `meta` / `preprocessor` / `ansatz` の 3 ブロックで `SurrogateBundle.setup` の宛先と 1 対 1。
  `_test_*.yaml` はテスト専用 preset (tests は preset 名を指すだけ)。
- `scripts/conf/eval.json` — シミュ入力のみ (entry の配列。1 entry = target +
  current_type + dt + current_params、掃引したいときだけ `sweep`:{param,start,stop,steps})。
  marimo が入口で `EvalSpec` へ落とし、以降 domain は型でしか受け取らない。
- `scripts/conf/draw.json` — 描画宣言のみ (計算入力と完全分離。artifact が入力仕様を
  自分で持つので描画側は `eval.json` を読まない)。`default` (表示設定の既定) /
  `results` (label ごとの override。`{"eval": label, ...DrawSpec のキー}`) /
  `compare` (既に回した結果を label 参照して 1 枚の格子に並べる)。`ReportSpec.from_dict`
  が唯一の入口で、以降は型 (`DrawSpec`/`ResultSpec`/`CompareSpec`) で渡す。
- `scripts/conf/style/*.mplstyle` — matplotlib スタイル (paper / presentation)。
