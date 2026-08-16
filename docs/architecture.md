# Architecture

CLAUDE.md から分離した詳細目録。ディレクトリの中身・設定ファイルの規約を知る必要があるときだけ読む。

## Directory

```
neurosurrogate/                  # ドメイン層 (marimo/MLflow 非依存)。依存の向きは core ← neurons ← sim.{catalog,spec,result} ← surrogate ← sim.{run,artifacts} で、**core は他ディレクトリを一切 import しない**。中身の無い `__init__.py` は置かない (再 export だけの層も作らない = 各実体を submodule から直接 import。そのため setuptools は `namespaces = true`)。**`_` 始まりのファイル名 = そのパッケージの外から import しない** (実測で内部専用のものだけが `_` を持つ)
  __init__.py                    # jax_enable_x64 を強制 ON
  core/  network.py              # Compartment/CompartmentType/NeuronGraph + DatasetConfig (**実体化済みの実行入力** = dt/net/current の 3 つだけ。名前の解決も JSON 往復も持たない)
         simulator.py            # unified_simulator (JAX Euler + lax.scan)
         coords.py               # xarray 座標の write 側 + transform_gate
         access.py               # 同スキーマの read 規約 (生 sel はここ以外で使わない)
         opcost.py               # OpCost 代数 (演算コスト集計)
  neurons/  generate.py          # **作り方だけ** (chain / build_traub19)。組んだ結果のカタログは持たない
            traub19.py           # 19-comp モデルの per-comp 定数 + ヘルパ (traub.c 代数的等価)
            compartments/        # hh.py / traub.py / _common.py + COMPARTMENT_TEMPLATES
  sim/  catalog/                 # **名前 → 実体の対応表だけ** (SimSpec のフィールドが引く選択肢。作り方は持たない)
          targets.py             # MCMODELS (SimSpec.target が引く適用先モデル)
          currents.py            # 注入電流波形 + CURRENT_MAP
        spec.py                  # **実験の記述だけ** (実行も結果も置換器も知らない): SimSpec (**唯一の仕様型**: 適用先 target × 電流。学習データの指定も評価条件もこれ 1 つ) + materialize (仕様 → DatasetConfig。名前 → 実体の解決はここだけ) + EvalSeries (spec+param+values の 1 掃引 = **保存の単位でもある**。points は派生、to_dict/from_dict/hash で往復)。surrogate より前の層 (SurrogateMeta.dataset がこれ)
        result.py                # **結果の器だけ** (計算も描画もしない): SimResult (spec+波形+axis のみ。識別も保存先 id も持たない) / attach (保存済み波形列 → 点列。再シミュ無しの run_points) / SeriesResults (点軸×run 軸に開いた結果の純粋なデータ。net/target/axis/values は点から読むだけ。**系列名も評価 run の id も持たない** = 描画層はこれだけ見る)
        run.py                   # **仕様 × surrogate を掛ける唯一の段** (marimo/mlflow 非依存。**何を**回すかは scripts/catalog.py): simulate (1 シミュ) / run_points (掃引点列 → 結果。系列 → 結果の唯一の入口) / replaceable / **run 軸** = replaced_runs (1 系列 × 学習 run → 置換できる run だけに絞る)。表示名も MLflow の id も出てこない。surrogate の後の層
        artifacts.py             # 結果 (`sim.result.SeriesResults`) → **単一 Artifact**。run 横断の summary/traces/metric を個別関数で生成。**点軸×run 軸の並びを成果物へ落とす唯一の場所**。成果物列の編成と詳細点の段付けは `artifact.bundle`
  surrogate/  meta.py            # SurrogateMeta (学習構造の単一源)
              bundle.py          # SurrogateBundle.setup/load/save + SURR_CLS/PREPROCESSOR_CLS
              replace.py         # 置換可否判定 + apply_surrogate
              ansatz/            # base.py + impl/{sindy,hybrid,hybrid_kernel,ude,_sindy_fit}.py
              closure/           # base.py / ude.py / sindy/{__init__,roles,entry,_catalog}.py
              preprocessor/      # base.py + impl/{pca,autoencoder}.py
              artifacts/         # surrogate の自己記述成果物を **単一 Artifact** ずつ返す。train.py=学習データ / model.py=neurograph・SINDy 係数・PCA scree
  artifact/  model.py            # Artifact (名前 + Figure/DataFrame) の運搬形
             plotting.py         # matplotlib 描画プリミティブと共通 style。ドメイン知識を持たない
             bundle.py           # **成果物編成の唯一の seam**。sim/waveform/surrogate の単一 Artifact を list[Artifact] に束ね、点の段名を付ける。描画失敗は変換せず呼び出し元へ伝播。scripts の描画生成はこの module だけを参照
  waveform/                      # 波形ドメイン: 常に 1 ペア (原系, 置換系) だけを見る (点軸も run 軸も持たない)
            dynamics.py          # DynamicMetrics + eFEL/波形誤差の計算 (素の値のみ)
            _tables.py           # その値を表に並べる (計算を増やさない)
            artifacts.py         # current/diff/simple/attractor/metrics を単一 Artifact として返す
scripts/  main.py                # Hydra エントリ
          tuning.py              # 描画への**入力**: Tuning (描き方の全キーの単一源。値は marimo の widget 1 箇所、束を解いて描画層へ渡すのは mlflow_io の各 module)
          catalog.py             # **何を回すか**の 1 枚カタログ: EVALS (素材 1 条件) / SERIES (掃引。置換器を持たない素の EvalSeries。回す側が sim.run.replaced_runs で run 軸を張る)。**描き方は持たない** (tuning.Tuning は marimo の widget が全キーを持つ)
          mlflow_io/             # MLflow I/O = **MLflow を知る唯一の場所**。experiment ごとに 1 module で、どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く / **その experiment に属する成果物を組む**」。**再 export しない** (呼ぶ側は from mlflow_io.report import ... と名乗る)
            __init__.py          # tracking URI をリポジトリ直下へ固定 (import 時に実行 = どの module を通っても最初に張られる) + TARGET_EXP
            save.py              # `report.render_report` が使う保存 module。成果物の MLflow 書込、run 名による段付け、Tuning の draw.json 保存を担う
            surrogate.py         # 学習 experiment: surrogate pickle/meta の読み書き (run_id ごとに @cache) + get_runs_df (run 一覧。読込不可 run はここで落とす) + sweep_siblings。モデル成果物生成は `render_report` の内部実装
            series.py            # 波形 experiment eval_series (**1 run = 1 EvalSeries** = 点列の波形 1 artifact。kind=original / kind=surrogate がフラットに並び、置換系は tags.original_hash で原系を名指す = 親子関係なし)。run_series は探索と実行が対 (決定的だから同じ入力は回さない) なので分けない
            report.py            # レポート experiment eval_report (**1 run = 1 レポート = 1 系列 × N モデル**)。run_and_log / find_report_run / load_report (→ **Report** = 描く中身 `SeriesResults` + 由来の run id。**MLflow の id を持つのはこの型だけ**でドメイン層は id を知らない) + render_report (**描画の唯一の interface** = report_run_id + Tuning。参照解決、bundle ロード、全成果物生成、その run への保存を隠す)
          marimo.py              # notebook 本体 (run 選択 + 系列 dropdown 1 件 + 評価/描画ボタン。組立は neurosurrogate 側の関数呼び出しのみ)
          conf/                  # 学習設定 (Hydra) のみ。下記「設定ファイル」参照
tests/    conftest.py (headless 化 + scripts/ を import path へ) / test_surrogate.py / test_inits.py / test_eval_mlflow.py (評価 run の保存/読込。tracking 先は tmp へ差し替え)
docs/     poster/ slide/         # typst
```

## 成果物の置き場 (MLflow)

描画ボタンが書く先は**そのレポート run の artifact** で、ローカルの `results/` は使わない
(リポジトリ直下に残っているのは移行前の置き土産)。**保存名は選ばせない**:

```
<レポート run>/
  draw.json                     # そのとき使った Tuning (描画 1 回につき 1 枚)
  traces.png metric.png         # run 横断 = この選択でしか出ない図
  summary.csv
  models/<表示名>/               # 比べた 1 本ずつの自己記述図
  series/original/              # 原系の入力電流
  series/<表示名>/p<点>/          # 置換系ごとの詳細図
```

- **束ねる単位がレポートなのは、欲しいものが「N 本のモデルを比べた結果」そのものだから。**
  1 レポート = 1 系列 × N モデルが、そのまま run 1 本に閉じる
- **記録した run を描画が書き換えない**: 学習 run にも波形 run にも書かない
  (それらは fit / 評価の記録のまま)
- 段の名前が run id でなく表示名 (`meta.label`) なのは、宛先が 1 run で衝突しないから
  = 凡例と同じ読み方で段を引ける
- 成果物ごとの由来 (sources) は持たない — どの run から読んだかはレポート run の tag
  (`original_series_id` / `surrogate_series_ids`) が既に指している
- 描き直しは同じ path を置き換える = レポート run は**最後に描いたものだけ**を持つ

## 設定ファイル

- `scripts/conf/config.yaml` + `surrogate/<preset>.yaml` — 学習設定。`surrogate` 直下は
  `meta` / `preprocessor` / `ansatz` の 3 ブロックで `SurrogateBundle.setup` の宛先と 1 対 1
  (`meta.datasets` は `SimSpec` のフィールドそのもの = target/current_type/dt/current_params)。
  `_test_*.yaml` はテスト専用 preset (tests は preset 名を指すだけ)。
- 評価条件は設定ファイルを持たない → `scripts/catalog.py` に型のまま並ぶ
  (`EVALS` / `SERIES`)。スキーマという型の弱い写しを二重に
  管理せず、綴り間違いは import 時に落ちる。**どの系列を回すか**は marimo の系列
  dropdown で **1 件**(選択肢 = 選択 run で置換できる系列)。1 レポート = 1 系列 なので、
  系列を選ぶことが「どのレポートを作る / 描くか」の選択そのもの。実験条件は滅多に変わらず、
  変えたら別の実験 = コードに焼いて差分に出す方が正しい。
- **描き方 (`tuning.Tuning`) はカタログに持たない** → 全キー (比較対象 comp・
  全 comp 図の表示制限・点軸の指標・詳細図の点 index・スパイク番号・折れ線の y レンジ)
  を marimo の widget が持つ。どれも図を見て決め直すもので、カタログに置くと「何を
  回すか」と同じ寿命に見えてしまう。comp の選択肢は選んだ 1 系列の適用先の comp 名
  (`SimSpec.net` が解く) なので、適用先と噛み合わない comp を選べない。
  **描画の入力はレポート run_id 1 つ + `Tuning` だけ** (描く側は「どう回したか」を
  再構成しない)。指標の
  選択肢は `waveform.dynamics.METRIC_KEYS` (取り出せるキーの単一源)。**何の図を出すかはどこにも
  書かない**: モデル側は run 自身が描けるもの (`artifact.bundle.surrogate_artifacts` が
  bundle の型から解く = SINDy なら ξ heatmap、PCA なら scree、固有図を持たない表現は
  何も出さない)、評価側は結果の形 (点が 2 つ以上なら点軸の折れ線が出る) が決める。
  図の見た目 (rcParams) は `plotting.RC_PARAMS` の 1 組だけで、切り替え機構は持たない。
