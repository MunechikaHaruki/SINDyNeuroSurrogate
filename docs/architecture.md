# Architecture

CLAUDE.md から分離した詳細目録。ディレクトリの中身・設定ファイルの規約を知る必要があるときだけ読む。

## Directory

```
neurosurrogate/                  # ドメイン層 (marimo/MLflow 非依存)。依存の向きは core ← neurons ← sim.{catalog,spec,eval} ← surrogate ← sim.report で、**core は他ディレクトリを一切 import しない**。中身の無い `__init__.py` は置かない (再 export だけの層も作らない = 各実体を submodule から直接 import。そのため setuptools は `namespaces = true`)。**`_` 始まりのファイル名 = そのパッケージの外から import しない** (実測で内部専用のものだけが `_` を持つ)
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
        spec.py                  # SimSpec (**唯一の仕様型**: 適用先 target × 電流。学習データの指定も評価条件もこれ 1 つ) + materialize (仕様 → DatasetConfig。名前 → 実体の解決はここだけ)。surrogate より前の層 (SurrogateMeta.dataset がこれ)
        eval.py                  # **どう**回すか (marimo/mlflow 非依存。**何を**回すかは scripts/catalog.py): simulate → SimResult (spec+波形+axis のみ。識別も保存先 id も持たない) + EvalSeries (spec+param+values+surrogate の 1 掃引実験 = **保存の単位でもある**。points は派生、simulate() は引数なし、attach() で保存済み波形を点列へ貼り直す) + **run 軸** = replaced_runs (1 系列 × 学習 run → run_id → 置換系。run 軸を掛ける唯一の場所) / SeriesResults (点軸×run 軸に開いた結果の純粋なデータ。net/target/axis/values は点から読むだけ。**系列名も評価 run の id も持たない** = 描画層はこれだけ見る)。表示名も MLflow の id も出てこない。surrogate の後の層
        report/              # 結果ドメイン: 軸に開いて報告へ畳む。**1 レポート = 1 系列 × N モデル** (1 系列の電流たちで N 本の surrogate を比べる = dest がレポートの root)。**結果 (`sim.eval.SeriesResults`) を描く層なので sim の下**
            series.py            # 波形 1 本で決まる図: original_figs (原系の入力電流) / detail_figs (選択点 × **1 モデル**の詳細図)。run 横断でない = 別のレポートで見ても同じ図
            report.py            # run 横断の図: summary_figs (比べた N 本のサマリ表。由来は学習 run 群) / wave_report_figs (波形格子 (行=run、列=点。行見出しが run の表示名) + 点軸メトリクスの折れ線。由来は読んだ波形 run) + run_names (label 衝突に連番 = run 軸の本数を知らないと決まらないので run 横断の図と同居)。**点軸×run 軸の並びを図表に落とす唯一の場所**
                                 # 2 module は互いを import せず、どれも `list[Artifact]` を返すだけ = **どの関数を呼んだかが保存段を決める** (図は属する run も由来も名乗らない)。描き方も束 (`Tuning`) では受けず素の引数で受ける。**学習 run 1 本の自己記述図はここに無い** (置換シミュの結果が要らない = `surrogate.figures.surrogate_figs`)
  surrogate/  meta.py            # SurrogateMeta (学習構造の単一源)
              bundle.py          # SurrogateBundle.setup/load/save + SURR_CLS/PREPROCESSOR_CLS
              replace.py         # 置換可否判定 + apply_surrogate
              ansatz/            # base.py + impl/{sindy,hybrid,hybrid_kernel,ude,_sindy_fit}.py
              closure/           # base.py / ude.py / sindy/{__init__,roles,entry,_catalog}.py
              preprocessor/      # base.py + impl/{pca,autoencoder}.py
              figures/           # surrogate の自己記述 (評価結果を受け取らない = 置換シミュ前に描ける)。__init__.py=集約 (surrogate_figs=**run 1 本が自分について描けるもの全部**。適用先も comp 名解決も学習 dataset から解く = 系列も評価 run も要らない。中身は bundle の型が決める = 何を描くかの宣言を受け取らない。内訳 closure_figs/preprocessor_figs/neuron_graph_figs/train_figs。**run 横断の summary_df はレポート側の産物**) / _train.py=学習データ / model.py=neurograph・SINDy 係数・PCA scree
  plotting.py                    # 描画プリミティブ (new_figure/place_legend/error_fig/collect/PanelSpec/TraceSpec/draw_engine/**Artifact** (成果物 1 件 = 名前 + 中身。図を出す全層 (waveform / surrogate.figures / sim.report) の共通の返り値型で、**包み直す層を作らない**)) + RC_PARAMS/use_style (図の見た目の既定。適用は render_report の 1 箇所)。**唯一 機能で切った層** = ドメイン知識を入れない
  waveform/                      # 波形ドメイン: 常に 1 ペア (原系, 置換系) だけを見る (点軸も run 軸も持たない)
            dynamics.py          # DynamicMetrics + eFEL/波形誤差の計算 (素の値のみ)
            _tables.py           # その値を表に並べる (計算を増やさない)
            _figures.py          # 波形/差分/相平面の図 + 電流プレビュー
            __init__.py          # 集約 (cell_figs / wave_report)
scripts/  main.py                # Hydra エントリ
          tuning.py              # 描画への**入力**: Tuning (描き方の全キーの単一源。値は marimo の widget 1 箇所、束を解いて描画層へ渡すのは mlflow_io の各 module)
          catalog.py             # **何を回すか**の 1 枚カタログ: EVALS (素材 1 条件) / SERIES (掃引。surrogate を持たない素の EvalSeries。回す側が with_surrogate して run 軸を張る)。**描き方は持たない** (tuning.Tuning は marimo の widget が全キーを持つ)
          mlflow_io/             # MLflow I/O = **MLflow を知る唯一の場所**。experiment ごとに 1 module で、どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く / **その experiment に属する成果物を組む**」。**再 export しない** (呼ぶ側は from mlflow_io.report import ... と名乗る)
            __init__.py          # tracking URI をリポジトリ直下へ固定 (import 時に実行 = どの module を通っても最初に張られる) + TARGET_EXP
            save.py              # 3 module に共通する保存の部品: run_name (段の名前 = MLflow の run 名。experiment を問わない) + stage (`<段>/<run 名>-<run id 先頭>`。段は必ず run に紐づく) + SaveEntry/slug/save_entries (成果物ごとの由来 sources/draw を meta.json へ。既存 meta.json には合流 = 同じ dest に別系列を描き足しても前の由来が消えない)
            surrogate.py         # 学習 experiment: surrogate pickle/meta の読み書き (run_id ごとに @cache) + get_runs_df (run 一覧。読込不可 run はここで落とす) + sweep_siblings + model_entries (models/<学習 run>/)
            series.py            # 波形 experiment eval_series (**1 run = 1 EvalSeries** = 点列の波形 1 artifact。kind=original / kind=surrogate がフラットに並び、置換系は tags.original_hash で原系を名指す = 親子関係なし)。run_series は探索と実行が対 (決定的だから同じ入力は回さない) なので分けない
            report.py            # レポート experiment eval_report (**1 run = 1 レポート = 1 系列 × N モデル**)。run_and_log / find_report_run / load_report (→ **Report** = 描く中身 `SeriesResults` + 由来の run id。**MLflow の id を持つのはこの型だけ**でドメイン層は id を知らない) + report_entries (report/<レポート run>/) / series_entries (series/<評価 run>/ = 原系 run の current と置換系 run の詳細図。Report が入力なのでレポート側に置く)。marimo は load_report で参照を解決し、3 つの entries 関数へ渡すだけ
          marimo.py              # notebook 本体 (run 選択 + 系列 dropdown 1 件 + 評価/描画ボタン。組立は neurosurrogate 側の関数呼び出しのみ)
          poster_assets.py       # results/<dir> → docs/poster/result へ poster 使用分だけコピー
          conf/                  # 学習設定 (Hydra) のみ。下記「設定ファイル」参照
tests/    conftest.py (headless 化 + scripts/ を import path へ) / test_surrogate.py / test_inits.py / test_eval_mlflow.py (評価 run の保存/読込。tracking 先は tmp へ差し替え)
docs/     poster/ slide/         # typst
results/                         # marimo 描画ボタンの書き先 (**保存名は選ばせない** = ここが全レポート共通の dest)。**MLflow の 3 experiment がそのまま 3 段**: models/<学習 run 名>/ = その run 自身について描けるもの (レポートを増やしても複製されない)、series/<評価 run 名>/ = 波形 1 本だけで決まるもの (原系 run の current、置換系 run の p<点>/ 詳細図 = 別のレポートで見ても同じ場所)、report/<レポート run 名>/ = その 1 レポート (= 1 系列 × N モデル) でしか出ない run 横断の産物 (summary.csv/traces/metric)。段の名前は `<MLflow の run 名を slug 化>-<run id 先頭 8 桁>` = ディレクトリから UI の run を一意に引ける (run 名は人が付け替えられて一意でなく、slug も単射でないので id を混ぜる)。meta.json は dest 直下 1 枚で、描き足すたびに合流。評価結果そのものは MLflow の評価 experiment
```

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
  書かない**: モデル側は run 自身が描けるもの (`surrogate.figures.surrogate_figs` が
  bundle の型から解く = SINDy なら ξ heatmap、PCA なら scree、固有図を持たない表現は
  何も出さない)、評価側は結果の形 (点が 2 つ以上なら点軸の折れ線が出る) が決める。
  図の見た目 (rcParams) は `plotting.RC_PARAMS` の 1 組だけで、切り替え機構は持たない。
