# Architecture

CLAUDE.md から分離した詳細目録。ディレクトリの中身・設定ファイルの規約を知る必要があるときだけ読む。

## Directory

```
neurosurrogate/                  # ドメイン層 (marimo/MLflow 非依存)。依存の向きは core ← neurons ← sim ← surrogate ← report で、**core は他ディレクトリを一切 import しない**。中身の無い `__init__.py` は置かない (再 export だけの層も作らない = 各実体を submodule から直接 import。そのため setuptools は `namespaces = true`)
  __init__.py                    # jax_enable_x64 を強制 ON
  core/  network.py              # Compartment/CompartmentType/NeuronGraph + DatasetConfig (**実体化済みの実行入力** = dt/net/current の 3 つだけ。名前の解決も JSON 往復も持たない)
         simulator.py            # unified_simulator (JAX Euler + lax.scan)
         coords.py               # xarray 座標の write 側 + transform_gate
         access.py               # 同スキーマの read 規約 (生 sel はここ以外で使わない)
         opcost.py               # OpCost 代数 (演算コスト集計)
  neurons/  generate.py          # **作り方だけ** (chain / build_traub19)。組んだ結果のカタログは持たない
            traub19.py           # 19-comp モデルの per-comp 定数 + ヘルパ (traub.c 代数的等価)
            compartments/        # hh.py / traub.py / common.py + COMPARTMENT_TEMPLATES
  sim/  catalog/                 # **名前 → 実体の対応表だけ** (SimSpec のフィールドが引く選択肢。作り方は持たない)
          targets.py             # MCMODELS (SimSpec.target が引く適用先モデル)
          currents.py            # 注入電流波形 + CURRENT_MAP
        spec.py                  # SimSpec (**唯一の仕様型**: 適用先 target × 電流。学習データの指定も評価条件もこれ 1 つ) + materialize (仕様 → DatasetConfig。名前 → 実体の解決はここだけ)。surrogate より前の層 (SurrogateMeta.dataset がこれ)
        eval.py                  # **どう**回すか (marimo/mlflow 非依存。**何を**回すかは scripts/catalog.py): simulate → SimResult (spec+波形+axis のみ。識別も保存先 id も持たない) + EvalSeries (spec+param+values+surrogate の 1 掃引実験 = **保存の単位でもある**。points は派生、simulate() は引数なし、attach() で保存済み波形を点列へ貼り直す)。run_id も表示名も出てこない。surrogate の後の層
  surrogate/  meta.py            # SurrogateMeta (学習構造の単一源)
              bundle.py          # SurrogateBundle.setup/load/save + SURR_CLS/PREPROCESSOR_CLS
              replace.py         # 置換可否判定 + apply_surrogate
              ansatz/            # base.py + impl/{sindy,hybrid,hybrid_kernel,ude,sindy_fit}.py
              closure/           # base.py / ude.py / sindy/{__init__,roles,entry,catalog}.py
              preprocessor/      # base.py + impl/{pca,autoencoder}.py
              figures/           # surrogate の自己記述 (評価結果を受け取らない = 置換シミュ前に描ける)。__init__.py=集約 (surrogate_figs=**run 1 本が自分について描けるもの全部**。中身は bundle の型が決める = 何を描くかの宣言を受け取らない。内訳 summary_df/closure_figs/preprocessor_figs/neuron_graph_figs/train_figs) / train.py=学習データ / model.py=neurograph・SINDy 係数・PCA scree
  plotting.py                    # 描画プリミティブ (new_figure/place_legend/error_fig/collect/PanelSpec/TraceSpec/draw_engine/ArtifactEntries) + RC_PARAMS/use_style (図の見た目の既定。適用は render_report の 1 箇所)。**唯一 機能で切った層** = ドメイン知識を入れない
  waveform/                      # 波形ドメイン: 常に 1 ペア (原系, 置換系) だけを見る (点軸も run 軸も持たない)
            dynamics.py          # DynamicMetrics + eFEL/波形誤差の計算 (素の値のみ)
            tables.py            # その値を表に並べる (計算を増やさない)
            figures.py           # 波形/差分/相平面の図 + 電流プレビュー
            __init__.py          # 集約 (cell_figs / wave_report)
  report/                        # 結果ドメイン: 軸に開いて報告へ畳む。**1 レポート = 1 系列 × N モデル** (1 系列の電流たちで N 本の surrogate を比べる = dest がレポートの root)。**ドメインを横断する唯一の層**
            results.py           # series_matrix (run 軸を掛ける唯一の場所) + SeriesView (1 系列を点軸×run 軸に開いた並び = 1 レポートの単位。束ねる型は持たない) + simulate_views (その場で回す。原系は EvalSeries.hash 単位で共有) + run_names
            grid.py              # 軸に沿った図表: 点軸メトリクスの表と折れ線 + 波形格子 (行=run、列=点)
            save.py              # SaveEntry/slug/save_entries (成果物ごとの由来 sources/draw を meta.json へ)
            build.py             # Tuning (1 レポートの描画条件 1 型: eval_comp/view_comps/metric/detail_point/spike_*/metric_ylim。**図の種類名は持たない**。既定値とキーの単一源はここ、値を与えるのは marimo の widget 1 箇所) + model_entries (run 自身が描けるもの) / eval_entries (結果の形が決めるもの) の組立 + report_entries (1 レポート分の SaveEntry 列を返す唯一の入口。**保存はしない** = どこへ書くかは呼び出し側)
scripts/  main.py                # Hydra エントリ
          catalog.py             # **何を回して何を描くか**の 1 枚カタログ: EVALS (素材 1 条件) / SERIES (掃引。surrogate を持たない素の EvalSeries。回す側が with_surrogate して run 軸を張る)。**描き方は持たない** (report.build.Tuning は marimo の widget が全キーを持つ)
          mlflow_io.py           # MLflow I/O (import 時に tracking URI をリポジトリ直下へ固定)。3 experiment: 学習 (surrogate) / 波形 eval_series (**1 run = 1 EvalSeries** = 点列の波形 1 artifact。kind=original / kind=surrogate がフラットに並び、置換系は tags.original_hash で原系を名指す = 親子関係なし) / レポート eval_report (**1 run = 1 レポート = 1 系列 × N モデル** = 学習 run 群 × 系列 1 つ。持つのは波形 run への参照表 refs.json だけで波形は複製しない。同一性 = 選択そのもの tags.report_hash、同じ選択なら参照表を更新)。**評価 = 1 系列 1 回** (run_and_log は系列 1 つを受けてレポート run_id を返す)、**描画の入力はその run_id 1 つ + Tuning** (report_entries_of(report_run_id, tuning) -> list[SaveEntry] = 波形の参照表も surrogate 本体もここで解決し、組立は report.report_entries へ委譲。marimo は保存先を決めて save_entries に流すだけ)。選択 → 既存レポート run_id の橋は find_report_run 1 本
          marimo.py              # notebook 本体 (run 選択 + 系列 dropdown 1 件 + 評価/描画ボタン。組立は neurosurrogate 側の関数呼び出しのみ)
          poster_assets.py       # results/<dir> → docs/poster/result へ poster 使用分だけコピー
          conf/                  # 学習設定 (Hydra) のみ。下記「設定ファイル」参照
tests/    conftest.py (headless 化 + scripts/ を import path へ) / test_surrogate.py / test_inits.py / test_eval_mlflow.py (評価 run の保存/読込。tracking 先は tmp へ差し替え)
docs/     poster/ slide/         # typst
results/  <保存名>/<系列名>/       # marimo 描画ボタンが書く 1 レポート (= 1 系列 × N モデル) の図 + meta.json。直下に系列共通の図 (current/traces/metric)、<model>/ 以下にモデルごとの自己記述図と選択点の詳細図 (評価結果そのものは MLflow の評価 experiment)
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
- **描き方 (`report.build.Tuning`) はカタログに持たない** → 全キー (比較対象 comp・
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
