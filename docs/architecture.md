# Architecture

CLAUDE.md から分離した詳細目録。ディレクトリの中身・設定ファイルの規約を知る必要があるときだけ読む。

## Directory

```
neurosurrogate/                  # ドメイン層 (marimo/MLflow 非依存)。依存の向きは core ← neurons ← sim.{catalog,spec,result} ← surrogate ← sim.{run,artifacts} ← artifact.bundle (waveform は core だけを見る枝で、合流点が artifact.bundle) で、**core は他ディレクトリを一切 import しない**。中身の無い `__init__.py` は置かない (再 export だけの層も作らない = 各実体を submodule から直接 import。そのため setuptools は `namespaces = true`)。**`_` 始まりのファイル名 = そのパッケージの外から import しない** (実測で内部専用のものだけが `_` を持つ)
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
        spec.py                  # **実験の記述だけ** (実行も結果も置換器も知らない): SimSpec (**唯一の仕様型**: 適用先 target × 電流。学習データの指定も評価条件もこれ 1 つ。**同一性は持たない** — hash は保存の単位だけが持つ) + materialize (仕様 → DatasetConfig。名前 → 実体の解決はここだけ) + EvalSeries (spec+param+values の 1 掃引 = **保存の単位でもある**。points は派生、to_dict/from_dict/hash で往復) + EvalSelection (掃引 × 学習 run 群 = **1 レポートの記述**。hash が選択そのものの同一性)。**記述 3 段が result の 波形 / SeriesRun / SeriesResults と 1 対 1**。surrogate より前の層 (SurrogateMeta.dataset がこれ)
        result.py                # **結果の器だけ** (計算も描画もしない): SeriesRun (**1 列** = 記述 EvalSeries + run_id (None=原系) + 波形 `list[xr.Dataset]`。**キャッシュの単位でも永続化の単位でもある** = mlflow_io.series の 1 run と一致) / SeriesResults (原系 1 列 + 置換系の列 tuple。全列が同じ記述を回したことを構築時に検査。run_id は列が持つので束は id をキーに持たない)。波形を包む型は無い — 点 i の計算入力は `series.points[i]`、対応は list の添字。net/target/axis/values/dt は記述から読む。**系列名も評価 run の id も持たない** = 描画層はこれだけ見る
        run.py                   # **仕様 × surrogate を掛ける唯一の段** (marimo/mlflow 非依存。**何を**回すかは scripts/catalog.py): simulate (1 シミュ → 波形) / run_column (series + run_id + surrogate → **1 列** `SeriesRun`。系列 → 結果の唯一の入口。両方 None が原系) / replaceable / **run 軸** = replaced_runs (1 系列 × 学習 run → 置換できる run だけに絞る)。表示名は出てこない (学習 run の id は列の標識として通るだけ)。surrogate の後の層
        artifacts.py             # 結果 (`sim.result.SeriesResults`) → **単一 Artifact**。run 横断の summary/traces/metric を個別関数で生成。**点軸×run 軸の並びを成果物へ落とす唯一の場所**。成果物列の編成と詳細点の段付けは `artifact.bundle`
  surrogate/  meta.py            # SurrogateMeta (学習構造の単一源)
              bundle.py          # SurrogateBundle.setup/load/save + pathの1区切りとして有効で一意な学習run名・選択順を束ねる SurrogateRuns (MLflow run IDは持たない)
              replace.py         # 置換可否判定 + apply_surrogate
              ansatz/            # base.py + impl/{sindy,hybrid,hybrid_kernel,ude,_sindy_fit}.py
              closure/           # base.py / ude.py / sindy/{__init__,roles,entry,_catalog}.py
              preprocessor/      # base.py + impl/{pca,autoencoder}.py
              artifacts/         # surrogate の自己記述成果物を **単一 Artifact** ずつ返す。train.py=学習データ / model.py=neurograph・SINDy 係数・PCA scree + 表現の型で振り分ける closure_artifact / preprocessor_artifact (対応する図が無ければ None)
  artifact/                      # `core` 同様に他ディレクトリを import しない基盤 (model.py / plotting.py)。bundle.py だけが合流点
             model.py            # Artifact / Artifacts / Tuning / Report。Report.save(path) が段ごとの PNG/CSV と draw.json を保存 (MLflow 非依存)
             plotting.py         # matplotlib 描画プリミティブと共通 style。ドメイン知識を持たない
             bundle.py           # **成果物編成の唯一の seam**。sim/waveform/surrogate の単一 Artifact を Artifacts に束ね、それを Report の段へ開く (build_report = 直下 / models/<run名>/ / series/<run名>/ の 3 段)。描画失敗は変換せず呼び出し元へ伝播
  waveform/                      # 波形ドメイン: 常に 1 ペア (原系, 置換系) だけを見る (点軸も run 軸も持たない)
            dynamics.py          # DynamicMetrics + eFEL/波形誤差の計算 (素の値のみ)
            _tables.py           # その値を表に並べる (計算を増やさない)
            artifacts.py         # current/diff/simple/attractor/metrics を単一 Artifact として返す
scripts/  main.py                # Hydra エントリ
          catalog.py             # **何を回すか**の 1 枚カタログ: EVALS (素材 1 条件) / SERIES (掃引。置換器を持たない素の EvalSeries。回す側が sim.run.replaced_runs で run 軸を張る) + comp_names (系列名 → その適用先の comp 名 = つまみの選択肢)。**描き方は持たない** (artifact.model.Tuning は marimo の widget が全キーを持つ)
          mlflow_io/             # MLflow I/O = **experiment と run を知る唯一の場所**。experiment ごとに 1 module (学習だけ成果物 surrogate.py と選択肢 runs.py の 2 つ) で、どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」。公開名は run_* (確保) / load_* (読み) / find_* (回さない問い合わせ) で揃える。レポート成果物の書き出しも report.py に閉じる。**再 export しない** (呼ぶ側は from mlflow_io.report import ... と名乗る)
            __init__.py          # tracking URI をリポジトリ直下へ固定 (import 時に実行 = どの module を通っても最初に張られる) + TARGET_EXP
            _query.py            # experiment id の解決 (`exp_id`。**書く側だけが作る**) と同一性 tag での最新 run 引き (`latest_by_tag`。**読む経路は experiment を作らない**)。4 点セットのうち experiment ごとに違わない 2 つをここに 1 つ置く (パッケージ外からは import しない)
            surrogate.py         # 学習 experiment の**成果物**: surrogate pickle/meta の読み書き (run_id ごとに @cache) + load_surrogate_runs (選んだ run 列 → SurrogateRuns。**選択を広げも縮めもしない** = 選択がそのまま run 軸)。どの run が居るか・選べるかは知らない。モデル成果物生成は artifact.bundle の関心
            runs.py              # 学習 experiment の**一覧と選択肢**: load_runs (run 表。読込不可 run はここで落とす) / find_selectable_runs (選んだ系列を置換できる run = run 表の中身。hydra の親子は見ない = sweep の 1 点も単独で選べる)。**選択肢の導出はここ** = UI は選択の結果を渡すだけ。surrogate の中身は見ず、読込可否に surrogate.load_meta だけ借りる (依存は runs → surrogate の一方向)
            series.py            # 波形 experiment eval_series (**1 run = 1 `sim.result.SeriesRun`** = 1 列。点列の波形 1 artifact。kind=original / kind=surrogate がフラットに並び、置換系は tags.original_hash で原系を名指す = 親子関係なし)。run_series は探索と実行が対 (決定的だから同じ入力は回さない) なので分けない。load_column が run → SeriesRun の唯一の読み口 (記述も run_id も一緒に戻る)
            report.py            # レポート experiment eval_report (**1 run = 1 レポート = 1 系列 × N モデル**)。run_report / find_report_run (どちらも系列名 × 学習 run 群で引く = 呼ぶ側はカタログを触らず、EvalSelection の組立はここに閉じる) / load_report (→ 描く中身 `SeriesResults` だけ) / log_report_artifacts (**描画の唯一の interface** = report_run_id + Tuning。参照解決 → artifact.bundleで編成 → Report保存 → 同じrunへの書き出しを隠す)
          marimo.py              # notebook 本体 (系列 dropdown 1 件 + preset 絞り → run 選択 + 評価/描画ボタン)。**セルに置くのは widget と、それを plain 値に均す 1 行だけ** — 選択肢の導出も選択の広げ方も呼び先の 1 関数が持つ
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
  models/<MLflow run名>/          # 比べた 1 本ずつの自己記述図
  series/original/              # 原系の入力電流
  series/<MLflow run名>/           # 置換系ごとの詳細図。点は draw.json の detail_point
```

- **束ねる単位がレポートなのは、欲しいものが「N 本のモデルを比べた結果」そのものだから。**
  1 レポート = 1 系列 × N モデルが、そのまま run 1 本に閉じる
- **記録した run を描画が書き換えない**: 学習 run にも波形 run にも書かない
  (それらは fit / 評価の記録のまま)
- 段の名前は path 安全化した MLflow run name。安全化後に重複する場合だけ完全な
  run id を添え、MLflow UI から元の学習 run を辿れて段も衝突しないようにする
- 成果物ごとの由来 (sources) は持たない — どの run から読んだかはレポート run の tag
  (`original_series_id` / `surrogate_series_ids`) が既に指している
- 描き直しは同じ path を置き換える。今回生成しなかった過去の path は残る

## 設定ファイル

- `scripts/conf/config.yaml` + `surrogate/<preset>.yaml` — 学習設定。`surrogate` 直下は
  `meta` / `preprocessor` / `ansatz` の 3 ブロックで `SurrogateBundle.setup` の宛先と 1 対 1
  (`meta.datasets` は `SimSpec` のフィールドそのもの = target/current_type/dt/current_params)。
  `_test_*.yaml` はテスト専用 preset (tests は preset 名を指すだけ)。
- 評価条件は設定ファイルを持たない → `scripts/catalog.py` に型のまま並ぶ
  (`EVALS` / `SERIES`)。スキーマという型の弱い写しを二重に
  管理せず、綴り間違いは import 時に落ちる。**どの系列を回すか**は marimo の系列
  dropdown で **1 件**。系列は run に依存しない**選択の起点**で、逆に選んだ系列を
  置換できる run だけが run 表に出る。1 レポート = 1 系列 なので、
  系列を選ぶことが「どのレポートを作る / 描くか」の選択そのもの。実験条件は滅多に変わらず、
  変えたら別の実験 = コードに焼いて差分に出す方が正しい。
- **描き方 (`artifact.model.Tuning`) はカタログに持たない** → 全キー (比較対象 comp・
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
