# Architecture

AGENTS.md から分離した詳細目録。ディレクトリの中身・設定ファイルの規約を知る必要があるときだけ読む。

## Directory

```
neurosurrogate/                  # ドメイン層 (MLflow 非依存)。依存の向きは core ← neurons ← sim.{_current_catalog,spec,result,waveform} ← surrogate ← sim.run で、**core は他ディレクトリを一切 import しない**。層の所属と許可は `tests/test_conventions.py` の `_GROUP_OF`/`_LAYERS` が実行可能な形で持つ (ここの記述はその要約。新しいディレクトリを足したらあの表への追記が要る)。中身の無い `__init__.py` は置かない (再 export だけの層も作らない = 各実体を submodule から直接 import。そのため setuptools は `namespaces = true`)。**`_` 始まりのファイル名 = そのパッケージの外から import しない** (実測で内部専用のものだけが `_` を持つ)
  __init__.py                    # jax_enable_x64 を強制 ON
  core/  network.py              # Compartment/CompartmentType/NeuronGraph + DatasetConfig (**実体化済みの実行入力** = dt/net/current の 3 つだけ。名前の解決も JSON 往復も持たない)
         simulator.py            # unified_simulator (JAX Euler + lax.scan)
         coords.py               # xarray 座標の write 側 + transform_gate
         access.py               # 同スキーマの read 規約 (生 sel はここ以外で使わない)
         opcost.py               # OpCost 代数 (演算コスト集計)
         diverge.py              # 置換系の数値的破綻判定 (diverged / log_divergence)。発散ログと指標 (`waveform.compare` の diverged) の両方から呼ばれる共通述語なのでどちらにも属させない
  neurons/  __init__.py          # NeuronGraph の語彙一式: COMPARTMENT_TYPES (型名 → CompartmentType) + _build_traub19 (組み方) + MCMODELS (`SimSpec.target` が引く適用先モデル) + HYBRID_SPLITS (hybrid サロゲートの学習/physics 分割 — 分割位置も残す物理もモデルの性質)。**組み方も組んだ結果もニューロンの語彙**なので、使う側 (sim/surrogate) でなくここが持つ
            traub19.py           # 19-comp モデルの per-comp 定数 + ヘルパ (traub.c 代数的等価)
            hh.py / traub.py     # comp 型の実装 (kernel/コスト/初期値)。`_common.py` は共有のゲート形状
  sim/  _current_catalog.py      # **名前 → 実体の対応表だけ** (SimSpec.current_type が引く選択肢): 注入電流波形 + CURRENT_MAP。sim の内部専用 (適用先モデルの対応表は neurons が持つ)
        spec.py                  # **実験の記述だけ** (実行も結果も置換器も知らない): SimSpec (**唯一の仕様型**: 適用先 target × 電流。学習データの指定も評価条件もこれ 1 つ。**同一性は持たない** — hash は保存の単位だけが持つ) + materialize (仕様 → DatasetConfig。名前 → 実体の解決はここだけ) + EvalSeries (spec+**replace_targets**+param+values の 1 掃引 = **保存の単位でもある**。どこを置換する実験かも記述の一部で、置換器は知らないまま対象を名前で挙げる。points は派生、to_dict/from_dict で往復。鍵は 2 本 — 原系 `hash` (置換範囲を含まない = 範囲だけ違う対照系列と原系 run を共有) と置換系 `replaced_hash` (含む))。**記述 2 段が result の 波形 / SeriesRun と 1 対 1**。surrogate より前の層 (SurrogateSpec.dataset がこれ)
        result.py                # **結果の器だけ** (計算も描画もしない): SeriesRun (**1 列** = 記述 EvalSeries + run_id (None=原系) + 波形 `list[xr.Dataset]`。**キャッシュの単位でも永続化の単位でもある** = mlflow_io._series の 1 run と一致) / SeriesResults (原系 1 列 + 置換系の列 tuple。全列が同じ記述を回したことを構築時に検査。run_id は列が持つので束は id をキーに持たない)。**答えるのは「どの列か・どの点か」だけ** (column/pair/run_ids) = 適用先も掃引軸も刻み幅も素通しせず、要るものは view.series から直接引く。波形を包む型は無い — 点 i の計算入力は `series.points[i]`、対応は list の添字。**系列名も評価 run の id も持たない**
        run.py                   # **仕様 × surrogate を掛ける唯一の段** (mlflow 非依存。**何を**回すかは scripts/catalog.py): _simulate (1 シミュ + 置換対象名 → 波形) / run_column (series + run_id + surrogate → **1 列** `SeriesRun`。系列 → 結果の唯一の入口。両方 None が原系)。**回す前に決まること (どの run が置換できるか) は持たない** → SurrogateRuns.replacing。表示名は出てこない (学習 run の id は列の標識として通るだけ)。surrogate の後の層
        waveform.py              # 波形の指標: 常に 1 ペア (原系, 置換系) だけを見る (点軸も run 軸も持たない)。公開は compare 1 つ = 1 ペア 1 comp の全指標 (rows=(orig, surr) / scalars / spikes=全 AP の特徴量 / diverged) を素の値の dict で返す。並べ方と描き方は持たない (図は live-textbook の冊)
  surrogate/  model.py           # SurrogateSpec (宣言 + hyperparams + JSON 往復 + **設定ツリー (spec + parts 3 ブロック) を解く唯一の場所** from_config + 名前 → 実装の解決 ansatz (**型**) / preprocessor_fit (**fit_pca / fit_ae 関数**) + **学習ドメインの述語** in_train_domain (置換の可否と学習 comp の選定 train_comp_ids の両方が引く = 置換専用でないのでここ)) と Surrogate (**学習済みのもの** = 保存する 3 点 spec/preprocessor/closure だけを持ち (ansatz は spec しか状態を持たないので派生)、答えるのは持つ・保存する・読むと、比べる表に並べる学習側の指標 summary だけ = 適用先を知らない)。**組み立ては型のメソッドでなくモジュール関数** fit_surrogate — hyperparams まで spec が持つので引数は設定ツリー 1 つ。**preprocessor → closure の学習順もここが持つ** (定式化に依らないので Ansatz の既定実装にしない)
              replace.py         # **適用先 (NeuronGraph) を知る唯一の場所** = 置換の語彙一式。**公開は applicable / replace の 2 つだけ** (残りは `_` = 内部の綴り): どこへ当てられるか (applicable ← _rejected_targets) + 当てられない理由の**事実** (_Absent / _TypeMismatch / _ParamsMismatch の直和 = 理由の網羅が型で見える。**文言は持たない** — 文にするのは提示側 _NotReplaceable.__str__ だけ) + 当てること (replace(surrogate, dataset, targets) = 全数検証、1 つでも通らなければ部分適用せず _NotReplaceable)。**判定と適用を分けない** (replace が断る条件 = applicable が偽になる条件そのもの)。型のメソッドでなくモジュール関数 = 適用先はサロゲートの持ち物でない
              runs.py            # 一意な名前と選択順を持つ SurrogateRuns。評価系列が挙げた置換対象を**全部**置換できる run だけへの絞り込み (部分一致は不可)
              parts/  __init__.py # Surrogate が差し替える 3 構成要素の**契約を集約**: Closure / Preprocessor / Ansatz。3 つは互いを参照する (Ansatz が両者を受け、型引数で Closure に束縛) ので契約は 1 モジュール = 抽象レベルのパッケージ間依存辺を持たない。実装は下の 3 パッケージが `from .. import` で引く。対等ではなく closure/preprocessor が leaf、ansatz が両者を合成。**抽象メソッドしか置かない** — 既定実装も受け渡しの型も持たない
                      train_inputs.py # TrainInputs (ansatz が組み closure の同定入口が受ける受け渡しの型)。契約でないので __init__.py に置かず、両側がここを引く
                ansatz/          # sindy.py / hybrid.py / ude.py。hybrid.py が物理骨格と SINDy hybrid を集約。**インスタンスを作らない** = 定式化は規則であって状態でないので全メソッドが classmethod。spec に依るメソッドは spec を引数に取る (`surrogate.ansatz.f(surrogate.spec, ...)`)。定式化だけの性質である params_match は例外
                closure/         # ude.py / sindy/{__init__,roles,_entry,_catalog}.py
                preprocessor/    # pca.py / autoencoder.py / fit_artifacts.py (fit 後に埋まる派生量 = 再構成統計と初期潜在。埋め方は全実装共通なので契約でなくここ)
              artifacts/         # surrogate の自己記述成果物 (置換シミュを回さず描ける = run をロードしただけで出る図。学習のときに学習 run の figures/ へ書く)。`__init__.py` が集合ごと返す `surrogate_artifacts` を持ち、個々の Artifact は submodule が返す (再 export はしない)。train.py=学習データ / _model.py=neurograph・SINDy 係数・PCA scree + 表現の型で振り分ける closure_artifact / preprocessor_artifact (対応する図が無ければ None)
  artifact/                      # `core` 同様に他ディレクトリを import しない基盤 (model.py / plotting.py)
             model.py            # Artifact (**自分を 1 つ書くだけ**の atomic な save。中身が拡張子を決める = 表 CSV / 図 PNG / dict JSON。置き場は知らない) / Artifacts (成果物の集合。save(path) で丸ごとその path へ)
             plotting.py         # matplotlib 描画プリミティブと共通 style。ドメイン知識を持たない
scripts/  main.py                # Hydra エントリ (学習。surrogate と、その run が自分について描ける図 figures/ を log)
          evaluate.py            # 評価の入口 `just eval <系列> <学習 run>...` (argparse)。mlflow_io.evaluation.evaluate を呼ぶだけ
          catalog.py             # **何を回すか**の 1 枚カタログ: EVALS (素材 1 条件) / SERIES (掃引。置換器を持たない素の EvalSeries だが、**どこを置換するかは挙げる** = replace_targets。回す側が SurrogateRuns.replacing で、対象を全部置換できる run だけの run 軸を張る)。**描き方は持たない** (つまみは live-textbook の冊が持つ)
          mlflow_io/             # MLflow I/O = **experiment と run を知る唯一の場所**。experiment ごとに 1 module で、どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」。公開名は run_* (確保) / load_* (読み) / log_* (書き) で揃え、複数段を完遂する evaluation.py だけ evaluate 1 本へ畳む。**再 export しない** (呼ぶ側は from mlflow_io.evaluation import ... と名乗る)
            __init__.py          # tracking URI をリポジトリ直下へ固定 (import 時に実行 = どの module を通っても最初に張られる) + _TARGET_EXP
            _query.py            # experiment id の解決 (`exp_id`。**書く側だけが作る**) と同一性 tag での最新 run 引き (`latest_by_tag`。**読む経路は experiment を作らない**)。4 点セットのうち experiment ごとに違わない 2 つをここに 1 つ置く (パッケージ外からは import しない)
            surrogate.py         # 学習 experiment の**成果物**: surrogate pickle/spec の読み書き (run_id ごとに @cache) + 学習 run の図 (log_surrogate_figures → figures/) + load_surrogate_runs (選んだ run 列 → SurrogateRuns。**選択を広げも縮めもしない** = 選択がそのまま run 軸)。どの run が居るか・選べるかは知らない (選ぶのは冊)
            _series.py           # 波形 experiment eval_series (**1 run = 1 `sim.result.SeriesRun`** = 1 列。kind=original / kind=surrogate がフラットに並び、置換系は tags.original_hash で原系を名指す = 親子関係なし)。波形は TS でも読める形 (下の「成果物の置き場」)。置換系の run は原系と比べた指標・射影・学習側の指標も持つ。run_series は探索と実行が対 (決定的だから同じ入力は回さない) なので分けない。_load_column が run → SeriesRun の唯一の読み口
            evaluation.py        # 評価 = 1 系列を原系と選んだ学習 run で回す (evaluate)。系列を置換できない run は回さない。どの列を並べて見るかは残さない (冊が選ぶ)
          conf/                  # 学習設定 (Hydra) のみ。下記「設定ファイル」参照
tests/    conftest.py (headless 化 + scripts/ を import path へ) / test_surrogate.py / test_inits.py / test_eval_mlflow.py (評価 run の保存/読込。tracking 先は tmp へ差し替え)
docs/     poster/ slide/         # typst
```

## 成果物の置き場 (MLflow)

図は Python で描かず、live-textbook の `research/sindy-surrogate-evaluation` が MLflow の REST から
読んで描く。Python が置くのは図に要るデータだけ:

```
<学習 run>/  surrogate/                spec.json + pickle
             figures/                  学習 run 1 本が自分について描ける図 (学習のときに一度描く)
<波形 run (原系)>/  waves/             下の形式。meta.json に comp 名 → id も入る
<波形 run (置換系)>/ waves/
                     metrics.json      comp 名 → 点ごとの compare (全 comp・全点)
                     summary.json      Surrogate.summary (学習側の指標)
                     projected/<comp>/ 置換した comp の原系ゲートを潜在空間へ射影したもの (waves と同じ形式)

waves/meta.json            t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]], nodes, names
waves/<点>/comp<id>.f32    その comp の変数 (features の並び順)。変数ごとに連続した little-endian float32
waves/<点>/I_ext.f32       waves/<点>/I_internal.f32 (node ごとに連続)
```

- **比べた束 (どの列を並べたか) は MLflow に残さない。** 冊が選び直すたびに描くので、
  束ごとに run を立てると同じ選択が二重になる
- 形式を変えたら `_series._FORMAT` を変える。鍵に混ざるので、古い形式の run は鍵が合わず回し直される
- **記録した run を描画が書き換えない**: 冊は読むだけ
- JSON に nan は書かない (null)。JS の JSON が読めないため

## 設定ファイル

- `scripts/conf/config.yaml` + `surrogate/<preset>.yaml` — 学習設定。`surrogate` 直下は
  `spec` (何を学習するか) + **`parts/` の 3 構成要素と同名のブロック**
  `preprocessor` / `ansatz` / `closure` で、**それぞれその層の入口 1 つの署名へ `**` で
  展開される** (`fit_pca` / `fit_ae` / `HybridAnsatz.split` = physics 分割の選択。
  hybrid 系は `physics_type` 必須で既定を持たない (comp_type 名で代替しない) /
  閉包項の同定入口 = SINDy の `from_sindy`・UDE の joint 学習)。1 ブロック = 1 宛先
  なので、受理するキーと既定値はその署名だけが決め、綴り違いや層違いは TypeError。**既定値は yaml でなく実装側の署名**が単一源 —
  共通ブロックに既定を書くと、その層の実装が違う preset へも黙って継承されるため。
  (`spec.datasets` は `SimSpec` のフィールドそのもの = target/current_type/dt/current_params)。
  `_test_*.yaml` はテスト専用 preset (tests は preset 名を指すだけ)。
- 評価条件は設定ファイルを持たない → `scripts/catalog.py` に型のまま並ぶ
  (`EVALS` / `SERIES`)。スキーマという型の弱い写しを二重に
  管理せず、綴り間違いは import 時に落ちる。**どの系列を回すか**は `just eval` の
  引数で、系列を置換できない学習 run は回さない。実験条件は滅多に変わらず、
  変えたら別の実験 = コードに焼いて差分に出す方が正しい。
- **描き方 (つまみ) はカタログに持たない** → 全キー (比較対象 comp・
  全 comp 図の表示制限・点軸の指標・詳細図の点・スパイク番号・折れ線の y レンジ)
  を live-textbook の冊が持つ。どれも図を見て決め直すもので、カタログに置くと「何を
  回すか」と同じ寿命に見えてしまう。comp の選択肢は原系の `waves/meta.json` の names
  なので、適用先と噛み合わない comp を選べない。**何の図を出すかはどこにも書かない**:
  モデル側は run 自身が描けるもの (`surrogate.artifacts.surrogate_artifacts` が bundle の
  型から解く = SINDy なら ξ heatmap、PCA なら scree)、評価側は結果の形 (点が 2 つ以上なら
  点軸の折れ線が出る) が決める。matplotlib の見た目は `plotting.RC_PARAMS` の 1 組だけ。
