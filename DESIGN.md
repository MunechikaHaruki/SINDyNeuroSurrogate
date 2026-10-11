# 設計

研究コードの設計に手を入れる前に読む。用語、ディレクトリとファイルの責務、どちらへ倒すかの判断基準と、これから広げるときの前提を持つ。

## 用語

**Surrogate scope**:
同じ種類・物理パラメータ両立規則に基づく、サロゲートの学習対象および適用可能対象。
_Avoid_: Train source, replaceables

**Surrogate specification**:
何を、どのデータと定式化で学習するかを宣言し、学習後も由来として残る仕様。
_Avoid_: Surrogate metadata

**Surrogate**:
学習仕様と学習済みの座標変換・閉包項を一体として扱うサロゲートモデル。
_Avoid_: Surrogate bundle

**Hybrid split**:
hybrid 定式化で、どの状態までを学習し、どこからを物理式で解き続けるかの分割。
分割位置も残す物理もニューロンモデルの性質なので `neurons` が持つ。
_Avoid_: Hybrid physics, physics config

## 構成

### ディレクトリ

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
              description.py     # 学習済み surrogate の中身を図に依らないデータで出す (describe)。閉包項と前処理の中身 (各実装の describe)、ネット、学習データ、学習 comp ごとに同定器へ渡した列・潜在・再構成 RMSE。置換シミュは回さない
scripts/  main.py                # Hydra エントリ (学習。surrogate とその中身 description/ を log)
          evaluate.py            # 評価の入口 `just eval <系列> <学習 run>...` (argparse)。mlflow_io.evaluation.evaluate を呼ぶだけ
          catalog.py             # **何を回すか**の 1 枚カタログ: EVALS (素材 1 条件) / SERIES (掃引。置換器を持たない素の EvalSeries だが、**どこを置換するかは挙げる** = replace_targets。回す側が SurrogateRuns.replacing で、対象を全部置換できる run だけの run 軸を張る)。**描き方は持たない** (つまみは live-textbook の冊が持つ)
          mlflow_io/             # MLflow I/O = **experiment と run を知る唯一の場所**。experiment ごとに 1 module で、どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」。公開名は run_* (確保) / load_* (読み) / log_* (書き) で揃え、複数段を完遂する evaluation.py だけ evaluate 1 本へ畳む。**再 export しない** (呼ぶ側は from mlflow_io.evaluation import ... と名乗る)
            __init__.py          # tracking URI をリポジトリ直下へ固定 (import 時に実行 = どの module を通っても最初に張られる) + _TARGET_EXP
            _query.py            # experiment id の解決 (`exp_id`。**書く側だけが作る**) と同一性 tag での最新 run 引き (`latest_by_tag`。**読む経路は experiment を作らない**)。4 点セットのうち experiment ごとに違わない 2 つをここに 1 つ置く (パッケージ外からは import しない)
            surrogate.py         # 学習 experiment の**成果物**: surrogate pickle/spec の読み書き (run_id ごとに @cache) + 学習 run の中身 (log_surrogate_description は学習時、run_surrogate_description は中身を持たない既存 run への書き足し → description/) + load_surrogate_runs (選んだ run 列 → SurrogateRuns。**選択を広げも縮めもしない** = 選択がそのまま run 軸)。どの run が居るか・選べるかは知らない (選ぶのは冊)
            _files.py            # 冊が読む形の書き出し: 波形 (write_waves / read_waves)、float32 の配列、nan を null にした JSON
            _series.py           # 波形 experiment eval_series (**1 run = 1 `sim.result.SeriesRun`** = 1 列。kind=original / kind=surrogate がフラットに並び、置換系は tags.original_hash で原系を名指す = 親子関係なし)。波形は TS でも読める形 (下の「成果物の置き場」)。置換系の run は原系と比べた指標・射影・学習側の指標も持つ。run_series は探索と実行が対 (決定的だから同じ入力は回さない) なので分けない。_load_column が run → SeriesRun の唯一の読み口
            evaluation.py        # 評価 = 1 系列を原系と選んだ学習 run で回す (evaluate)。系列を置換できない run は回さない。選んだ学習 run に中身が無ければ書き足す。どの列を並べて見るかは残さない (冊が選ぶ)
          conf/                  # 学習設定 (Hydra) のみ。下記「設定ファイル」参照
tests/    conftest.py (scripts/ を import path へ) / test_conventions.py (層の向き・公開名・import の機械検査) / test_surrogate.py / test_inits.py / test_eval_mlflow.py (評価 run の保存/読込。tracking 先は tmp へ差し替え)
```

### 成果物の置き場 (MLflow)

図は Python で描かず、live-textbook の `research/sindy-surrogate-evaluation` (評価) と
`research/sindy-surrogate-model` (学習 run) が MLflow の REST から読んで描く。Python が置くのは
図に要るデータだけ:

```
<学習 run>/  surrogate/                spec.json + pickle
             description/model.json    閉包項と前処理の中身 (describe)、ネット、列の名前
             description/train/        学習データ (waves と同じ形式、点は 1 つ)
             description/fit/<comp_id>/{x,latent,recon_rmse}.f32
                                       学習 comp ごとの同定器への入力・潜在・再構成 RMSE (列ごとに連続)
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

### 設定ファイル

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
  学習側は run の中身の `kind` (SINDy なら係数、PCA なら寄与率)、評価側は結果の形
  (点が 2 つ以上なら点軸の折れ線が出る) が決める。

## 判断基準

リファクタ・設計変更をエージェントに任せるための**判断基準**。

`AGENTS.md` の Coding Standards が「守る規則」、`tests/test_conventions.py` が「機械で検査される
不変条件」であるのに対し、ここは**真偽が決まらない判断** (この抽象を消すべきか / この class を
割るべきか) の拠り所を置く。迷ったら**現在の見通しを優先**する — 将来の柔軟性は根拠にならない。

各節の末尾に**発火条件**を書いてある。設計変更中にその条件を踏んだら、指示を待たずにこの
基準で判断してよい。

---

### 1. 抽象は「今すぐ必要」でなければ足さない、薄ければ消す

Protocol / 抽象基底クラス / dataclass wrapper / factory 関数 / `extra` のような自由 dict は、
「あると便利かも」段階では**作らない**。既にあるものは使用箇所を数えて、薄ければ削除し
コンストラクタ・本体クラス直接使用へ寄せる。

- 「Union type にして後で分岐」より「別 class で並列」。dispatch は 1 箇所 (load 時など) に集める
- `with_*` のような immutable-copy ヘルパは、呼び出しが 1〜2 箇所なら `dataclasses.replace()`
  かコンストラクタ直書きで済ませる
- 抽象を剥がすと**同型の別実装を並べられる**ようになるのが典型的な効き方
  (`Surrogate` Protocol を消したことで `HybridSINDyNeuroSurrogate` を並列追加できた)

**発火条件**: 抽象を新設しようとしたとき → 「今すぐ必要か? 将来の何のためか?」を明示できなければ
足さない。既存の抽象を見つけたとき → 使用箇所を数え、薄ければ削除を提案する。

### 2. class 分割は「別概念が混ざったとき」だけ

メソッドが増えても、それが**同一ライフサイクル** (学習 → 保持 → 適用 → 評価 → 保存) の中で
**同じデータを共有している**なら凝集は自然で、割らない。割ると引数渡しが増えるだけで、
薄いラッパが乱立する。

- 別概念 (ドメイン判定ロジック、外部 I/O など) が紛れ込んでいたら、適切な既存クラスへ移す
- 共有データを直接触る数行の付随操作 (metrics / save) は据え置く

1 と対になる: 1 は「不要な分離を消す」、2 は「不要な分離をしない」。

**発火条件**: クラス肥大を見たとき → 即分割せず「別概念の混入か / 同一ライフサイクルの一段階か」を
先に判定する。

### 3. 運ぶだけの型を作らない

同じデータを別の形へ詰め替える中間の型・保存専用の構造体は置かない。UI が持つ dict は
dict のまま**ドメインの解く関数 1 つ**まで届かせ、そこから先は plain 値だけを流す。

- 値の型に持たせるのは atomic な操作 1 つだけ (`Artifact.save(path) -> None`)。
  置き場や段の組み立ては**呼ぶ側**の関心
- 各要素が 1 回しか参照されない中間 dict (名前 → 値の倉庫) も消して直書きにする
- 返り値で path 列を持ち上げる代わりに、書いた先を見て数える

**発火条件**: 型を通そうとしたとき → 「この型は運ぶだけか?」を疑う。運ぶだけなら消して、
受け手の関数がそのまま解く。

### 4. ドメイン型を第一級に持ち回る

sympy 式・Symbol などドメインの実体は**型のまま**持ち回り、`str` 化は外部ライブラリ境界だけで
行う。**文字列キーの dict は「型から直接導ける値を名前で逆引きしている」サイン**で、たいてい
表ごと消せる。

- 派生値は表を作らず型から算出する (feature の OpCost = `op_cost(bound_expr)`)
- 同一性・重複検査も型で行う (str キー衝突 → `Counter(exprs)`)
- 外部 API が str を要求する箇所だけ str 化し、**境界で型と突き合わせて検証する**
  (fit 時に `[str(e) for e in feature_exprs] == sindy.get_feature_names()`)
- 名前規約 (座標名・変数名) が複数箇所で重複生成されていたら、改名前に規約を 1 関数へ集約する

**発火条件**: `dict[str, ...]` や `list[str]` を書こうとしたとき → その str が外部 API の
要求か、単なる逆引きキーかを判定する。

### 5. 表示用の文字列を識別子にしない

結果 dict のキーに「系列名#点値」のような**表示用文字列**を使わない。軸の位置をそのまま組にした
**構造タプル** (`(系列名, 点, run軸キー)`) にする。文字列に潰すと、引く側が添字や部分一致で軸を
復元する羽目になる。

**発火条件**: 表示に使う文字列をキーに使おうとしたとき → 「その文字列は本当に人が読む場所に
出るか」を確認する。出ないならキーを構造化し、表示が要る箇所ではその場で組んで捨てる。

### 6. ドメインは数値を出し、冊は描くだけ

ドメイン (`neurosurrogate/`) が出すのは数値と式 (LaTeX) だけで、表示のための加工 (係数の丸め、
0 係数を落とす、下付きの整形、配置、色) は live-textbook の冊が持つ。逆に冊はドメインの計算を
しない。冊に許すのは間引きや区間分けのような表示のための変換だけ。

**発火条件**: ドメインに表示名・色・並べ方を書きそうになったとき → 冊へ回す。冊で指標や
再構成誤差を計算しそうになったとき → Python で計算して MLflow に置く。

### 7. 後方互換は取らない

pickle / artifact のスキーマを変えて過去の MLflow run がロード不能になるのは**許容する**。
互換 shim (`__setstate__` での rename、旧属性フォールバック) も、旧形式 run を検出して選択対象から
外す防御コードも入れない。研究コードなので過去 run は学習し直せる。

**発火条件**: 保存形式を変えるとき → 互換性を検討せず進める。`AttributeError` が出たら回答は
「新スキーマで学習し直す」。

### 8. 実行経路を二重に持たない

同じ操作に 2 経路を作らない。重い計算 (学習と評価) は CLI (`scripts/main.py`、`just eval`) だけが
持ち、結果は MLflow に置く。描くのは live-textbook の冊だけで、Python で図を描く入口
(`scripts/draw.py` やレポート run) を復活させない。

**発火条件**: 新しい操作を足すとき → 計算なら CLI に、見せ方なら冊に置く。

### 9. リファクタの進め方

この順序が経験的に通る:

1. 使われていない dict / Protocol / wrapper を**消す**
2. 責務を本体 class へ**集約する**
3. 新機能を**並列 class として足す**

逆順 (先に新機能を足す) だと、残った抽象に新実装を合わせる作業が発生する。

**検証**: 段ごとに `just test` (20 秒以内) と `just lint` を通す。`tests/test_conventions.py` が
落ちたら設計規約を壊しているので、テストを緩めるのでなくコードを直す。テストは自由に足してよい。

**止まる条件** — 次のどれかに当たったら手を止めて報告する:
- 数値的な振る舞いが変わる (指標・波形が変わる) 変更が必要になったとき
- 判断が本ファイルの基準と衝突するとき、または基準がその状況を扱っていないとき
- 依存の向き (`tests/test_conventions.py::test_import_direction_follows_layers`) を変えないと
  進めないとき = レイヤ定義そのものの変更は設計判断なので独断で書き換えない

### 10. git は明示指示があるときだけ

実装・テスト・lint が完了しても、commit / push / branch へ勝手に進まない。「実装完了」を報告して
止まる。一度の指示は**その変更にだけ**効く。

## KAM 縮約を単一区画の外へ広げるときの前提

Kepler-Abbott-Marder 縮約を Traub の単一区画から Multi-Compartment と solver へ広げるときに守ること。

### Multi-Compartment

- 縮約後も膜電位を残す（φ 群の重み α₀=1）。区画間の結合式をそのまま使える。
- 群分けは区画ごとに確かめ直す。g_Na=g_DR=0 の区画では h,n が電流に効かないので群から除く。
- 伝播が保たれるかは元の系との比較で見る。

### solver との組み合わせ

- m を膜電位へ吸収する縮約と Rush–Larsen は、どちらも m の刻みの制約を外す。組み合わせても刻みの得は掛け算にならない。
- 誤差は「縮約だけ → 粗い solver → NN 補正」と段階を分けて測る。
