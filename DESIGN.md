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
neurosurrogate/                  # ドメイン層 (記録・設定の基盤を知らない)。依存の向きは core ← neurons ← sim.{_current_catalog,spec,result,waveform} ← surrogate ← sim.run ← run で、**core は他ディレクトリを一切 import しない**。層の所属と許可は `tests/test_conventions.py` の `_GROUP_OF`/`_LAYERS` が実行可能な形で持つ (ここの記述はその要約。新しいディレクトリを足したらあの表への追記が要る)。中身の無い `__init__.py` は置かない (再 export だけの層も作らない = 各実体を submodule から直接 import。そのため setuptools は `namespaces = true`)。**`_` 始まりのファイル名 = そのパッケージの外から import しない** (実測で内部専用のものだけが `_` を持つ)
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
        spec.py                  # **実験の記述だけ** (実行も結果も置換器も知らない): SimSpec (**唯一の仕様型**: 適用先 target × 電流。学習データの指定も評価条件もこれ 1 つ。**同一性は持たない**) + materialize (仕様 → DatasetConfig。名前 → 実体の解決はここだけ) + EvalSeries (spec+**replace_targets**+param+values の 1 掃引 = **保存の単位でもある**。どこを置換する実験かも記述の一部で、置換器は知らないまま対象を名前で挙げる。points は派生、to_dict/from_dict で往復 (run の spec.json の中身)。原系 `hash` (置換範囲を含まない) と置換系 `replaced_hash` (含む) は束ねてよい列かの検査に使う。run の置き場の鍵は段取りが組む)。**記述 2 段が result の 波形 / SeriesRun と 1 対 1**。surrogate より前の層 (SurrogateSpec.dataset がこれ)
        result.py                # **結果の器だけ** (計算も描画もしない): SeriesRun (**1 列** = 記述 EvalSeries + run_id (None=原系) + 波形 `list[xr.Dataset]`。**キャッシュの単位でも永続化の単位でもある** = runs/ の原系・置換系の 1 run と一致) / SeriesResults (原系 1 列 + 置換系の列 tuple。全列が同じ記述を回したことを構築時に検査。run_id は列が持つので束は id をキーに持たない)。**答えるのは「どの列か・どの点か」だけ** (column/pair/run_ids) = 適用先も掃引軸も刻み幅も素通しせず、要るものは view.series から直接引く。波形を包む型は無い — 点 i の計算入力は `series.points[i]`、対応は list の添字。**系列名も評価 run の id も持たない**
        run.py                   # **仕様 × surrogate を掛ける唯一の段** (保存を知らない。**何を**回すかは evaluation/ の conf.ts): _simulate (1 シミュ + 置換対象名 → 波形) / run_column (series + run_id + surrogate → **1 列** `SeriesRun`。系列 → 結果の唯一の入口。両方 None が原系)。**回す前に決まること (どの run が置換できるか) は持たない** → surrogate.replace.applicable。表示名は出てこない (学習 run の id は列の標識として通るだけ)。surrogate の後の層
        waveform.py              # 波形の指標: 常に 1 ペア (原系, 置換系) だけを見る (点軸も run 軸も持たない)。公開は compare 1 つ = 1 ペア 1 comp の全指標 (rows=(orig, surr) / scalars / spikes=全 AP の特徴量 / diverged) を素の値の dict で返す。並べ方と描き方は持たない (図は research/)
  surrogate/  model.py           # SurrogateSpec (宣言 + hyperparams + JSON 往復 + **設定ツリー (spec + parts 3 ブロック) を解く唯一の場所** from_config + 名前 → 実装の解決 ansatz (**型**) / preprocessor_fit (**fit_pca / fit_ae 関数**) + **学習ドメインの述語** in_train_domain (置換の可否と学習 comp の選定 train_comp_ids の両方が引く = 置換専用でないのでここ)) と Surrogate (**学習済みのもの** = 保存する 3 点 spec/preprocessor/closure だけを持ち (ansatz は spec しか状態を持たないので派生)、答えるのは持つ・保存する・読むと、比べる表に並べる学習側の指標 summary だけ = 適用先を知らない)。**組み立ては型のメソッドでなくモジュール関数** fit_surrogate — hyperparams まで spec が持つので引数は設定ツリー 1 つ。**preprocessor → closure の学習順もここが持つ** (定式化に依らないので Ansatz の既定実装にしない)
              replace.py         # **適用先 (NeuronGraph) を知る唯一の場所** = 置換の語彙一式。**公開は applicable / replace の 2 つだけ** (残りは `_` = 内部の綴り): どこへ当てられるか (applicable ← _rejected_targets) + 当てられない理由の**事実** (_Absent / _TypeMismatch / _ParamsMismatch の直和 = 理由の網羅が型で見える。**文言は持たない** — 文にするのは提示側 _NotReplaceable.__str__ だけ) + 当てること (replace(surrogate, dataset, targets) = 全数検証、1 つでも通らなければ部分適用せず _NotReplaceable)。**判定と適用を分けない** (replace が断る条件 = applicable が偽になる条件そのもの)。型のメソッドでなくモジュール関数 = 適用先はサロゲートの持ち物でない
              parts/  __init__.py # Surrogate が差し替える 3 構成要素の**契約を集約**: Closure / Preprocessor / Ansatz。3 つは互いを参照する (Ansatz が両者を受け、型引数で Closure に束縛) ので契約は 1 モジュール = 抽象レベルのパッケージ間依存辺を持たない。実装は下の 3 パッケージが `from .. import` で引く。対等ではなく closure/preprocessor が leaf、ansatz が両者を合成。**抽象メソッドしか置かない** — 既定実装も受け渡しの型も持たない
                      train_inputs.py # TrainInputs (ansatz が組み closure の同定入口が受ける受け渡しの型)。契約でないので __init__.py に置かず、両側がここを引く
                ansatz/          # sindy.py / hybrid.py / ude.py。hybrid.py が物理骨格と SINDy hybrid を集約。**インスタンスを作らない** = 定式化は規則であって状態でないので全メソッドが classmethod。spec に依るメソッドは spec を引数に取る (`surrogate.ansatz.f(surrogate.spec, ...)`)。定式化だけの性質である params_match は例外
                closure/         # ude.py / sindy/{__init__,roles,_entry,_catalog}.py
                preprocessor/    # pca.py / autoencoder.py / fit_artifacts.py (fit 後に埋まる派生量 = 再構成統計と初期潜在。埋め方は全実装共通なので契約でなくここ)
              description.py     # 学習済み surrogate の中身を図に依らないデータで出す (describe)。閉包項と前処理の中身 (各実装の describe)、ネット、学習データ、学習 comp ごとに同定器へ渡した列・潜在・再構成 RMSE。置換シミュは回さない
  run.py                         # **計算の入口** (`python -m neurosurrogate.run <spec.json> <run のディレクトリ>`。層 `entry` = どこからも import されない最上位): spec.json 1 つ → run のディレクトリ。kind = train / sim (meta.json では train / original / replaced)。隣の .partial に書いてから名前を変える (半端な run を残さない)。置換できない組み合わせは終了コード 3 で何も書かない。**run の保存形式 (波形の write/read) を知るのはここだけ**
tests/    test_conventions.py (層の向き・公開名・import の機械検査) / test_surrogate.py (preset と系列は段取りに JSON で出させて読む) / test_inits.py / test_run.py (neurosurrogate/run.py の 3 種類の run の往復と、回し直さないこと)
evaluation/    # TS はすべてここ。直下は評価の冊だけで、.ts は役割のディレクトリ (orchestrate/ read/ traub/)。評価の冊 (TS + Svelte。画面と部品は context-router の resources/.web/。context-router の resources/50-59-research/ に symlink で入る。研究の文書 51・52・54 は context-router 側)。Python とは import し合わず、開発サーバーが配る runs/ と設定 JSON だけを読む。evaluation/traub/traub.ts は Traub のシミュレータの TS 版で、ずれは reference.json のテストで見張る
evaluation/orchestrate/conf.ts        # **何を回すか**の宣言だけ: PRESETS (学習の設定。preset が決める値を平らに書き、`expand` が設定ツリーへ開く。n_components を配列で書くと掃引)、SERIES (評価の掃引 = Python の EvalSeries の形。**どこを置換するかも挙げる** = replace_targets)、EXPERIMENTS (見比べる preset と系列の組。冊はここから選び、どの実験にも入らない preset と系列は冊に出ない)。**描き方は持たない**。node/bun に依存しない (ブラウザの冊も import する)
evaluation/orchestrate/spec.ts        # spec.json の形 (Spec、FORMAT)、index.json の行、段取り・口・図が受け渡す状態の型
evaluation/orchestrate/orchestrate.ts # **段取り** (bun): 宣言 → spec.json → ハッシュ (canonical + sha1。**鍵を組むのはここだけ**) → 置き場 (runs/models/<ハッシュ>/ か runs/inputs/<入力のハッシュ>/<original かモデルの鍵>/) が無ければ `python -m neurosurrogate.run` を呼ぶ → runs/index.json を書き直す。置き換えたモデルは学習 run のハッシュを spec の model に入れて名指す。入口は二つで同じ関数 (train / evaluate) を呼ぶ: 本人が使う操作盤の冊 (53.03-runs) のボタンが切り離した子プロセスで起こす launch と、エージェント用の CLI (`bun evaluation/orchestrate/orchestrate.ts train|eval|list`。人は使わない)。宣言の組の状態 (status: 済み・計算中・未・置換できない) と、launch で起こした仕事の状態 (runs/.jobs/ の中身・pid・ログ・終了コード) もここ
evaluation/orchestrate/api.ts         # **操作盤の口**: 開発サーバーが /api/resources/50-59-research/53-evaluation/orchestrate に載せる (context-router の resources/.web/routes/api/resources は区域の api.ts を載せるだけの汎用の仕組み)。GET = status (実験ごとの preset・系列・組のハッシュと状態、仕事のログの末尾。図はこのハッシュで runs/ を読む。ブラウザではハッシュを組めないため)、POST = 宣言にある preset 名・系列名だけを受けて launch。任意のコマンドは通さない
evaluation/read/runs.ts        # 図が runs/ を読む口 (ブラウザ)
evaluation/read/model-json.ts  # 学習 run の description/model.json を図の形に直す
evaluation/traub/traub.ts      # Traub の単一区画の TS シミュレータ。53.01・53.02 と context-router の 52.01 が使う。Python とのずれは traub.reference.json のテストで見張る
runs/     models/ と inputs/ と index.json  # 実験の結果 (Git の外)。下の「run の置き場」
```

### run の置き場

保存の単位はモデルと入力の二つ。学習したモデルは `runs/models/<学習のハッシュ>/`、入力 (電流と掃引 =
系列から置換範囲を除いたもの) ごとの結果は `runs/inputs/<入力のハッシュ>/` の下に、元のモデルなら
`original/`、置き換えたモデルなら `<モデルの鍵>/` (学習 run のハッシュ + 置換範囲) で置く。1 run = spec.json 1 つで、
置き場は spec から決まる (`orchestrate.ts` の `dirOf`)。同じ spec は同じ置き場なので、ディレクトリがあれば回さない。
run は `runs/` の下のパス (id) で指す。図は Python で描かず、`evaluation/53.05-evaluation` (評価) と
`53.04-trained-model` (学習 run) が `runs.ts` を通して、口が宣言から組んだ id で読む。Python が置くのは図に要る
データだけ:

```
runs/index.json                全 run の id + meta.json + spec (段取りが書く。図はここから一覧を作る)
<run>/  spec.json              回した条件 (format = 形式の版, kind, ...)
        meta.json              kind (train / original / replaced), name (図に出す名前。同一性に入らない), created, unit (掃引の単位)
models/<ハッシュ>/  surrogate/                spec.json + pickle
                    summary.json              Surrogate.summary (学習側の指標)
                    description/model.json    閉包項と前処理の中身 (describe)、ネット、列の名前
                    description/train/        学習データ (waves と同じ形式、点は 1 つ)
                    description/fit/<comp_id>/{x,latent,recon_rmse}.f32
                                              学習 comp ごとの同定器への入力・潜在・再構成 RMSE (列ごとに連続)
inputs/<ハッシュ>/  input.json                電流と掃引 (段取りが書く)
                    original/waves/           下の形式。meta.json に comp 名 → id も入る
                    <モデルの鍵>/ waves/
                                  metrics.json    comp 名 → 点ごとの compare (全 comp・全点。相手は original/)
                                  projected/<comp>/ 置換した comp の original/ のゲートを潜在空間へ射影したもの (waves と同じ形式)

waves/meta.json            t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]], nodes, names
waves/<点>/comp<id>.f32    その comp の変数 (features の並び順)。変数ごとに連続した little-endian float32
waves/<点>/I_ext.f32       waves/<点>/I_internal.f32 (node ごとに連続)
```

- spec の 2 種類: `train` は設定ツリー (`surrogate`)、`sim` は入力 (`input`) とモデル (`model` = `{train, replace_targets}`、
  元のモデルは null)。置換範囲はモデルの側に入るので、置換範囲だけが違う系列は同じ入力のディレクトリに別のモデルとして並ぶ
- **ハッシュを組むのは TS だけ。** Python は渡された spec とディレクトリを埋めるだけで、学習 run は spec の `model.train`、
  比べる相手は隣の `original/` で引く (JS と Python で数値の JSON 表記が違うので、鍵を 2 か所で組まない)
- **比べた束 (どの列を並べたか) は残さない。** 図が選び直すたびに描く
- 形式を変えたら `neurosurrogate/run.py` の `_FORMAT` と `spec.ts` の `FORMAT` を変える。spec に入るので、
  古い形式の run は置き場が合わず回し直される
- **記録した run を描画が書き換えない**: 図は読むだけ
- JSON に nan は書かない (null)。JS の JSON が読めないため
- **書きかけと印**: `<置き場>.partial/` は書きかけで、中の `pid` が書いているプロセス (生きていれば計算中、死んでいれば残骸として次に片付ける。同じ run を二つのプロセスが書かないよう、生きた書き手がいれば run.py は終了コード 4 で退き、段取りは終わるのを待つ)。`<置き場>.unreplaceable` は置換できないと分かったモデルの印 (置換できるかは Python だけが知るので答えを覚える)。`.jobs/<組のハッシュ>.{json,pid,log,exit}` は操作盤のボタンが起こした段取りの中身・pid・ログ・終了コード (同じ組の二重起動を pid で弾き、終了コードが 0 でないか書かずに消えたものを失敗として操作盤に出す)

### 設定ファイル

- 学習の設定も評価の条件も `evaluation/orchestrate/conf.ts` に型付きで書く。YAML の設定ファイルは持たない。
  綴りの浅い誤りは型が、名前の誤り (comp 名、項ライブラリの名前) は pytest が Python のドメインに当てて見つける
  (`test_catalog_is_self_consistent`、`fit_preset`)。実験条件は滅多に変わらず、変えたら別の実験 =
  コードの差分に出る方が正しい。
- 学習の設定ツリーは `spec` (何を学習するか) + **`parts/` の 3 構成要素と同名のブロック**
  `preprocessor` / `ansatz` / `closure` で、**それぞれその層の入口 1 つの署名へ `**` で
  展開される** (`fit_pca` / `fit_ae` / `HybridAnsatz.split` = physics 分割の選択。
  hybrid 系は `physics_type` 必須で既定を持たない (comp_type 名で代替しない) /
  閉包項の同定入口 = SINDy の `from_sindy`・UDE の joint 学習)。1 ブロック = 1 宛先
  なので、受理するキーと既定値はその署名だけが決め、綴り違いや層違いは TypeError。**既定値は宣言でなく実装側の署名**が単一源 —
  宣言に既定を書くと、その層の実装が違う preset へも黙って継承されるため。
  (`spec.datasets` は `SimSpec` のフィールドそのもの = target/current_type/dt/current_params)。
- preset は継承しない。各 preset が決める値を全部書き、どの preset でも同じ値 (学習用の電流・刻み・全区画で学習) だけを
  `expand` が埋める。共通の値の組は TS の展開で使い回してよい。掃引は n_components の配列だけ。
  `_test_*` はテスト専用 preset (tests は preset 名を指すだけ) で、どの実験にも入れない。
- 使わなくなった preset と系列は消す。run は消さず、操作盤の一覧に「実験に無い」と出る。
- **どの系列を回すか**は操作盤で選ぶ組 (エージェントは CLI の eval の引数) で、系列を置換できない学習 run は回さない。
- **描き方 (つまみ) は宣言に持たない** → 全キー (比較対象 comp・
  全 comp 図の表示制限・点軸の指標・詳細図の点・スパイク番号・折れ線の y レンジ)
  を evaluation/ の図が持つ。どれも図を見て決め直すもので、宣言に置くと「何を
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
0 係数を落とす、下付きの整形、配置、色) は evaluation/ の図が持つ。逆に図はドメインの計算を
しない。図に許すのは間引きや区間分けのような表示のための変換だけ。

**発火条件**: ドメインに表示名・色・並べ方を書きそうになったとき → 冊へ回す。冊で指標や
再構成誤差を計算しそうになったとき → Python で計算して run に置く。

### 7. 後方互換は取らない

pickle / run のスキーマを変えて過去の run がロード不能になるのは**許容する**。
互換 shim (`__setstate__` での rename、旧属性フォールバック) も、旧形式 run を検出して選択対象から
外す防御コードも入れない。研究コードなので過去 run は学習し直せる。

**発火条件**: 保存形式を変えるとき → 互換性を検討せず進める。`AttributeError` が出たら回答は
「新スキーマで学習し直す」。

### 8. 実行経路を二重に持たない

同じ操作に 2 経路を作らない。重い計算 (学習と評価) は `neurosurrogate/run.py` だけが持ち、段取りは
`orchestrate.ts` の関数だけが持つ (操作盤のボタンとエージェント用の CLI はその関数を呼ぶ薄い入口)。結果は runs/ に置く。描くのは evaluation/ の図だけで、Python で図を描く入口
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

## Python と TS のニューロン設定

ニューロン固有の設定（パラメータと、指数・シグモイド・線形×指数の三つの形に収まるレート）だけを一つの JSON に置き、Python と TS が読む。
シミュレータ本体と三つの形の関数は両方で手書きする。形に収まらない c・r・q のレートは両方で手書きし、共通の基準値のテストで見張る。
重い計算と保存は Python、段取りは TS、描画は Svelte が runs/ から読む。

## 原著とずれている箇所

- Traub19 の basal と apical の寸法が原著と逆（2026-09-28 原著 p.637 で確認）。原著は apical（10 区画）が 2.89 µm・120 µm・2188 µm²、basilar（8 区画）が 2.42 µm・110 µm・1673 µm²。
  コードは index 0..7 に apical の寸法、9..18 に basilar の寸法を入れている。φ（7769 と 5941）の比 1.308 は面積比 2188/1673 と一致し、φ は原著どおり 1..8 が basilar。
  参考にした snsbook の code/part1/multi/traub.c も同じく逆で、そこから引き継いだ。逆のままだと入力区画（原著 3・15）と basal・soma・apical の添字がずれる。
