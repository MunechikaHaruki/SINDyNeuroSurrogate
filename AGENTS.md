---
description: "研究コードの repo の規則と設計。用語、run と設定の形式、判断基準、コマンド"
---

# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。全体の地図と実験の流れの図は `README.md`、
`remote/neurosurrogate/` のファイルごとの持つものと依存の図は `remote/neurosurrogate/AGENTS.md`（生成物）にある。設計に手を入れる前にこのファイルを読む。

## 持つもの

最上位は、手元の側 `local/`、計算する機械の側 `remote/`、両方が使う形 `shared/` の 3 つだけ。`local/` と `remote/` の直下は、人やエージェントが直に走らせる入口（手元は spec を書き出す `resolve.ts`、送って回す `remote.sh`、実験の状態を出す `cli.ts`。計算する機械は API `serve.ts` と、spec を回す `run-spec.ts`）と、入口が使う部品の役割ごとのディレクトリと、依存やイメージを決めるファイルだけ。手元は 前処理 `local/experiments/`（人が書く宣言 `local/experiments/config/`、それを解くスキーマ `local/experiments/schema/`、解決済みの spec `local/experiments/resolved/` の 3 つ）→ 通信、計算する機械は 通信 → 段取りと計算 → 統計量 `remote/read/` の順に並ぶ。結果を読むのは使う側（`local/cli.ts` と repo の外の冊）で、それぞれ API を直に叩き、宣言と API の実験の状態から自分で組み立てる。

| 中身 | 場所 |
| --- | --- |
| 人が書いて確かめる宣言（プリセット `presets.json` の比べる物差し `measures`・既定 `defaults`・仕方 `methods`・系列 `series`、実験群ごとの実験層 `<群>.yaml`、決め打ちの実験 `<実験>.json`） | `local/experiments/config/` |
| API に送る前処理の TS。宣言を解く部品（プリセットを混ぜ、系列とモデルの形を持つ `presets.ts`、実験層と決め打ちの JSON から解決済みの実験を組む `schema.ts`、並べる `experiments.ts`） | `local/experiments/schema/` |
| 入口。送る spec を組んで `local/experiments/resolved/` に書く（write）、古くないか確かめる（check） | `local/resolve.ts` |
| 手元の側の通信。計算する機械へ ssh でコードを送り、イメージを用意して API を立て、段取りとテストを回す script | `local/remote.sh` |
| 入口で、使う側の一つ。エージェント用の CLI（API の実験の目次を読み、実験を状態ごとに並べて書く） | `local/cli.ts` |
| 手元の TS の npm の依存 | `local/package.json`、`local/bun.lock` |
| 宣言から組んだ、送る spec.json そのものと、テスト専用の学習の設定 `tests.json` と、API が画面の答えを組むのに読む宣言の写し `plans.json`（生成物。手で書き換えない。計算する機械へは `local/remote.sh` が向こうの resolved/ に写し、計算する機械の側は `SINDY_RESOLVED` でその場所を受け取る） | `local/experiments/resolved/` |
| 手元と計算する機械が受け渡す形（行きの spec と run の一覧と状態 `spec.ts`、spec の鍵 `hash.ts`、API の口と返事 `api.ts`、run のファイルの形と置き場と読み方 `runs.ts`（読むのは計算する機械だけ）） | `shared/` |
| 数値計算の Python（元のモデル `remote/neurosurrogate/physics/`、学習と置換 `remote/neurosurrogate/surrogate/`、計算の入口 `remote/neurosurrogate/run.py`、GPU の実測 `remote/neurosurrogate/bench.py`、API が呼ぶ eFEL の特徴量 `remote/neurosurrogate/spikes.py`）とそのテスト | `remote/neurosurrogate/`、`remote/tests/` |
| 入口。送られてきた resolved/ の spec を回す段取り（先に要る run を用意し、Python の計算の入口を起こす。画面から起こした仕事と、run・list の CLI もここ） | `remote/run-spec.ts` |
| 段取りの部品。実験の状態 `status.ts`、置き場（resolved/ からパスで spec を引く、runs/ の置き場と run 1 本の状態）`places.ts` | `remote/run/` |
| API の答えを作る部品。口ごとの答えを組む `experiments.ts`（目次・実験 1 つ・run の一覧）と、run の置き場ごとの `inputs.ts`・`models.ts`・`data.ts`・`parts.ts`・`bench.ts`、run のファイルを読んで覚え、回線に出す形（間引き・軌跡・分布）にする `waves.ts`、波形から先の統計量を全部、頼まれるたびに純粋関数で計算する `comparison.ts`（波形 1 本の eFEL の特徴量・0 mV の横切り・山の平均の形・発散、原系と列の点ごとの rmse・mae・ISI の統計・周期の差・形の相関・横切りの差・Γ、列ごとの Γ の平均と発火の始まり）、GPU の実測のツールごとの出力を細胞の数ごとの行に集める `bench.ts`（ncu の区画ごとのカーネルの合わせ方もここ） | `remote/read/` |
| 計算する機械の側の通信。答えを小さい JSON で返す API（口は実験の目次と実験 1 つ、run の置き場ごとの inputs・models・data・parts・bench、run の一覧、仕事を起こす口。口を `remote/run-spec.ts`・`remote/run/`・`remote/read/` の関数に渡すだけ） | `remote/serve.ts` |
| 計算する機械のイメージと Python の依存 | `remote/Dockerfile`、`remote/pyproject.toml`、`remote/uv.lock` |
| run | 計算する機械の `~/sindy/runs/`（手元の runs は `local/remote.sh fetch` の写しだけ。どちらも Git の外） |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）
- 結果を読む図と画面
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

## 流れ

手元（`local/`）と計算する機械（`remote/`）は、JSON だけを受け渡す。図は `README.md` にある。

```
手元 local/:       experiments/config/ → resolve.ts → experiments/resolved/ ─┐      ┌─→ cli.ts・repo の外の冊
                                                               │ 行き   │ 帰り
計算する機械 remote/:          run/ → neurosurrogate/ → runs → read/ → serve.ts
```

- `local/` と `remote/` は互いのコードを import せず、両方が使うのは `shared/` だけ（`local/experiments/schema/boundary.test.ts`）。
- 人やエージェントが直に走らせる入口は `local/` と `remote/` の直下だけに置く。マシン間の通信もここが受け持つ（手元は `local/remote.sh`、計算する機械は `remote/serve.ts`）。
- 計算する機械へ送るのは `remote/`・`shared/` と、手元が組んだ `local/experiments/resolved/` の写しだけで、宣言と手元のコードは送らない（`local/remote.sh`）。計算する機械は宣言を知らず、送られてきた resolved/（場所は `SINDY_RESOLVED`）の spec を回し、run の量を返すだけ。
- runs は生データ（波形と学習の書き出し、GPU の実測で使ったツールの出力をツールごとのファイルのまま）だけを持つ。波形から先の統計量（eFEL の特徴量から Γ と列ごとのまとめまで）も、どの列を比べるかの組み立ても、設定の表も、計算する機械の API が頼まれるたびに計算して組み、すぐ描ける形で返す（`remote/read/`）。使う側は口の型（`shared/api.ts`）だけを知り、run のパスや列の位置（`shared/runs.ts`）も宣言（`local/experiments/`）も読まず、描き方（色、凡例、数式の書き方）と選んでいる状態だけを持つ。比べる物差し（Γ の許容幅 Δ）は宣言（`local/experiments/config/presets.json` の `measures`）が持ち、宣言の写し `plans.json` で API に渡る。見せ方の線（時刻まで合うとみなす Γ の線）は API（`remote/read/experiments.ts`）が持つ。`shared/` に置くのは両方が使う形だけで、宣言の形（`Series`・`ModelDecl`）は `local/experiments/schema/presets.ts`、`Bench`・`Plan` は `local/experiments/schema/schema.ts` が持ち、その写しの形（`Plans`）だけを `shared/spec.ts` が持つ。

| 渡るもの | 書く側 | 読む側 |
| --- | --- | --- |
| 実験の条件（spec.json。`local/experiments/resolved/<実験>/` に置いたものをそのまま渡す）、テスト専用の学習の設定（`local/experiments/resolved/tests.json`）、宣言の写し（`local/experiments/resolved/plans.json`） | 手元（`local/resolve.ts`） | 段取りが Python へ、pytest、API |
| 実験の結果（runs/） | Python | 計算する機械の API |
| 実験の目次と実験 1 つ、run の置き場ごとの答え（間引いた線、指標、表の値。口の形は `shared/api.ts`） | 計算する機械の API | 使う側（`local/cli.ts` と repo の外の冊） |
| ゲートのレートの式（physics/ の JSON） | 人 | Python |

## 用語

**群 (Group)**:
区画の状態のうち、まとめて 1 つの圧縮器と右辺で置き換える組。状態の名前（`cells` の `states`）で書く。

**form**:
`hybrid` は群に入らなかった状態と膜電位を元の式のまま解く。`sindy` は膜電位も含めて丸ごと同定し、群は全状態の 1 つだけ。

**物理の状態 (physics)**:
hybrid で群に入らなかった状態。元のモデルの微分の式 (`CellType.derivs`) で解く。

**kernel**:
サロゲートの 1 刻みは一つの汎用関数で、群ごとに潜在を decode して状態を名前で埋め、残りの状態は元の式で微分し、名前で組み直す。学習は群ごとに、圧縮器を先に当ててから右辺を同定する。群の右辺は自分の潜在と膜電位だけを使い、群の間で潜在を混ぜない。初期値は元のモデルの初期値と圧縮器の `gate_inits`、演算コストは群の圧縮器と右辺、膜電位の式、物理の状態のコストの和。群分けを足すのに書くのは設定だけ。

## 形式

### run の置き場

保存の単位は学習データ (`runs/data/<ハッシュ>/`)、学習器 (`runs/parts/<ハッシュ>/`。群 1 つの圧縮器と右辺)、学習器の並びを組んだモデル (`runs/models/<ハッシュ>/`)、入力 (`runs/inputs/<入力のハッシュ>/`)、GPU の実測 (`runs/bench/<ハッシュ>/`) の五つで、入力の下に元のモデル (`<入力のハッシュ>/original/`) と置き換えたモデル (`<入力のハッシュ>/<モデルのハッシュ>@<置換する区画を + でつないだもの>/`。例 `3ea34b09c3f4@soma`。どのモデルでどこを置き換えたかが名前で読める) を置く。学習器は学習データを、モデルは学習器を (KAM のモデルは学習器を持たず、群を spec に直に持って学習データを指す)、評価と GPU の実測はモデルをハッシュで指す。学習器の run は役割の名前の置き場 (圧縮器 `runs/parts/<ハッシュ>/compressor/`、右辺 `runs/parts/<ハッシュ>/rhs/`、学習データの上 `runs/parts/<ハッシュ>/fit/`) に分け、種類 (PCA・AE・SINDy・UDE) の違いは中身の `kind` と置き場の有無だけが持つ。KAM は細胞全体の膜電流から組む縮約 (Kepler 1992) なので、学習器でなくモデルの単位に置く。図は置き場 1 つを 1 枚のカードにする。群の学習器はほかの群に依らないので、同じ群を持つモデルどうし (`pca-sindy.h+n` と `pca-sindy.h+n_a+c` の h+n) は同じ学習器を共有する。1 run = spec.json 1 つで、置き場は spec から決まる (`remote/run/places.ts` の `dirOf`)。同じ spec は同じ置き場なので、ディレクトリがあれば回さない。図は Python で描かず、Python が置くのは図に要るデータだけ:

```
<run>/  spec.json              回した条件 (format = 形式の版, kind, ...)
        meta.json              kind (data / part / model / original / replaced / bench), name (図に出す名前。同一性に入らない), created, elapsed (計算にかかった秒), unit (掃引の単位),
                               env (回した backend・device・jax と python の版・host・コードとイメージの版。同一性に入らない)
data/<ハッシュ>/    waves/                    学習データ (下の waves と同じ形式、点は 1 つ)
parts/<ハッシュ>/   learner.joblib            学習器 (圧縮器と右辺)
                    compressor/decode.json    潜在 → 群のゲートの曲線を潜在の軸ごとに (ほかの軸は学習データの平均)。区画の型の状態の並びも
                    compressor/plane.json     AE だけ。群のゲート 2 本ずつの平面の上で encoder が返す潜在の格子 (等高線に使う。範囲は群の全ゲートを合わせた学習データの幅を少し広げたもので、どの軸も同じ。ほかのゲートは学習データの平均)
                    rhs/field.json            hybrid だけ。学習器の右辺と、元のモデルの潜在の速さ (群のゲートを decoder の曲線の上に置いて元の式で動かしたもの) の (V, z) の格子
                    compressor/own.json       圧縮器に固有の中身 (kind で形が変わる。PCA の寄与率と向きと平均、AE の隠れ層)
                    rhs/own.json              右辺に固有の中身 (kind で形が変わる。SINDy の項と係数、UDE の層と損失)
                    fit/<comp_id>/{latent,recon_rmse}.f64
                                              学習 comp ごとの潜在 (列ごとに連続) と再構成 RMSE
                    fit/<comp_id>/summary.json 学習 comp の全区間の再構成の MSE と、それをゲートの分散で割ったもの
models/<ハッシュ>/  summary.json              surrogate.summary (組んだ 1 刻みの演算コスト。学習器の中身は学習器の run だけが持つ)
                    kam.json                  KAM のモデルだけ。描いた区画、吸収したゲート、群ごとのゲートと潜在の名前と Kepler 1992 の式(6)の重みと式(10)の条件
                    equivalent/               KAM のモデルだけ。学習データの区画の元のモデルの軌道を等価電位にしたもの (V、吸収したゲートと群のゲートの等価電位、群ごとの ψ。waves と同じ形式)
                    reduced/                  KAM のモデルだけ。同じ電流を縮約で回した waves
inputs/<ハッシュ>/  input.json                電流と掃引 (段取りが書く)
                    original/ waves/          下の形式。meta.json に comp 名 → id も入る
                    <モデルのハッシュ>@<区画>/ waves/
                                  projected/<comp>/ 置換した comp の original/ のゲートを潜在空間へ射影したもの (waves と同じ形式)
                                  slope/<comp>/ と rhs/<comp>/ その上の潜在の傾きの正解 (射影の前進差分) と、学習器の右辺が返す dz/dt (同じ形式)
bench/<ハッシュ>/   <細胞の数>/bench.json      bench.py が自分で測ったもの (時間、状態の置き方と量の予想、発火の回数、pid)
                    <細胞の数>/jax.memory_stats.{placed,after}.json
                                              JAX の memory_stats そのまま (入力を置いた後と回した後)
                    <細胞の数>/nvidia-smi.{apps,gpu}.csv
                                              nvidia-smi の CSV そのまま (このプロセスの使用量と GPU の全容量)
                    <細胞の数>/ncu.csv         ncu の標準出力そのまま (1 刻みの区画ごとのカーネルの指標。spec の ncu が真のとき)

waves/meta.json            t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]], nodes, names
waves/<点>/comp<id>.f64    その comp の変数 (features の並び順)。変数ごとに連続した little-endian float64
waves/<点>/I_ext.f64       waves/<点>/I_internal.f64 (node ごとに連続)
```

- spec はどれも `device` (`cpu` か `gpu`) を持つ。同じ条件でも計算の順序が変わって結果がずれるので、同一性に入れる。学習器は `gpu` (GPU を使うのは AE の学習だけ。テスト専用のモデルの学習器は `cpu`)、学習データ・モデル・評価は `cpu`、GPU の実測は `gpu`。決めるのは `local/resolve.ts` の `dataSpec`・`partSpec`・`modelSpec`・`simSpec`・`benchSpec`
- spec の種類: `data` は学習データの SimSpec、`part` は区画の型・形・群 1 つと、学習データのハッシュと学習区画、`model` は区画の型・形・吸収と学習器のハッシュの並び (群の順。KAM のモデルは空で、代わりに `kam` に群と、図を描く学習データのハッシュと区画を持つ。KAM は区画の型の既定の params で組む。KAM と学習器を混ぜたモデルは組まない)、
  `sim` は入力 (`input`) とモデル (`model` = `{surrogate, replace_targets}`、`surrogate` はモデルのハッシュ、元のモデルは null)、`bench` は sim と同じに細胞の数と ncu。
  宣言のモデル 1 つ (`surrogate.Config`) を、`local/resolve.ts` が学習データ・群ごとの学習器・それを組んだモデルの spec に分ける。置換範囲はモデルの側に入るので、置換範囲だけが違う系列は同じ入力のディレクトリに別のモデルとして並ぶ
- **ハッシュを組むのは TS の `shared/hash.ts` だけ。** Python は渡された spec とディレクトリを埋めるだけで、指す run は spec のハッシュ (`data`・`parts`・`model.surrogate`) で run の置き場から引き (`remote/neurosurrogate/store.py`)、
  比べる相手は隣の `<入力のハッシュ>/original/` で引く (JS と Python で数値の JSON 表記が違うので、鍵を 2 か所で組まない)
- **比べた束 (どの列を並べたか) は残さない。** 図が選び直すたびに描く
- 形式の版は spec の種類 (data・part・model・sim・bench) ごとに持つ。形式を変えたら、変えた種類の版を `remote/neurosurrogate/run.py` の `_FORMATS` と `shared/spec.ts` の `FORMATS` で上げる (揃っているかは `remote/run/places.test.ts`)。spec に入るので、
  その種類の古い run と、ハッシュでそれを指す後ろの run (学習器を変えればモデル・置換系・GPU の実測) だけが置き場が合わず回し直される。元のモデルの評価や学習データは、変えなければ残る
- **置き場の口は自分の run と、自分の spec がハッシュで指す run だけを読む**: data・parts・models・inputs・bench の口が開ける run は、その run と spec の欄が指す run (学習器なら学習データ、モデルなら学習器、評価と GPU の実測ならモデルと隣の元のモデル) だけ。原系と置換系の比べは、両方を持つ入力の口が返す。ほかの run は名前で指すだけ (`remote/serve.test.ts` が口ごとの読み先を確かめる)
- **記録した run を描画が書き換えない**: 図は読むだけ
- JSON に nan は書かない (null)。JS の JSON が読めないため
- **書きかけと印**: `<置き場>.partial/` は書きかけで、中の `pid` が書いているプロセス (生きていれば計算中、死んでいれば残骸として次に片付ける。同じ run を二つのプロセスが書かないよう、生きた書き手がいれば run.py は終了コード 4 で退き、段取りは終わるのを待つ)。`<置き場>.unreplaceable` は置換できないと分かったモデルの印 (置換できるかは Python だけが知るので答えを覚える)。`.jobs/<組のハッシュ>.{json,pid,log,exit}` は外の画面が `--job` で起こした段取りの中身・pid・ログ・終了コード (同じ組の二重起動を pid で弾き、終了コードが 0 でないか書かずに消えたものを失敗として返す)

### レートの式

ゲートのレート α, β は `remote/neurosurrogate/physics/<モデル>.json` に 1 本 1 行で書き、Python (`cells.py`) がこの行から式を組む。ほとんどは `rate(u) = A f(z)`、`z = (θ - u) / s` の形で、`f` は `exp` (e^z)、`sigmoid` (1 / (1 + e^z))、`linexp` (z / (e^z - 1)) のどれか。`u` は原著の基準電位からの電位 (Traub は V + 60、HH は静止電位から)。JSON には `f` の名前と `A`・`theta`・`s` だけを書く (例: Traub の m の開くレートは `"alpha": { "form": "linexp", "A": 1.28, "theta": 13.1, "s": 4.0 }`)。レートの演算コストもこの形から導く。c・r (電位で場合分けし、β を α との差で置く) と Ca で動く q はこの形に収まらない。

### 設定ファイル

- 学習の設定も評価の条件も `local/experiments/config/presets.json` に書く。値は既定 (`defaults`) も含めてすべて JSON に出し、コードは混ぜて形を確かめるだけにする (JSON だけで何を回すかが分かるように)。モデルは 縮約の仕方 × 群分け で決まる。名前は記号に詰めず、軸ごとに綴る (単語は `-`、群の中のゲートは小文字を `+`、群どうしは `_`、モデルの仕方と群分けは `.` でつなぐ。例 仕方 `kam-absorb-m`・`pca-sindy`、群分け `h+n_a+c`、系列 `traub-19comp-steady-replace-soma`、実験 `all-groupings-1comp`)。仕方と系列には既定と違う値だけを書き、`local/experiments/schema/presets.ts` が defaults → プリセット → 差分の順に混ぜて形を確かめる (`methodOf`・`modelFrom`・`seriesFrom`)。プリセットは JSON のままにし、YAML は実験の宣言だけに使う。
- 学習の設定 (`surrogate.Config`) は、区画の型 (`cell`)、`form`、学習データ (`data` = `SimSpec`)、群のリスト (`groups`)、学習を絞る区画 (`train_comp`)。
  群は `gates`・`compressor`・`d`・`rhs` と、それぞれの fit 関数へ `**` で展開する `compressor_params`・`rhs_params`。
  `gates` は状態の名前 (Traub は M N C A H B S R Q XI、HH は M H N)。`compressor: kam` は学習しない KAM 縮約で、右辺も `kam`、`d` は 1。UDE は AE を一緒に学習するので、圧縮器は AE に限る。SINDy のレート項は、`library_specs` に `gates` を書かなければ群の状態のものに絞る (`_own_rates`)。
  既定値は fit 関数の署名だけが持ち、綴り違いは TypeError になる。
- 値の既定は `local/experiments/config/presets.json` の `defaults` が持つ。値から導く既定 (教師は細胞と同じ、圧縮は書かなければ仕方の名前、右辺は KAM なら KAM でほかは SINDy、項ライブラリは細胞のゲートの項と `basis`) と、どの学習でも同じ形 (学習用の電流・全区画で学習) だけは `local/experiments/schema/presets.ts` が導く。散文の説明は持たない。テスト専用のモデル `_test_*` は presets.json に入れず `local/experiments/schema/test-models.ts` に TS の辞書で書き、仕方に差分を重ね、どの実験にも入れない。設定は `resolve.ts write` が `local/experiments/resolved/tests.json` に値で書き、pytest はそれを読む。
- 実験は presets.json でなく、実験群ごとに `local/experiments/config/<群>.yaml`（実験層）に書く。今の実験群は単一区画 `1comp`、19 区画 `19comp`、GPU の実測 `gpu`。実験層は二つの文書で、一つ目は frontmatter（実験群の道案内の一文 `description` だけ）、二つ目はどの run にも共通の設定 `setup` と回す run の並び `runs`（run の名前 → 何を回して何を見るかの一文 `description` と、setup を置き換える欄）。欄は仕方 `methods`、群分け（全部の群分けを組むゲートの並び `gates` か、群分けの名前 `groupings`）、系列 `series`、GPU の実測 `benches`（測る系列。元のモデルと run のモデルを全部、細胞の数 `cells` で、`ncu` なら Nsight Compute でも測る）。run に書いた欄だけが setup を置き換え、群分けは gates と groupings をまとめて置き換える。各実験群には、短い学習データと系列で数分で回り切るテスト用の run `test-<群>` を置き、形式やコードを変えたらまずこれを回す（仕方と系列は名前に `-test` を付けた差分で書く）。形は `local/experiments/schema/schema.ts` が zod で決め、返すのは解決済みの実験 (モデルと系列を値で持ち、回す組を並べたもの) だけで、`resolve.ts` と画面はこれだけを読み、実験層の形を知らない。仕方と系列はプリセットの名前か、名前を付けた差分 `{ "name", "from", ...差分 }` で指し、値を少し変えた実験はプリセットを足さずに差分で書く。モデルの名前は `<仕方の名前>.<群分け>` で、同じ名前がほかの実験で違う中身を指せば落とす。
- プリセットは組み合わせを展開する実験（群分けの掃引など）のためだけに使う。組み合わせを展開しない決め打ちの構成（発表の再現など）は、プリセットを通さず、解決済みの実験の形 (`local/experiments/schema/schema.ts` の `PlanJson`: 学習の設定と系列を値のまま) で `local/experiments/config/<実験>.json` に手で書き、`local/experiments/schema/experiments.ts` が読む。一回限りの値をプリセットに足さない。
- 実際に送る spec.json は、そのままの中身で `local/experiments/resolved/` に生成物として置く (`bun local/resolve.ts write`)。パスは識別のための名前で送らない: 学習データ `<実験>/data/<対象>.<電流>.<パラメータ>.json`、学習器 `<実験>/parts/<圧縮器と右辺>.<群>.json` (どちらも名前は中身から付け、同じ中身は実験の中で 1 ファイル。同じ名前に違う中身が来たら resolve が止まる)、モデル `<実験>/models/<モデル>.json`、`<実験>/sim/<系列>/original.json` と `<モデル>.json`、`<実験>/bench/<GPU の実測>.json`。段取り (`remote/run/`) は宣言を読まず、resolved/ の spec だけを回す。実験ごとの状態も resolved/ の下のディレクトリから組み、並びは名前の順 (宣言の順は手元が持つ) (指す run を先に回す: 学習器は学習データを、モデルは学習器を、評価と GPU の実測はモデルと (評価なら) 元のモデルを)。画面も仕事を resolved/ のパスで起こす。宣言やプリセットを直したら write で書き直し、差分で、送る spec が何から何に変わったか (中身が変われば置き場が変わり、回し直しになる) を確かめる。古いと `local/resolve.test.ts` が落ち、`local/remote.sh` は送らない (`resolve.ts check`)。
- 使わなくなった仕方と系列は消す。run は消さず、`list` の一覧に残る。
- 描き方 (どの区画を見るか、表示の範囲) は宣言に持たず、読む側 (repo の外の図) が持つ。

## 判断基準

迷ったら今の読みやすさを優先する。将来の柔軟性は根拠にならない。

- **抽象は浅く保つ。** 抽象は重複を消すためでなく、ずれを防ぐためだけに使う。読む人 (エージェントも本人も初見で読む) が辿る段を増やさない。
- **変種はデータで書く。** 新しい群分けや区画は設定か表に足し、変種ごとの関数を書かない。並びを手で揃える箇所を作らない。
- **約束はテストで持つ。** 守らせたいことは `remote/tests/`（Python）と各 TS の隣のテストに書き、文章でしか守られない規則を増やさない。テストは 20 秒以内。
  今あるのは、physics が surrogate を import しないことと、記録・設定・描画の道具を import しないこと (`remote/tests/test_conventions.py`)、恒等の圧縮と真の右辺を入れたサロゲートがどの群分けでも元のモデルと同じ膜電位になること (`remote/tests/test_surrogate.py`)、どのファイルにも docstring があり、`remote/neurosurrogate/AGENTS.md` が docstring と import から作り直したものと一致すること (`remote/tests/test_design.py`)。
- **数値を変えないリファクタは、新旧の出力を突き合わせる。** 同じ spec を変更前と変更後のコードで回し、run の中身 (学習の書き出し、指標、変数ごとの波形) が一致することを確かめる。
- **後方互換は取らない。** 保存形式を変えたら、変えた種類の版を `run.py` の `_FORMATS` と `spec.ts` の `FORMATS` で上げ、古い run は回し直す。互換の shim は入れない。
- **実行経路を二重に持たない。** 重い計算は `run.py`、回すのは送られてきた resolved/ の spec だけで段取りは `remote/run-spec.ts`、統計量を計算して小さくして返すのは `remote/read/` と `remote/serve.ts`、描画は repo の外の図だけ。
- **波形は回線に出さない。** runs は計算する機械に置いたまま、図は API に、間引いた線と指標を組んだ答えを頼む。帯域の細い回線（テザリング）でも回る量に保つ。
- **数値の振る舞いが変わる変更は、手を止めて報告する。**
- **git は明示指示があるときだけ。**

## 規則

- **TS と Python はコードを import し合わない。** TS が読むのはデータだけ（設定 JSON と runs の run）。シミュレータは TS にも二重にあってよく、ずれは共通の基準値のテストで見張る。
- 依存の向きは `physics ← surrogate ← run`。表は `remote/tests/test_conventions.py` の `_ALLOWED`。落ちたらテストを緩めずコードを直す。
- `remote/neurosurrogate/` は記録・設定の基盤（MLflow や Hydra）を知らない。run の保存形式を知るのは `run.py` と、置き場から学習器とモデルを読む `store.py` だけ。Python は数値計算と eFEL だけを持ち、それより先の統計量は `remote/read/comparison.ts`、段取りは TS が持つ。
- この repo は図や画面を持たず、外の部品に依存しない。結果を読む図は repo の外に置き、この repo の TS を import して描く。
- ファイルは動く側で置く。手元だけが使うものは `local/`、計算する機械だけが使うものは `remote/`、両方が使う形だけを `shared/` に置き、最上位にはほかを置かない。`local/` と `remote/` の直下には、直に走らせる入口と、役割ごとのディレクトリと、依存やイメージを決めるファイル（`package.json`、`Dockerfile`、`pyproject.toml` など）だけを置き、コードは役割のディレクトリに番号なしの名前で平らに置く。テストは隣の `<名前>.test.ts`、参照値は `<名前>.reference.json`。人が書く宣言（JSON と YAML）は `local/experiments/config/` に置き、コードは置かない。
- `_` 始まりの名前は「そのファイルの外から参照しない」印（`remote/tests/test_conventions.py` が見る）。
- ファイルの持つものは各ファイルの先頭の docstring に書く。`remote/neurosurrogate/AGENTS.md` は生成物なので手で書き換えない（`remote/tests/test_design.py` が作り直し、古いと落ちる）。
- 学習・評価と runs は ssh で入れる NVIDIA の GPU の機械（環境変数 `SINDY_HOST`）に置く。機械の名前と接続先は repo に書かない。run の meta.json の `env` に、回した backend・device・版・host・コード・イメージを書く。
- 大きな改装のあとは下のテストを通す。

## Commands

```bash
SINDY_HOST=<host> local/remote.sh up                  # remote/・shared/ と resolved/ の写しを送り、イメージを用意し、API を立てる
SINDY_HOST=<host> local/remote.sh run <段取りの引数>   # 段取り（下の run・list）を向こうで回す
SINDY_HOST=<host> local/remote.sh test                # テストを向こうで回す
SINDY_HOST=<host> local/remote.sh fetch <run の id>   # run を手元の runs/ へ持ってくる
# 手元の側（local/）
bun install --cwd local                  # 手元の TS の依存
bun local/resolve.ts write   # 宣言から送る spec.json を local/experiments/resolved/ に書き直す（check で確かめるだけ）
bun test local shared                    # 手元と受け渡す形の TS のテスト
SINDY_API=http://<host>:8787 bun local/cli.ts   # 実験を状態ごとに並べる
# 計算する機械の側（remote/。手元でもテストとテストの学習の評価は回る。gpu の学習 spec は GPU の機械でしか回らない）
# 手元で回すときは、読む spec の場所を export SINDY_RESOLVED=local/experiments/resolved で渡す（イメージは /code/resolved）
uv sync --project remote                 # Python の依存
bun remote/run-spec.ts run <resolved/ の下のパス>...   # その spec を回す。指す学習データ・学習器・モデルと元のモデルを先に回す
bun remote/run-spec.ts list           # 完成した run の一覧
uv run --project remote pytest -q remote/tests   # 規約、区画の式、サロゲート、neurosurrogate/run.py
bun test remote                          # 段取り・API・指標の TS のテスト
```

整形と lint は `ruff format` と `ruff check`、型は `remote/` で `uv run mypy .`。道具は [uv](https://docs.astral.sh/uv/) と [bun](https://bun.sh/)。計算する機械のイメージ（`remote/Dockerfile`）は、ベースと道具を digest、Python の依存を `uv.lock` で固定する。段取りと API の TS は npm の依存を持たない。
