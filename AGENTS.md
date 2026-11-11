---
description: "研究コードの repo の規則と設計。用語、run と設定の形式、判断基準、コマンド"
---

# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。全体の地図と実験の流れの図は `README.md`、
`neurosurrogate/` のファイルごとの持つものと依存の図は `neurosurrogate/AGENTS.md`（生成物）にある。設計に手を入れる前にこのファイルを読む。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 数値計算（元のモデル `neurosurrogate/physics/`、学習と置換 `neurosurrogate/surrogate/`、計算の入口 `neurosurrogate/run.py`、GPU の実測 `neurosurrogate/bench.py`） | `neurosurrogate/` |
| 人が書いて確かめる宣言（プリセット `presets.json` の既定 `defaults`・仕方 `methods`・系列 `series`、実験 `<主題>.yaml`） | `config/` |
| 宣言を読む部品（プリセットを混ぜる `presets.ts`、実験の形 `schema.ts`、判定 `checks.ts`、並べる `experiments.ts`） | `evaluation/experiments/` |
| spec の形、段取り | `evaluation/orchestrate/` |
| 結果を読む指標と間引き、Traub の TS シミュレータ | `evaluation/read/`、`evaluation/traub/` |
| 結果を小さい JSON で返す API と口の形 | `evaluation/serve/` |
| 計算する機械のイメージと、そこへ送って回す script | `Dockerfile`、`remote.sh` |
| run | 計算する機械の `~/sindy/runs/`（手元の runs は `remote.sh fetch` の写しだけ。どちらも Git の外） |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）
- 結果を読む図と画面
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

## 流れ

宣言 → 段取り（TS）→ 計算（Python）→ runs → API（TS）→ repo の外の画面、の順に渡る。図は `README.md` にある。置き場の間を渡るのはデータだけ:

| 渡るもの | 書く側 | 読む側 |
| --- | --- | --- |
| 実験の条件（spec.json） | TS（段取り） | Python |
| 実験の結果（runs/） | Python | TS（API） |
| 間引いた線と指標と主張の判定（API の JSON。口の形は `evaluation/serve/api.ts`） | TS（API） | 画面 |
| ゲートのレートの式（physics/ の JSON） | 人 | Python と TS |

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

保存の単位は学習したモデル (`runs/models/<学習のハッシュ>/`) と入力 (`runs/inputs/<入力のハッシュ>/`) の二つで、入力の下に元のモデル (`<入力のハッシュ>/original/`) と置き換えたモデル (`<入力のハッシュ>/<モデルの鍵>/`、学習 run のハッシュ + 置換範囲) を置く。1 run = spec.json 1 つで、置き場は spec から決まる (`orchestrate.ts` の `dirOf`)。同じ spec は同じ置き場なので、ディレクトリがあれば回さない。図は Python で描かず、Python が置くのは図に要るデータだけ:

```
runs/index.json                全 run の id + meta.json + spec (段取りが書く。図は API の index から一覧を作る)
<run>/  spec.json              回した条件 (format = 形式の版, kind, ...)
        meta.json              kind (train / original / replaced), name (図に出す名前。同一性に入らない), created, unit (掃引の単位),
                               env (回した backend・device・jax と python の版・host・コードとイメージの版。同一性に入らない)
models/<ハッシュ>/  surrogate/                config.json + parts.joblib (群ごとの圧縮器と右辺)
                    summary.json              surrogate.summary (学習側の指標)
                    description/model.json    右辺と圧縮器の中身 (describe)、ネット、列の名前。群が 2 つ以上なら groups に群ごとの中身
                    description/train/        学習データ (waves と同じ形式、点は 1 つ)
                    description/fit/<comp_id>/{x,latent,recon_rmse}.f64
                                              学習 comp ごとの同定器への入力・潜在・再構成 RMSE (列ごとに連続)
inputs/<ハッシュ>/  input.json                電流と掃引 (段取りが書く)
                    original/ waves/          下の形式。meta.json に comp 名 → id も入る
                              spikes.json     comp 名 → 点ごとのスパイクの特徴量 (全 comp・全点)
                    <モデルの鍵>/ waves/, spikes.json
                                  projected/<comp>/ 置換した comp の original/ のゲートを潜在空間へ射影したもの (waves と同じ形式)

waves/meta.json            t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]], nodes, names
waves/<点>/comp<id>.f64    その comp の変数 (features の並び順)。変数ごとに連続した little-endian float64
waves/<点>/I_ext.f64       waves/<点>/I_internal.f64 (node ごとに連続)
```

- spec はどちらも `device` (`cpu` か `gpu`) を持つ。同じ条件でも計算の順序が変わって結果がずれるので、同一性に入れる。学習は `gpu` (GPU を使うのは AE の学習だけ)、評価とテストの学習は `cpu`。決めるのは `orchestrate.ts` の `trainSpec` と `simSpec`
- spec の 2 種類: `train` は設定 (`surrogate` = `surrogate.Config`)、`sim` は入力 (`input`) とモデル (`model` = `{train, replace_targets}`、
  元のモデルは null)。置換範囲はモデルの側に入るので、置換範囲だけが違う系列は同じ入力のディレクトリに別のモデルとして並ぶ
- **ハッシュを組むのは TS だけ。** Python は渡された spec とディレクトリを埋めるだけで、学習 run は spec の `model.train`、
  比べる相手は隣の `<入力のハッシュ>/original/` で引く (JS と Python で数値の JSON 表記が違うので、鍵を 2 か所で組まない)
- **比べた束 (どの列を並べたか) は残さない。** 図が選び直すたびに描く
- 形式を変えたら `neurosurrogate/run.py` の `_FORMAT` と `spec.ts` の `FORMAT` を変える。spec に入るので、
  古い形式の run は置き場が合わず回し直される
- **記録した run を描画が書き換えない**: 図は読むだけ
- JSON に nan は書かない (null)。JS の JSON が読めないため
- **書きかけと印**: `<置き場>.partial/` は書きかけで、中の `pid` が書いているプロセス (生きていれば計算中、死んでいれば残骸として次に片付ける。同じ run を二つのプロセスが書かないよう、生きた書き手がいれば run.py は終了コード 4 で退き、段取りは終わるのを待つ)。`<置き場>.unreplaceable` は置換できないと分かったモデルの印 (置換できるかは Python だけが知るので答えを覚える)。`.jobs/<組のハッシュ>.{json,pid,log,exit}` は外の画面が `--job` で起こした段取りの中身・pid・ログ・終了コード (同じ組の二重起動を pid で弾き、終了コードが 0 でないか書かずに消えたものを失敗として返す)

### レートの式

ゲートのレート α, β は `neurosurrogate/physics/<モデル>.json` に 1 本 1 行で書き、Python (`cells.py`) と TS (`evaluation/traub/traub.ts`、Traub だけ) が同じ行から同じ式を組む。ほとんどは `rate(u) = A f(z)`、`z = (θ - u) / s` の形で、`f` は `exp` (e^z)、`sigmoid` (1 / (1 + e^z))、`linexp` (z / (e^z - 1)) のどれか。`u` は原著の基準電位からの電位 (Traub は V + 60、HH は静止電位から)。JSON には `f` の名前と `A`・`theta`・`s` だけを書く (例: Traub の m の開くレートは `"alpha": { "form": "linexp", "A": 1.28, "theta": 13.1, "s": 4.0 }`)。レートの演算コストもこの形から導く。c・r (電位で場合分けし、β を α との差で置く) と Ca で動く q はこの形に収まらない。

### 設定ファイル

- 学習の設定も評価の条件も `config/presets.json` に書く。値は既定 (`defaults`) も含めてすべて JSON に出し、コードは混ぜて形を確かめるだけにする (JSON だけで何を回すかが分かるように)。学習 run は 縮約の仕方 × 群分け で決まる。群分けは群をゲートの小文字の名前でつなぎ、群どうしを `_` で区切る。仕方と系列には既定と違う値だけを書き、`evaluation/experiments/presets.ts` が defaults → プリセット → 差分の順に混ぜて形を確かめる (`methodOf`・`trainFrom`・`seriesFrom`)。プリセットは JSON のままにし、YAML は実験の宣言だけに使う。
- 学習の設定 (`surrogate.Config`) は、区画の型 (`cell`)、`form`、学習データ (`data` = `SimSpec`)、群のリスト (`groups`)、学習を絞る区画 (`train_comp`)。
  群は `gates`・`compressor`・`d`・`rhs` と、それぞれの fit 関数へ `**` で展開する `compressor_params`・`rhs_params`。
  `gates` は状態の名前 (Traub は M N C A H B S R Q XI、HH は M H N)。`compressor: kam` は学習しない KAM 縮約で、右辺も `kam`、`d` は 1。UDE は AE を一緒に学習するので、圧縮器は AE に限る。SINDy のレート項は、`library_specs` に `gates` を書かなければ群の状態のものに絞る (`_own_rates`)。
  既定値は fit 関数の署名だけが持ち、綴り違いは TypeError になる。
- 値の既定は `config/presets.json` の `defaults` が持つ。値から導く既定 (教師は細胞と同じ、圧縮は仕方の名前、右辺は KAM なら KAM でほかは SINDy、項ライブラリは細胞のゲートの項と `basis`) と、どの学習でも同じ形 (学習用の電流・全区画で学習) だけは `evaluation/experiments/presets.ts` が導く。散文の説明は持たない。テスト専用の学習 `_test_*` は presets.json に入れず `evaluation/experiments/test-trains.ts` に TS の辞書で書き、仕方に差分を重ね、どの実験にも入れない。
- 実験は presets.json でなく `config/<主題>.yaml` に書く (主題は拡張 `mc`、改良の探索 `sweep`、性質の発見 `property`)。宣言の形は主題ごとに `evaluation/experiments/schema.ts` が zod で決め、主題の問いに合わせて自由に変えてよい。`schema.ts` が返すのは解決済みの実験 (学習 run と系列を値で持ち、回す組を並べたもの) だけで、段取り・API・画面はこれだけを読み、主題の形を知らない。仕方と系列はプリセットの名前か、名前を付けた差分 `{ "name", "from", ...差分 }` で指し、値を少し変えた実験はプリセットを足さずに差分で書く。学習 run の名前は `<仕方の名前>_<群分け>` で、同じ名前がほかの実験で違う中身を指せば落とす。今の主題の宣言は、実験ごとに何を回して何を見るかの一文 `description` と、群分け (`gates` か `groupings`) と仕方と、回す系列 `series` か回す前に書いた主張 (名前 → 読む系列 `on`、文 `claim`、絞る群分け) を持つ。回す系列は `series` と主張の `on` を合わせたもの。判定は探索の問い (改良の探索、性質の発見) にだけ置き、拡張 (`mc`) と GPU の実測には置かず、回す系列だけを書く。判定の関数は `evaluation/experiments/checks.ts` が主張の名前ごとに持ち、API の `claims` が runs から判定して返す。宣言をコードでなく YAML にするのは、コードを読まずに何を回して何を言うかを追えるようにするため（機械が書く runs と、値の presets.json は JSON）。主張と判定の線は結果を見てから書かない。
- 使わなくなった仕方と系列は消す。run は消さず、`list` の一覧に残る。
- 描き方 (どの区画を見るか、表示の範囲) は宣言に持たず、読む側 (repo の外の図) が持つ。

## 判断基準

迷ったら今の読みやすさを優先する。将来の柔軟性は根拠にならない。

- **抽象は浅く保つ。** 抽象は重複を消すためでなく、ずれを防ぐためだけに使う。読む人 (エージェントも本人も初見で読む) が辿る段を増やさない。
- **変種はデータで書く。** 新しい群分けや区画は設定か表に足し、変種ごとの関数を書かない。並びを手で揃える箇所を作らない。
- **約束はテストで持つ。** 守らせたいことは `tests/` に書き、文章でしか守られない規則を増やさない。テストは 20 秒以内。
  今あるのは、physics が surrogate を import しないことと、記録・設定・描画の道具を import しないこと (`tests/test_conventions.py`)、恒等の圧縮と真の右辺を入れたサロゲートがどの群分けでも元のモデルと同じ膜電位になること (`tests/test_surrogate.py`)、どのファイルにも docstring があり、`neurosurrogate/AGENTS.md` が docstring と import から作り直したものと一致すること (`tests/test_design.py`)。
- **数値を変えないリファクタは、新旧の出力を突き合わせる。** 同じ spec を変更前と変更後のコードで回し、run の中身 (学習の書き出し、指標、変数ごとの波形) が一致することを確かめる。
- **後方互換は取らない。** 保存形式を変えたら `run.py` の `_FORMAT` と `spec.ts` の `FORMAT` を上げ、古い run は学習し直す。互換の shim は入れない。
- **実行経路を二重に持たない。** 重い計算は `run.py`、段取りは `orchestrate.ts`、結果を小さくして返すのは `evaluation/serve/serve.ts`、描画は repo の外の図だけ。
- **波形は回線に出さない。** runs は計算する機械に置いたまま、図は API に間引いた線と指標を頼む。帯域の細い回線（テザリング）でも回る量に保つ。
- **数値の振る舞いが変わる変更は、手を止めて報告する。**
- **git は明示指示があるときだけ。**

## 規則

- **TS と Python はコードを import し合わない。** TS が読むのはデータだけ（設定 JSON と runs の run）。シミュレータは TS にも二重にあってよく、ずれは共通の基準値のテストで見張る。
- 依存の向きは `physics ← surrogate ← run`。表は `tests/test_conventions.py` の `_ALLOWED`。落ちたらテストを緩めずコードを直す。
- `neurosurrogate/` は記録・設定の基盤（MLflow や Hydra）を知らない。run の保存形式を知るのは `run.py` だけ。Python は数値計算だけを持ち、指標は `evaluation/read/compare.ts`、段取りは TS が持つ。
- この repo は図や画面を持たず、外の部品に依存しない。結果を読む図は repo の外に置き、この repo の TS を import して描く。
- `evaluation/` には TS だけを役割ごとのディレクトリに番号なしの名前で置く。テストは隣の `<名前>.test.ts`、参照値は `<名前>.reference.json`。人が書く宣言（JSON と YAML）は `evaluation/` でなく `config/` に置き、コードを置かない。
- `_` 始まりの名前は「そのファイルの外から参照しない」印（`tests/test_conventions.py` が見る）。
- ファイルの持つものは各ファイルの先頭の docstring に書く。`neurosurrogate/AGENTS.md` は生成物なので手で書き換えない（`tests/test_design.py` が作り直し、古いと落ちる）。
- 学習・評価と runs は ssh で入れる NVIDIA の GPU の機械（環境変数 `SINDY_HOST`）に置く。機械の名前と接続先は repo に書かない。run の meta.json の `env` に、回した backend・device・版・host・コード・イメージを書く。
- 大きな改装のあとは下のテストを通す。

## Commands

```bash
SINDY_HOST=<host> ./remote.sh up                  # コードを送り、イメージを用意し、API を立てる
SINDY_HOST=<host> ./remote.sh run <段取りの引数>   # 段取り（下の train・eval・list）を向こうで回す
SINDY_HOST=<host> ./remote.sh test                # テストを向こうで回す
SINDY_HOST=<host> ./remote.sh fetch <run の id>   # run を手元の runs/ へ持ってくる
uv sync                        # 手元の依存導入（テストと、テストの学習の評価は手元でも回る。gpu の学習 spec は GPU の機械でしか回らない）
# 学習と評価
bun evaluation/orchestrate/orchestrate.ts train <学習 run>...             # 学習。学習 run は実験の中の名前 (kam_hn) か tests の名前
bun evaluation/orchestrate/orchestrate.ts eval <系列> <学習 run かハッシュ>...  # 評価。入力を元のモデルと置き換えたモデルで回す。未学習なら学習もする
bun evaluation/orchestrate/orchestrate.ts list                            # run の一覧。引数なしで仕方・テスト・系列の一覧、index で index.json を書き直す
uv run pytest -q               # 規約、区画の式、サロゲート、neurosurrogate/run.py
bun test evaluation experiments  # TS のテスト（依存は bun install）
```

整形と lint は `uv run ruff format` と `uv run ruff check`、型は `uv run mypy`。道具は [uv](https://docs.astral.sh/uv/) と [bun](https://bun.sh/)。計算する機械のイメージ（`Dockerfile`）は、ベースと道具を digest、依存を `uv.lock` と `bun.lock` で固定する。
