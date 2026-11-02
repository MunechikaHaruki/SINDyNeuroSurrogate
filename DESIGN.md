# 設計

研究コードの設計に手を入れる前に読む。用語、run と設定の形式、どちらへ倒すかの判断基準を持つ。ファイルごとの責務は各ファイルの docstring と [neurosurrogate/FILES.md](neurosurrogate/FILES.md) にある。

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
runs/index.json                全 run の id + meta.json + spec (段取りが書く。図はここから一覧を作る)
<run>/  spec.json              回した条件 (format = 形式の版, kind, ...)
        meta.json              kind (train / original / replaced), name (図に出す名前。同一性に入らない), created, unit (掃引の単位)
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

- 学習の設定も評価の条件も `conf.json` に書く。学習 run は 縮約の仕方 (`methods`) × 群分け で決まり、名前は `<仕方>_<群分け>` (`kam_hn`、`pca_hn_ac`)。群分けは群をゲートの小文字の名前でつなぎ、群どうしを `_` で区切る。学習 run を一本ずつ書き下さず、`conf.ts` が名前から組む。`conf.json` には既定と違う値だけを書き、`conf.ts` が既定を埋めて形を確かめ、仕方に群分けを重ねる。YAML の設定ファイルは持たない。
- 学習の設定 (`surrogate.Config`) は、区画の型 (`cell`)、`form`、学習データ (`data` = `SimSpec`)、群のリスト (`groups`)、学習を絞る区画 (`train_comp`)。
  群は `gates`・`compressor`・`d`・`rhs` と、それぞれの fit 関数へ `**` で展開する `compressor_params`・`rhs_params`。
  `gates` は状態の名前 (Traub は M N C A H B S R Q XI、HH は M H N)。`compressor: kam` は学習しない KAM 縮約で、右辺も `kam`、`d` は 1。UDE は AE を一緒に学習するので、圧縮器は AE に限る。SINDy のレート項は、`library_specs` に `gates` を書かなければ群の状態のものに絞る (`_own_rates`)。
  既定値は fit 関数の署名だけが持ち、綴り違いは TypeError になる。
- 既定 (細胞は Traub、教師は細胞と同じ、圧縮は仕方の名前、右辺は KAM なら KAM でほかは SINDy、項ライブラリは細胞のゲートの項と `basis`、系列は単一区画の soma への定常電流で soma を置き換える) と、どの学習でも同じ値 (学習用の電流・刻み・全区画で学習) は `conf.ts` が埋める。宣言を選んだ理由は各宣言の `description` に書き、spec には入れない。テスト専用の学習 `_test_*` は conf.json に入れず `evaluation/orchestrate/test-trains.ts` に TS の辞書で書き、仕方に差分を重ね、どの実験にも入れない。
- 使わなくなった仕方と系列は消す。run は消さず、`list` の一覧に残る。
- 描き方 (どの区画を見るか、表示の範囲) は宣言に持たず、読む側 (repo の外の図) が持つ。

## 判断基準

迷ったら今の読みやすさを優先する。将来の柔軟性は根拠にならない。

- **抽象は浅く保つ。** 抽象は重複を消すためでなく、ずれを防ぐためだけに使う。読む人 (エージェントも本人も初見で読む) が辿る段を増やさない。
- **変種はデータで書く。** 新しい群分けや区画は設定か表に足し、変種ごとの関数を書かない。並びを手で揃える箇所を作らない。
- **約束はテストで持つ。** 守らせたいことは `tests/` に書き、文章でしか守られない規則を増やさない。テストは 20 秒以内。
  今あるのは、physics が surrogate を import しないことと、記録・設定・描画の道具を import しないこと (`tests/test_conventions.py`)、恒等の圧縮と真の右辺を入れたサロゲートがどの群分けでも元のモデルと同じ膜電位になること (`tests/test_surrogate.py`)、どのファイルにも docstring があり、`neurosurrogate/FILES.md` が docstring と import から作り直したものと一致すること (`tests/test_design.py`)。
- **数値を変えないリファクタは、新旧の出力を突き合わせる。** 同じ spec を変更前と変更後のコードで回し、run の中身 (学習の書き出し、指標、変数ごとの波形) が一致することを確かめる。
- **後方互換は取らない。** 保存形式を変えたら `run.py` の `_FORMAT` と `spec.ts` の `FORMAT` を上げ、古い run は学習し直す。互換の shim は入れない。
- **実行経路を二重に持たない。** 重い計算は `run.py`、段取りは `orchestrate.ts`、描画は repo の外の図だけ。
- **数値の振る舞いが変わる変更は、手を止めて報告する。**
- **git は明示指示があるときだけ。**
