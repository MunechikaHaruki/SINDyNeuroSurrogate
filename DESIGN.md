# 設計

研究コードの設計に手を入れる前に読む。用語、ファイルの責務、どちらへ倒すかの判断基準と、これから広げるときの前提を持つ。

## 用語

**群 (Group)**:
区画の状態のうち、まとめて 1 つの圧縮器と右辺で置き換える組。状態の名前（`cells` の `states`）で書く。

**form**:
`hybrid` は群に入らなかった状態と膜電位を元の式のまま解く。`sindy` は膜電位も含めて丸ごと同定し、群は全状態の 1 つだけ。

**物理の状態 (physics)**:
hybrid で群に入らなかった状態。元のモデルの微分の式 (`CellType.derivs`) で解く。

## 構成

### ファイル

`neurosurrogate/` は元のモデルを解く `physics/` と、学習して置き換える `surrogate/` に分け、スパイクの特徴量を取る `spikes.py` と入口の `run.py` を直下に置く。physics は surrogate を import しない (`tests/test_conventions.py` の `_ALLOWED`)。ファイルは上から順に読める関数で書く。

| ファイル | 持つもの |
| --- | --- |
| `physics/cells.py` | 区画の式。状態の名前ごとの微分 (`derivs`)、膜電位の式 (`dv`)、初期値、状態ごとの演算コスト。kernel はそこから組む (`physical_cell`)。HH・Traub・passive、Traub19 の区画の定数とネット (`NETS`)。レートの式は隣の `hh.json`・`traub.json` |
| `physics/currents.py` | 注入電流の名前 → 波形 |
| `physics/kam.py` | KAM 縮約の ψ 群。学習せず、元の区画の式から ψ ↔ ゲートの写しと dψ/dt を組む。サロゲートの群の圧縮器と右辺 (`kam`) として使い、学習した圧縮と同じ段取りで比べる |
| `physics/simulate.py` | ネットを解く前進 Euler (`simulate`)、波形の Dataset の読み書き、実験の記述 (`SimSpec`・`EvalSeries`) |
| `surrogate/compress.py` | 圧縮器 (PCA、AutoEncoder) |
| `surrogate/sindy.py` | SINDy の項のカタログと、列への束縛、疎回帰 |
| `surrogate/ude.py` | 潜在の右辺を MLP にし、AutoEncoder と一緒に学習する |
| `surrogate/model.py` | 設定 (`Config`・`Group`)、学習 (`fit`)、置換に使う区画の型 (`surrogate_cell`)、置換の可否と置換、中身の書き出し |
| `spikes.py` | 1 本の波形のスパイクの特徴量 (eFEL)。元のモデルとの比較は図の `evaluation/read/compare.ts` が持つ |
| `run.py` | 計算の入口。spec.json → run のディレクトリ。run の保存形式を知るのはここだけ |

```
evaluation/orchestrate/conf.ts        何を回すか: preset (群分けを含む)、評価の系列、見比べる組
evaluation/orchestrate/spec.ts        spec.json の形
evaluation/orchestrate/orchestrate.ts 段取り: 宣言 → spec.json → ハッシュ → 置き場が無ければ run.py を呼ぶ
evaluation/orchestrate/api.ts         操作盤の口
evaluation/read/                      図が runs/ を読む口と、元のモデルと比べる指標 (compare.ts)
evaluation/traub/traub.ts             Traub の単一区画の TS シミュレータ
runs/                                 実験の結果 (Git の外)
```

サロゲートの kernel は一つの汎用関数で、群ごとに潜在を decode して状態を名前で埋め、残りの状態は元の式で微分し、名前で組み直す。初期値は元のモデルの初期値と圧縮器の `gate_inits`、演算コストは群の圧縮器と右辺、膜電位の式、物理の状態のコストの和。群分けを足すのに書くのは設定だけ。

### run の置き場

保存の単位はモデルと入力の二つ。学習したモデルは `runs/models/<学習のハッシュ>/`、入力 (電流と掃引 =
系列から置換範囲を除いたもの) ごとの結果は `runs/inputs/<入力のハッシュ>/` の下に、元のモデルなら
`original/`、置き換えたモデルなら `<モデルの鍵>/` (学習 run のハッシュ + 置換範囲) で置く。1 run = spec.json 1 つで、
置き場は spec から決まる (`orchestrate.ts` の `dirOf`)。同じ spec は同じ置き場なので、ディレクトリがあれば回さない。
run は `runs/` の下のパス (id) で指す。図は Python で描かず、`evaluation/05-evaluation` (評価) と
`04-trained-model` (学習 run) が `runs.ts` を通して、口が宣言から組んだ id で読む。Python が置くのは図に要る
データだけ:

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
  比べる相手は隣の `original/` で引く (JS と Python で数値の JSON 表記が違うので、鍵を 2 か所で組まない)
- **比べた束 (どの列を並べたか) は残さない。** 図が選び直すたびに描く
- 形式を変えたら `neurosurrogate/run.py` の `_FORMAT` と `spec.ts` の `FORMAT` を変える。spec に入るので、
  古い形式の run は置き場が合わず回し直される
- **記録した run を描画が書き換えない**: 図は読むだけ
- JSON に nan は書かない (null)。JS の JSON が読めないため
- **書きかけと印**: `<置き場>.partial/` は書きかけで、中の `pid` が書いているプロセス (生きていれば計算中、死んでいれば残骸として次に片付ける。同じ run を二つのプロセスが書かないよう、生きた書き手がいれば run.py は終了コード 4 で退き、段取りは終わるのを待つ)。`<置き場>.unreplaceable` は置換できないと分かったモデルの印 (置換できるかは Python だけが知るので答えを覚える)。`.jobs/<組のハッシュ>.{json,pid,log,exit}` は操作盤のボタンが起こした段取りの中身・pid・ログ・終了コード (同じ組の二重起動を pid で弾き、終了コードが 0 でないか書かずに消えたものを失敗として操作盤に出す)

### 設定ファイル

- 学習の設定も評価の条件も `evaluation/orchestrate/conf.ts` に型付きで書く。YAML の設定ファイルは持たない。
- 学習の設定 (`surrogate.Config`) は、区画の型 (`cell`)、`form`、学習データ (`data` = `SimSpec`)、群のリスト (`groups`)、学習を絞る区画 (`train_comp`)。
  群は `gates`・`compressor`・`d`・`rhs` と、それぞれの fit 関数へ `**` で展開する `compressor_params`・`rhs_params`。SINDy のレート項は、`library_specs` に `gates` を書かなければ群の状態のものに絞る (`_own_rates`)。
  既定値は fit 関数の署名だけが持ち、綴り違いは TypeError になる。
- preset は継承しない。各 preset が決める値を全部書き、どの preset でも同じ値 (学習用の電流・刻み・全区画で学習) だけを
  `expand` が埋める。掃引は `d` の配列だけ。`_test_*` はテスト専用の preset で、どの実験にも入れない。
- 使わなくなった preset と系列は消す。run は消さず、操作盤の一覧に「実験に無い」と出る。
- 描き方 (どの区画を見るか、表示の範囲) は宣言に持たず、evaluation/ の図が持つ。

## 判断基準

迷ったら今の読みやすさを優先する。将来の柔軟性は根拠にならない。

- **抽象は浅く保つ。** 抽象は重複を消すためでなく、ずれを防ぐためだけに使う。読む人 (エージェントも本人も初見で読む) が辿る段を増やさない。
- **変種はデータで書く。** 新しい群分けや区画は設定か表に足し、変種ごとの関数を書かない。並びを手で揃える箇所を作らない。
- **約束はテストで持つ。** 守らせたいことは `tests/` に書き、文章でしか守られない規則を増やさない。テストは 20 秒以内。
- **数値を変えないリファクタは、新旧の出力を突き合わせる。** 同じ spec を変更前と変更後のコードで回し、run の中身 (学習の書き出し、指標、変数ごとの波形) が一致することを確かめる。
- **後方互換は取らない。** 保存形式を変えたら `run.py` の `_FORMAT` と `spec.ts` の `FORMAT` を上げ、古い run は学習し直す。互換の shim は入れない。
- **実行経路を二重に持たない。** 重い計算は `run.py`、段取りは `orchestrate.ts`、描画は evaluation/ の図だけ。
- **数値の振る舞いが変わる変更は、手を止めて報告する。**
- **git は明示指示があるときだけ。**

## KAM 縮約を単一区画の外へ広げるときの前提

Kepler-Abbott-Marder 縮約を Traub の単一区画から Multi-Compartment と solver へ広げるときに守ること。

### Multi-Compartment

- 縮約後も膜電位を残す（φ 群の重み α₀=1）。区画間の結合式をそのまま使える。
- 群分けは区画ごとに確かめ直す。g_Na=g_DR=0 の区画では h,n が電流に効かないので群から除く。
- 伝播が保たれるかは元の系との比較で見る。

### solver との組み合わせ

- m を膜電位へ吸収する縮約と Rush–Larsen は、どちらも m の刻みの制約を外す。組み合わせても刻みの得は掛け算にならない。

## Python と TS のニューロン設定

ニューロン固有の設定（パラメータと、指数・シグモイド・線形×指数の三つの形に収まるレート）だけを一つの JSON に置き、Python と TS が読む。
JSON は Python のパッケージの中の `neurosurrogate/physics/<モデル>.json` に置き、Python はパッケージの中だけで完結する。TS は `evaluation/traub/traub.ts` から `traub.json` を import する（HH は TS 側に実装が無いので Python だけが読む）。今はレート（HH は 6 本、Traub は m・h・n・a・b・s の 12 本）と Traub の基準電位だけで、密度などのパラメータはまだ両方に手書きしている。レートの演算コストも JSON の形から導く。
シミュレータ本体と三つの形の関数は両方で手書きする。形に収まらない c・r・q のレートは両方で手書きし、共通の基準値のテストで見張る。
重い計算と保存は Python、段取りは TS、描画は Svelte が runs/ から読む。

## 原著とずれている箇所

- Traub19 の basal と apical の寸法が原著と逆（2026-09-28 原著 p.637 で確認）。原著は apical（10 区画）が 2.89 µm・120 µm・2188 µm²、basilar（8 区画）が 2.42 µm・110 µm・1673 µm²。
  コードは index 0..7 に apical の寸法、9..18 に basilar の寸法を入れている。φ（7769 と 5941）の比 1.308 は面積比 2188/1673 と一致し、φ は原著どおり 1..8 が basilar。
  参考にした snsbook の code/part1/multi/traub.c も同じく逆で、そこから引き継いだ。逆のままだと入力区画（原著 3・15）と basal・soma・apical の添字がずれる。
