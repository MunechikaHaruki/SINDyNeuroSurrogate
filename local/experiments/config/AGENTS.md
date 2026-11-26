---
description: "実験の宣言（基準のファイルと実験群の YAML）の書き方。宣言を書くときに読む"
---

# 実験の宣言

- 宣言は、送る形そのままの JSON を基準のファイルとして置き、実験群の YAML にそこからの差分を書く。学習器の基準 `parts/<名前>.json` は学習の設定 (`../../../remote/neurosurrogate/surrogate/model.py` の `Config`) から群の並びを除き、ゲートを持たない群 `group` を 1 つ持つ形 (群分けが群の数だけ並べ、ゲートを入れる)。評価の系列の基準 `inputs/<名前>.json` は入力 (`spec`・`param`・`values`) と置き換える区画 (`replace_targets`)。比べる物差しは `measures.json`。値の既定 (学習の電流と長さ、項ライブラリ、刻み、刺激する区画など) はすべて基準のファイルに書き、コードは差分を当てて形を確かめるだけにする (ファイルを読めば何を送るかが分かるように)。
- 差分は JSON Merge Patch (RFC 7386: オブジェクトは欄ごとに重ね、`null` はその欄を消し、配列と値は置き換える)。当てた結果を送る形の型 (`../schema/base.ts` の `Part`・`SeriesShape`) で確かめるので、差分に書けるのは送る形にある欄だけ。`null` は欄を消す印なので、`null` という値 (`param`・`train_comp`) は差分で書けず、要るなら基準のファイルを足す。どの学習器・系列にも重ねる差分は、個別の差分の後に、群の差分、実験の差分の順に当てる。開いた辞書 (`current_params`・`compressor_params`・`rhs_params`) の中の綴りと、圧縮・右辺・項ライブラリの組み合わせは TS では確かめず、Python の署名と pytest が落とす。
- 名前は記号に詰めず、軸ごとに綴る (単語は `-`、群の中のゲートは小文字を `+`、群どうしは `_`、学習器と群分けは `.`、系列と置き換える範囲は `@` でつなぐ。例 学習器 `kam-absorb-m`・`pca-sindy`、群分け `h+n_a+c`、系列 `traub-19comp-steady@na-dr`、実験 `all-groupings-1comp`)。
- 群の `gates` は状態の名前 (Traub は M N C A H B S R Q XI、HH は M H N)。`compressor: kam` は学習しない KAM 縮約で、右辺も `kam`、`d` は 1。UDE は AE を一緒に学習するので、圧縮器は AE に限る。SINDy のレート項は、`library_specs` に `gates` を書かなければ群の状態のものに絞る (`_own_rates`)。既定値は fit 関数の署名だけが持ち、綴り違いは TypeError になる。
- テスト専用のモデル `_test_*` は `../schema/test-models.ts` に TS の辞書で書き、学習器の基準に差分を重ね、どの実験にも入れない。設定は `resolve.ts write` が `../resolved/tests.json` に値で書き、pytest はそれを読む。
- 実験群の YAML は二つの文書で、一つ目は frontmatter（実験群の道案内の一文 `description` だけ）、二つ目は群の基準 `base` と実験の並び `runs`（実験の id → 書き換える欄。書き換えなければ `{}`）。実験は base に書いた欄を丸ごと置き換え、差分の欄 (`part`・`input`) だけは群の差分の上に重ねる。宣言はパラメタだけを持ち、何を回して何を見る実験かの説明は冊の側（親 repo の `outputs/experiments/groups/<群>.yaml` の `experiments`）が持つ。
- 欄の意味は `../schema/schema.ts` の `FIELDS` の注釈にある。回す組は 学習器 × 群分け × 系列 の直積。値を少し変えた実験は基準のファイルを足さずに差分で書く。
- 各実験群には、短い学習データと系列で数分で回り切るテスト用の実験 `test-<群>` を置き、形式やコードを変えたらまずこれを回す。返すのは解決済みの実験 (モデルと系列を値で持ち、回す組を並べたもの) だけで、`resolve.ts` と画面はこれだけを読み、宣言の形を知らない。モデルの名前は `<学習器の名前>.<群分け>`、モデルと系列の名前は実験の中で一意で、実験をまたいでは同じ名前が違う中身を指してよい。
- 実際に送る spec.json は、そのままの中身で `../resolved/` に生成物として置く (`bun local/resolve.ts write`)。パスは識別のための名前で送らない (名前の付け方は `../../resolve.ts`)。段取り (`../../../remote/run/`) は宣言を読まず、resolved/ の spec だけを回し、指す run を先に回す。画面も仕事を resolved/ のパスで起こす。宣言を直したら write で書き直し、差分で、送る spec が何から何に変わったか (中身が変われば置き場が変わり、回し直しになる) を確かめる。古いと `../../resolve.test.ts` が落ち、`../../remote.sh` は送らない (`resolve.ts check`)。
- 使わなくなった基準のファイルは消す。今の宣言（resolved/）が指さなくなった run（形式の版が古いものも）は、計算する機械の runs/ から消してよい。
- 描き方 (どの区画を見るか、表示の範囲) は宣言に持たず、読む側 (repo の外の図) が持つ。
