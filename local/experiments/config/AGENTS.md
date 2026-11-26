---
description: "実験の宣言（基準のファイルと実験群の YAML）の書き方。宣言を書くときに読む"
---

# 実験の宣言

- 宣言は、送る形そのままの JSON を基準として 1 つずつ置き、実験群の YAML にそこからの差分を書く。学習器の基準 `learner.json` は学習の設定 (`../../../remote/neurosurrogate/surrogate/model.py` の `Config`) から群の並びを除き、ゲートを持たない群 `group` を 1 つ持つ形 (群分けが群の数だけ並べ、ゲートを入れる)。評価の系列の基準 `input.json` は電流 1 本 (`spec`) と置き換える区画 (`replace_targets`)。比べる物差しは `measures.json`。値の既定 (学習の電流と長さ、項ライブラリ、刻み、刺激する区画など) はすべて基準に書き、仕方 (KAM・AE など)、電流の種類、細胞の違いは YAML の差分に書く。コードは差分を当てて形を確かめるだけにする (ファイルを読めば何を送るかが分かるように)。
- 差分は JSON Merge Patch (RFC 7386: オブジェクトは欄ごとに重ね、`null` はその欄を消し、配列と値は置き換える)。当てた結果を送る形の型 (`../schema/base.ts` の `Part`・`SeriesShape`) で確かめるので、差分に書けるのは送る形にある欄だけ。`null` は欄を消す印なので、`null` という値 (`train_comp`) は差分で書けず、要るなら基準を直す。実験の `train` (学習の電流への差分) は、学習器の差分の後に当てる。開いた辞書 (`current_params`・`compressor_params`・`rhs_params`) の中の綴りと、圧縮・右辺・項ライブラリの組み合わせは TS では確かめず、Python の署名と pytest が落とす。
- 掃引 (どの電流パラメータをどの値で振るか) は送る形に無く、群の YAML の系列の指し方 `{ sweep: { <電流パラメータ>: [値...] }, ...差分 }` だけに書く。振るパラメータは 1 つで、`current_params` に同じ欄を書かない。resolve は値ごとに `current_params` へ入れた入力 1 本に開き、1 点を run 1 本として `sim/<系列>/<original かモデル>/<パラメータ>=<値>.json` に置く (掃引しない系列は `single.json`。開き方は `../../../shared/spec.ts` の `pointsOf` だけが持つ)。点を足しても、ほかの点の run は回し直さない。GPU の実測は系列の点を細胞に順に割り当てる。
- 名前は人が付けず、中身から決まる。学習器は名前が宣言そのもので `<圧縮器>-<右辺>[-d<潜在>][-absorb-<ゲート>]` (圧縮器と右辺が同じなら 1 語。例 `kam`・`kam-absorb-m`・`pca-sindy`・`ae-sindy-d5`。`../schema/base.ts` の `learnerDiff` が差分に開き、書き方は一通り)。名前に出ない設定は宣言に書けないので、要るなら文法か基準を直す。系列は `<細胞>-<電流の種類 (& の後ろ)>@<置き換える区画を + でつないだもの>` (例 `traub-steady@soma`・`traub19-steady@c05+c07+soma+c09+c11`。学習の系列も `traub-train@soma`)、モデルは `<学習器>.<群分け>` (群の中のゲートは小文字を `+`、群どうしは `_`)、掃引の点は `<パラメータ>=<値>`。実験の id (`all-groupings-1comp`) だけは人が付ける。区画の組などに人が読むラベルが要るなら、宣言でなく見せる側で別名を付ける。
- 群の `gates` は状態の名前 (Traub は M N C A H B S R Q XI、HH は M H N)。`compressor: kam` は学習しない KAM 縮約で、右辺も `kam`、`d` は 1。UDE は AE を一緒に学習するので、圧縮器は AE に限る。SINDy のレート項は、`library_specs` に `gates` を書かなければ群の状態のものに絞る (`_own_rates`)。既定値は fit 関数の署名だけが持ち、綴り違いは TypeError になる。
- テスト専用のモデル `_test_*` は `../schema/test-models.ts` に TS の辞書で書き、学習器の基準に差分を重ね、どの実験にも入れない。設定は `resolve.ts write` が `../resolved/tests.json` に値で書き、pytest はそれを読む。
- 実験群の YAML は二つの文書で、一つ目は frontmatter（実験群の道案内の一文 `description` だけ）、二つ目は実験の並び `experiments`（実験の id → 欄）。実験は欄を全部自分で書き、実験どうしで同じものは YAML のアンカー（`&名前` で付け、`*名前` で引き、`<<: *名前` で重ねる）で使い回す（上書きの規則はコードに持たない）。宣言はパラメタだけを持ち、何を回して何を見る実験かの説明は冊の側（親 repo の `outputs/experiments/groups/<群>.yaml` の `experiments`）が持つ。
- 欄 (`learners`・`train`・`groupings`・`inputs`・`bench`) の意味は `../schema/schema.ts` の `Setting` の注釈にある。学習器は名前で、系列は基準からの差分 (`{ sweep, replace_targets, spec }` のうち要るもの、基準のままなら `{}`) で書き、`bench` の `series` は系列の名前で指す。置き換える区画を変えた系列は、同じ電流の差分に `replace_targets` を足して並べる。回す組は 学習器 × 群分け × 系列 の直積。
- 各実験群には、短い学習データと系列で数分で回り切るテスト用の実験 `test-<群>` を置き、形式やコードを変えたらまずこれを回す。返すのは解決済みの実験 (モデルと系列を値で持ち、回す組を並べたもの) だけで、`resolve.ts` と画面はこれだけを読み、宣言の形を知らない。モデルと系列の名前は実験の中で一意で、実験をまたいでは同じ名前が違う中身を指してよい。
- 実際に送る spec.json は、そのままの中身で `../resolved/` に生成物として置く (`bun local/resolve.ts write`)。パスは識別のための名前で送らない (名前の付け方は `../../resolve.ts`)。段取り (`../../../remote/run/`) は宣言を読まず、resolved/ の spec だけを回し、指す run を先に回す。画面も仕事を resolved/ のパスで起こす。宣言を直したら write で書き直し、差分で、送る spec が何から何に変わったか (中身が変われば置き場が変わり、回し直しになる) を確かめる。古いと `../../resolve.test.ts` が落ち、`../../remote.sh` は送らない (`resolve.ts check`)。
- 今の宣言（resolved/）が指さなくなった run（形式の版が古いものも）は、計算する機械の runs/ から消してよい。
- 描き方 (どの区画を見るか、表示の範囲) は宣言に持たず、読む側 (repo の外の図) が持つ。
