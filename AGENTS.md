# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（`neurosurrogate/physics/` 元のモデルを解く、`neurosurrogate/surrogate/` 学習して置き換える、直下にスパイクの特徴量 `spikes.py` と入口 `run.py`） | `neurosurrogate/` |
| 計算の入口（spec.json → run のディレクトリ。`python -m neurosurrogate.run`） | `neurosurrogate/run.py` |
| GPU の実測（細胞を並べて GPU で回し、時間・VRAM・ncu の L1 と L2 を測る。run.py の `bench` が細胞の数ごとに別のプロセスで呼ぶ） | `neurosurrogate/bench.py` |
| 回せるもののプリセット（既定 `defaults`、縮約の仕方 `methods`、系列 `series` の値をすべて持つ。テスト専用の学習は `evaluation/orchestrate/test-trains.ts` が持つ。`evaluation/orchestrate/conf.ts` がプリセットと差分を混ぜて学習 run と系列にする部品を持つ） | `presets.json` |
| プリセットを混ぜる部品（`conf.ts`）と spec.json の形（`spec.ts`）、段取り（`orchestrate.ts`。外の画面は export した関数、手元では CLI で、同じ関数を呼ぶ） | `evaluation/orchestrate/` |
| 実験の結果（学習したモデルは `models/<ハッシュ>/`、入力ごとの結果は `inputs/<入力のハッシュ>/` の下に元のモデルと置き換えたモデル）と run の一覧 | 計算する機械の `~/sindy/runs/`（手元の runs は `remote.sh fetch` で持ってきた写しだけで、どちらも Git の外） |
| 実験。分類（`experiments/mc/` 拡張、`experiments/sweep/` 改良の探索、`experiments/property/` 性質の発見）ごとに、宣言 `experiments.json`、その形 `schema.ts`（主題が自由に決め、解決済みの実験を返す）、判定 `checks.ts`（`experiments/mc/` は判定を持たない）。主題が使う部品は `parts.ts`、共通の型と線 `GAMMA_LINE` は `common.ts`、並べて名前で引けるようにするのは `experiments.ts` | `experiments/` |
| 結果を読む TS（元のモデルと比べる指標 `compare.ts`、学習の書き出しの読み方 `model-json.ts`、波形の切り分けと間引き `waves.ts`、Traub の TS シミュレータ） | `evaluation/read/`、`evaluation/traub/` |
| runs の隣で結果を小さい JSON にして返す API（`serve.ts`）と口の形（`api.ts`） | `evaluation/serve/` |
| 計算する機械のイメージ（ベースと道具は digest、Python の依存は `uv.lock`、TS の依存は `bun.lock` で固定） | `Dockerfile` |
| `evaluation/` の TS の依存（Traub の TS シミュレータの ODE ソルバー odex） | `package.json`、`bun.lock` |
| 計算する機械へコードを送り、イメージを用意し、API と段取りを向こうで回す script | `remote.sh` |
| 用語、run と設定の形式、設計の判断基準 | `DESIGN.md` |
| ファイルごとの持つものと依存の図（生成物。正本は各ファイルの先頭の docstring。`uv run python tests/test_design.py` で作り直す） | `neurosurrogate/FILES.md` |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）
- 結果を読む図と画面
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**TS と Python はコードを import し合わない。** TS が読むのはデータだけ（設定 JSON と runs の run）。
シミュレータは TS にも二重にあってよく、ずれは共通の基準値のテストで見張る。

## TS

- この repo は図や画面を持たず、外の部品に依存しない。単独で動く研究コードだけを置く。結果を読む図は repo の外に置き、この repo の TS を import して描く。
- `evaluation/` には TS だけを、役割ごとのディレクトリ（`evaluation/orchestrate/` 何を回すかと段取り、`evaluation/read/` 比べる指標と学習の書き出しの読み方、`evaluation/serve/` 結果を返す API、`evaluation/traub/` TS シミュレータ）に番号なしの名前で置く。テストは隣の `<名前>.test.ts`、参照値は `<名前>.reference.json`。`evaluation/traub/traub.ts` はレートの式と単一区画のパラメータを Python と共有する `neurosurrogate/physics/traub.json` から読む。
- 実験は repo の根の `experiments/` に、宣言の JSON と判定の TS を分類ごとのディレクトリで並べる（presets.json と同じ宣言の層なので `evaluation/` に入れない）。
- TS のテストは `bun test evaluation experiments`（依存は `bun install`）。

## Coding Standards

- 抽象は浅く、上から順に読める関数で書く。変種（群分け、区画）は設定か表に足す（`DESIGN.md` の判断基準）
- `_` 始まりの名前は「そのファイルの外から参照しない」印（`tests/test_conventions.py` が見る）
- 大きな改装のあとは下の Commands のテストを通す。テストは自由に足してよいが 20s 以下に抑える

## 計算する機械

学習・評価と runs は、ssh で入れる NVIDIA の GPU の機械（名前は環境変数 `SINDY_HOST`）に置く。回線に出すのはコードと指示と小さい JSON だけで、波形は動かさない。

- 学習は GPU（spec の `device: gpu`。GPU を使うのは AE の学習と GPU の実測（spec の `kind: bench`、置き場は計算する機械の runs の bench の下）だけで、1 細胞のシミュレーションは CPU）、評価とテストの学習は CPU。同じ spec でも結果がずれるので、`device` は spec に入って run を分ける。
- run の meta.json の `env` に、回した backend・device・版・host・コード（commit と書きかけの差分のハッシュ）・イメージを書く。
- 結果を読む画面は API（`evaluation/serve/serve.ts`、POST /<口>、:8787）だけを呼ぶ。API は間引いた線・軌跡・分布・指標を返し、波形を回線に出さない。
- 機械の名前と接続先は repo に書かず、環境変数で渡す。

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
```

整形と lint は `uv run ruff format` と `uv run ruff check`、型は `uv run mypy`。

## Architecture

- 依存の向き: `physics ← surrogate ← run`。physics は surrogate を import しない。表は `tests/test_conventions.py` の `_ALLOWED`。
- `neurosurrogate/` は記録・設定の基盤（MLflow や Hydra）を知らない。run の保存形式を知るのは `run.py` だけ。
- Python は数値計算（シミュレータ、学習、eFEL のスパイクの特徴量）だけを持ち、元のモデルと比べる指標は TS の `evaluation/read/compare.ts`、段取り（何を回すか、掃引、ハッシュ、重複の回避、run の一覧）は TS が持つ。Python では図を描かない。run の形式は `DESIGN.md`。
- `tests/test_conventions.py` が落ちたら、テストを緩めずコードを直す。
- 設定の形、どちらへ倒すかの判断基準は `DESIGN.md` が持つ。設計に手を入れる前に読む。
- ファイルの持つものは各ファイルの先頭の docstring に書く。`neurosurrogate/FILES.md` は生成物なので手で書き換えない（`tests/test_design.py` が古いと落ちる）。
