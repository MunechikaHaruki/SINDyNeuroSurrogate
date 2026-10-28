# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（`neurosurrogate/physics/` 元のモデルを解く、`neurosurrogate/surrogate/` 学習して置き換える、直下にスパイクの特徴量 `spikes.py` と入口 `run.py`） | `neurosurrogate/` |
| 計算の入口（spec.json → run のディレクトリ。`python -m neurosurrogate.run`） | `neurosurrogate/run.py` |
| 何を回すかの宣言（`conf.json`。縮約の仕方・テスト専用の学習・系列。`conf.ts` が読み、学習 run を `<仕方>_<群分け>` の名前から組む）と spec.json の形（`spec.ts`）、段取り（`orchestrate.ts`。本人は操作盤の冊のボタン、エージェントは CLI で、同じ関数を呼ぶ） | `evaluation/orchestrate/` |
| 実験の結果（学習したモデルは `models/<ハッシュ>/`、入力ごとの結果は `inputs/<入力のハッシュ>/` の下に元のモデルと置き換えたモデル）と run の一覧 | `runs/`（Git の外） |
| 結果を読む TS（元のモデルと比べる指標 `compare.ts`、学習の書き出しの読み方 `model-json.ts`、Traub の TS シミュレータ） | `evaluation/read/`、`evaluation/traub/` |
| 用語、設計の判断基準、広げるときの前提 | `DESIGN.md` |
| ファイルごとの持つものと依存の図（生成物。正本は各ファイルの先頭の docstring。`uv run python tests/test_design.py` で作り直す） | `neurosurrogate/FILES.md` |
| 研究コードの手入れのやること（数値も主張も変えないものと、原著とずれている箇所。jdex の 53.05） | `todo.md` |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）。context-router の jdex が持つ。
- 結果を読む冊（.svelte）と画面の口。context-router の `projects/research-master/` が持つ。
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**TS と Python はコードを import し合わない。** TS が読むのはデータだけ（設定 JSON と `runs/` の run）。
シミュレータは TS にも二重にあってよく、ずれは共通の基準値のテストで見張る。

## TS

- この repo は冊（.svelte）を持たず、context-router の部品にも依存しない。単独で動く研究コードだけを置く。結果を読む冊（評価の 53、操作盤、計測の 59.02 など）は context-router の `projects/research-master/` が持ち、この repo の TS を import して描く。
- `evaluation/` には TS だけを、役割ごとのディレクトリ（`evaluation/orchestrate/` 何を回すかと段取り、`evaluation/read/` 比べる指標と学習の書き出しの読み方、`evaluation/traub/` TS シミュレータ）に番号なしの名前で置く。テストは隣の `<名前>.test.ts`、参照値は `<名前>.reference.json`。`evaluation/traub/traub.ts` はレートの式を Python と共有する `neurosurrogate/physics/traub.json` から読む。
- TS のテスト・ESLint・型は、今は context-router で `bun run --cwd areas/infra/ontology verify`。

## Coding Standards

- 抽象は浅く、上から順に読める関数で書く。変種（群分け、区画）は設定か表に足す（`DESIGN.md` の判断基準）
- `_` 始まりの名前は「そのファイルの外から参照しない」印（`tests/test_conventions.py` が見る）
- 大きな改装のあとは下の Commands のテストを通す。テストは自由に足してよいが 20s 以下に抑える

## Commands

```bash
uv sync                        # 依存導入
# 学習と評価の CLI はエージェント用。本人は操作盤の冊 (jdex の 53.01) から回すので、本人に CLI を案内しない
bun evaluation/orchestrate/orchestrate.ts train <学習 run>...             # 学習。学習 run は <仕方>_<群分け> (kam_hn、pca_hn_ac) か tests の名前
bun evaluation/orchestrate/orchestrate.ts eval <系列> <学習 run かハッシュ>...  # 評価。入力を元のモデルと置き換えたモデルで回す。未学習なら学習もする
bun evaluation/orchestrate/orchestrate.ts list                            # run の一覧。引数なしで仕方・テスト・系列の一覧、index で index.json を書き直す
uv run pytest -q               # 規約、区画の式、サロゲート、neurosurrogate/run.py
```

ruff の整形と lint は編集のたび、mypy はターンの終わりに、context-router の hook が走らせる。行数・複雑度・保守性指数は context-router の冊 59.02（`projects/research-master/frontend/meta/code-metrics.svelte`）で見る。

## Architecture

- 依存の向き: `physics ← surrogate ← run`。physics は surrogate を import しない。表は `tests/test_conventions.py` の `_ALLOWED`。
- `neurosurrogate/` は記録・設定の基盤（MLflow や Hydra）を知らない。run の保存形式を知るのは `run.py` だけ。
- Python は数値計算（シミュレータ、学習、eFEL のスパイクの特徴量）だけを持ち、元のモデルと比べる指標は TS の `evaluation/read/compare.ts`、段取り（何を回すか、掃引、ハッシュ、重複の回避、run の一覧）は TS が持つ。Python では図を描かない。run の形式は `DESIGN.md`。
- `tests/test_conventions.py` が落ちたら、テストを緩めずコードを直す。
- 設定の形、どちらへ倒すかの判断基準は `DESIGN.md` が持つ。設計に手を入れる前に読む。
- ファイルの持つものは各ファイルの先頭の docstring に書く。`neurosurrogate/FILES.md` は生成物なので手で書き換えない（`tests/test_design.py` が古いと落ちる）。
