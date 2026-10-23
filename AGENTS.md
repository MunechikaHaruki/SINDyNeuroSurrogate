# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（`physics/` 元のモデルを解く、`surrogate/` 学習して置き換える、直下にスパイクの特徴量 `spikes.py` と入口 `run.py`） | `neurosurrogate/` |
| 計算の入口（spec.json → run のディレクトリ。`python -m neurosurrogate.run`） | `neurosurrogate/run.py` |
| 何を回すかの宣言（`conf.ts`。preset・系列・実験）と spec.json の形（`spec.ts`）、段取り（`orchestrate.ts`。本人は操作盤の冊 `runs` のボタン、エージェントは CLI で、同じ関数を呼ぶ）、操作盤の口（`api.ts`） | `evaluation/orchestrate/` |
| 実験の結果（学習したモデルは `models/<ハッシュ>/`、入力ごとの結果は `inputs/<入力のハッシュ>/` の下に元のモデルと置き換えたモデル）と run の一覧 | `runs/`（Git の外） |
| 評価の冊（回して元の系と比べるもの、学習 run を読むもの） | `evaluation/` の直下。図が runs/ を読む口は `evaluation/read/`、Traub の TS シミュレータは `evaluation/traub/`。context-router の `resources/research/` に symlink で入り、研究の文書（51 構成、52 理論、54 結果、55 これから、56 修論、57 監査）と一つの区域に見える |
| 用語、ファイルの責務、設計の判断基準、広げるときの前提 | `DESIGN.md` |
| 研究コードの手入れのやること（数値も主張も変えないもの。jdex の 55.09） | `todo.md` |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）。jdex の textbook が持ち、図からは前提としてリンクで引く。教材の側からは図を指さない。
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**図と Python はコードを import し合わない。** 図が読むのはデータだけ（設定 JSON と `runs/` の run）。
シミュレータは TS の図にも二重にあってよく、ずれは共通の基準値のテストで見張る。

## 図

- 書き方（部品、宣言、見た目の規約）は教材と同じで、スキル book に従う。画面と部品は context-router の `resources/.web/` が持つ。
- この repo が持つ冊は評価だけで、`evaluation/名前.svelte`。context-router の `resources/research/evaluation` として入り、番号（53.MM）は教材と同じ Johnny.Decimal の体系で、冊の番号も含めて context-router の `resources/index.json` と `resources/research/index.json` が持つ。冊を足したらそこに番号を足す。研究の文書は context-router の `resources/research/` が持ち、コードと一緒に変わらないものはこの repo に置かない。
  冊の直下には冊だけを置き、`.ts` は役割ごとのディレクトリ（`orchestrate/` 何を回すかと段取り、`read/` runs/ を読む口、`traub/` TS シミュレータ）に番号なしの名前で置く。テストは隣の `<名前>.test.ts`、参照値は `<名前>.reference.json`。`traub/traub.ts` はレートの式を Python と共有する `neurosurrogate/physics/traub.json` から読む（symlink 越しの相対 import は context-router の `resources/.web/tsconfig.json` の `rootDirs` が解く）。
- 常駐の開発サーバー（`http://127.0.0.1:4321/`）で教材と一緒に開く。テスト・ESLint・型は context-router の `resources/.web/` で `bun run web-check`。

## Coding Standards

- 抽象は浅く、上から順に読める関数で書く。変種（群分け、区画）は設定か表に足す（`DESIGN.md` の判断基準）
- `_` 始まりの名前は「そのファイルの外から参照しない」印（`tests/test_conventions.py` が見る）
- 大きな改装のあとは下の Commands のテストを通す。テストは自由に足してよいが 20s 以下に抑える

## Commands

```bash
uv sync                        # 依存導入
# 学習と評価の CLI はエージェント用。本人は操作盤の冊 (evaluation/runs) から回すので、本人に CLI を案内しない
bun evaluation/orchestrate/orchestrate.ts train <preset>...               # 学習。preset は evaluation/orchestrate/conf.ts の PRESETS (掃引は d の配列)
bun evaluation/orchestrate/orchestrate.ts eval <系列> <preset かハッシュ>...  # 評価。入力を元のモデルと置き換えたモデルで回す。preset の未学習分は学習もする
bun evaluation/orchestrate/orchestrate.ts list                            # run の一覧。引数なしで preset と系列の一覧、index で index.json を書き直す
uv run pytest -q               # 規約、区画の式、サロゲート、neurosurrogate/run.py
```

ruff の整形と lint は編集のたび、mypy はターンの終わりに、context-router の hook が走らせる。行数・複雑度・保守性指数は context-router の冊 51.09（`resources/research/architecture/code-metrics`）で見る。

## Architecture

- 依存の向き: `physics ← surrogate ← run`。physics は surrogate を import しない。表は `tests/test_conventions.py` の `_ALLOWED`。
- `neurosurrogate/` は記録・設定の基盤（MLflow や Hydra）を知らない。run の保存形式を知るのは `run.py` だけ。
- Python は数値計算（シミュレータ、学習、eFEL のスパイクの特徴量）だけを持ち、元のモデルと比べる指標は図の `evaluation/read/compare.ts`、段取り（何を回すか、掃引、ハッシュ、重複の回避、run の一覧）は TS が持つ。Python では図を描かない。run の形式は `DESIGN.md`。
- `tests/test_conventions.py` が落ちたら、テストを緩めずコードを直す。
- ファイルの責務、設定の形、どちらへ倒すかの判断基準は `DESIGN.md` が持つ。設計に手を入れる前に読む。
