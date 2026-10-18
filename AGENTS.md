# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（ドメイン層） | `neurosurrogate/` |
| 学習（Hydra）と評価（`just eval`）の入口、MLflow の読み書き | `scripts/` |
| 実験の結果 | MLflow (`just mlflow`) |
| 結果を解釈する図（ブラウザで動かす `.svelte`）と、図が使う計算（`.ts`）とテスト | `research/50-59-research/`。コードの構成（`51-architecture/`）、モデルを式で書きそこから計算できる図（`52-theory/`）、回して元の系と比べるものと学習 run を読むもの（`53-evaluation/`） |
| 用語、ディレクトリとファイルの責務、設計の判断基準、広げるときの前提 | `DESIGN.md` |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）。jdex の textbook が持ち、図からは前提としてリンクで引く。教材の側からは図を指さない。
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**図と Python はコードを import し合わない。** 図が読むのはデータだけ（設定 JSON と MLflow の run）。
シミュレータは TS の図にも二重にあってよく、ずれは共通の基準値のテストで見張る。

## 図

- 書き方（部品、宣言、見た目の規約）は教材と同じで、スキル book に従う。画面と部品はルートの `.web/` が持つ。
- 図の番号は教材と同じ Johnny.Decimal の体系から取る（区域はルートの `AGENTS.md`）。図は `research/50-59-research/<NN-分類>/<NN.MM-名前>.svelte`。
  計算はテストで数値を固定できるよう同じ分類に番号なしの役割の名前（`traub.ts`、`mlflow.ts`）で切り出し、テストは `<名前>.test.ts`、参照値は `<名前>.reference.json`。
- 常駐の開発サーバー（`http://127.0.0.1:4321/`）で教材と一緒に開く。テスト・ESLint・型は context-router の `.web/` で `bun run web-check`。

## Coding Standards

- 一時変数は同じ値を何度も使うときだけ許す（NG: `x = obj.attr; f(x)` / OK: `f(obj.attr)`）
- `__init__.py` に `__all__` を定義しない。過剰な複雑さになる
- `_` 始まりの名前とモジュール名は「外から参照しない」印。外から使うものに付けない
- 大きな改装のあとは `just test` が通ることを確認する。テストは自由に足してよいが 20s 以下に抑える

## Commands

```bash
uv sync                                      # 依存導入
uv run scripts/main.py                       # fit + MLflow log のみ (kernel は回さない)
uv run scripts/main.py surrogate=_hh_informed  # Hydra プリセット切替
uv run scripts/main.py --multirun            # preset の hydra.sweeper.params 直積 sweep
just test                  # pytest + main.py
just format && just lint   # ruff / ruff + mypy (scripts/ も含む)
just mlflow                # MLflow server (port 5100)。Mac ではログイン時に常駐する
just eval <系列> <学習 run>...  # 評価して波形 run を MLflow に置く (系列の一覧は just eval -h)
just traub                 # traub_* preset を順に --multirun
just clean-cache / clean-log
just clean-run / clean-test # MLflow run 全削除 / smoke_test experiment のみ削除
```

## Architecture

- 依存の向き: `core ← neurons ← sim.{_current_catalog,spec,result} ← surrogate ← sim.run`
- `neurosurrogate/` はドメイン層で、MLflow/Hydra を import しない。
- 評価結果は波形 run に、学習 run の中身は学習 run の `description/` に置き、Python では図を描かない。形式は `DESIGN.md`。
- 動的に呼ばれる入口（Hydra entry / `just eval` の main / `neurons/{hh,traub}.py`）はテスト側の免除リストに明記する。
- `tests/test_conventions.py` が落ちたら、テストを緩めずコードを直す。
- 各ディレクトリとファイルの責務、設定ファイルの規約、どちらへ倒すかの判断基準は `DESIGN.md` が持つ。設計に手を入れる前に読む。
