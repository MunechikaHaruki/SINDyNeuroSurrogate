# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（ドメイン層） | `neurosurrogate/` |
| 計算の入口（spec.json → run のディレクトリ。`python -m neurosurrogate.run`） | `neurosurrogate/run.py` |
| 何を回すかの宣言と spec.json の形（`conf.ts`）、段取り（`orchestrate.ts`。本人は操作盤の冊 `53.05-runs` のボタン、エージェントは CLI で、同じ関数を呼ぶ）、操作盤の口（`api.ts`） | `evaluation/orchestrate/` |
| 実験の結果（1 run = spec 1 つ、ディレクトリ名は spec のハッシュ）と run の一覧 | `runs/`（Git の外） |
| 評価の冊（回して元の系と比べるもの、学習 run を読むもの） | `evaluation/` の直下。図が runs/ を読む口は `evaluation/read/`、Traub の TS シミュレータは `evaluation/traub/`。context-router の `resources/research/` に symlink で入り、研究の文書（51 構成、52 理論、54 これから）と一つの面に見える |
| 用語、ディレクトリとファイルの責務、設計の判断基準、広げるときの前提 | `DESIGN.md` |

## 持たないもの

- 汎用の方法の説明や教材（HH、縮約法そのもの、統計など）。jdex の textbook が持ち、図からは前提としてリンクで引く。教材の側からは図を指さない。
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**図と Python はコードを import し合わない。** 図が読むのはデータだけ（設定 JSON と `runs/` の run）。
シミュレータは TS の図にも二重にあってよく、ずれは共通の基準値のテストで見張る。

## 図

- 書き方（部品、宣言、見た目の規約）は教材と同じで、スキル book に従う。画面と部品はルートの `.web/` が持つ。
- この repo が持つ冊は評価だけで、`evaluation/53.MM-名前.svelte`。番号は教材と同じ Johnny.Decimal の体系から取る（区域はルートの `AGENTS.md`）。研究の文書は context-router の `resources/research/50-59-research/` が持ち、コードと一緒に変わらないものはこの repo に置かない。
  冊の直下には冊だけを置き、`.ts` は役割ごとのディレクトリ（`orchestrate/` 何を回すかと段取り、`read/` runs/ を読む口、`traub/` TS シミュレータ）に番号なしの名前で置く。テストは隣の `<名前>.test.ts`、参照値は `<名前>.reference.json`。
- 常駐の開発サーバー（`http://127.0.0.1:4321/`）で教材と一緒に開く。テスト・ESLint・型は context-router の `.web/` で `bun run web-check`。

## Coding Standards

- 一時変数は同じ値を何度も使うときだけ許す（NG: `x = obj.attr; f(x)` / OK: `f(obj.attr)`）
- `__init__.py` に `__all__` を定義しない。過剰な複雑さになる
- `_` 始まりの名前とモジュール名は「外から参照しない」印。外から使うものに付けない
- 大きな改装のあとは `just test` が通ることを確認する。テストは自由に足してよいが 20s 以下に抑える

## Commands

```bash
uv sync                        # 依存導入
# 学習と評価の CLI はエージェント用。本人は操作盤の冊 (evaluation/53.05-runs) から回すので、本人に CLI を案内しない
bun evaluation/orchestrate/orchestrate.ts train <preset>...               # 学習。preset は evaluation/orchestrate/conf.ts の PRESETS (掃引は preset が run の列で持つ)
bun evaluation/orchestrate/orchestrate.ts eval <系列> <preset かハッシュ>...  # 評価。原系と置換系を runs/ に置く。preset の未学習分は学習もする
bun evaluation/orchestrate/orchestrate.ts list                            # run の一覧。引数なしで preset と系列の一覧、index で index.json を書き直す
just test                      # pytest + 段取りから学習 1 本 (使い捨ての置き場)
just format && just lint       # ruff / ruff + mypy
just clean-cache
```

## Architecture

- 依存の向き: `core ← neurons ← sim.{_current_catalog,spec,result} ← surrogate ← sim.run`
- `neurosurrogate/` はドメイン層で、MLflow や Hydra のような記録・設定の基盤を知らない。run の保存形式を知るのは最上位の `run.py` だけ。
- Python は数値計算（シミュレータ、学習、指標）だけを持ち、段取り（何を回すか、掃引、ハッシュ、重複の回避、run の一覧）は TS が持つ。Python では図を描かない。run の形式は `DESIGN.md`。
- 動的に呼ばれる入口（段取りが起動する `neurosurrogate/run.py` の main / `neurons/{hh,traub}.py`）はテスト側の免除リストに明記する。
- `tests/test_conventions.py` が落ちたら、テストを緩めずコードを直す。
- 各ディレクトリとファイルの責務、設定ファイルの規約、どちらへ倒すかの判断基準は `DESIGN.md` が持つ。設計に手を入れる前に読む。
