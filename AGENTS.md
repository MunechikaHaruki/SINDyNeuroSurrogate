# SINDyNeuroSurrogate

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、
演算コスト削減と波形再現性を評価する研究コード。

## 持つもの

| 中身 | 場所 |
| --- | --- |
| 研究コード（ドメイン層） | `neurosurrogate/` |
| 学習（Hydra）と評価（`just eval`）の入口、MLflow の読み書き | `scripts/` |
| 実験の結果 | MLflow (`just mlflow`)。図は親の live-textbook の `research/sindy-surrogate-evaluation` が描く |
| この領域の用語 | `CONTEXT.md` |
| 設計の詳細と判断基準 | `docs/architecture.md`、`docs/agents/`、`docs/adr/` |

## 持たないもの

- 概念の説明や教材。この repo からは参照しない。
- 結果を解釈する図。親の live-textbook の `research/` の冊が持つ。
- ポスター、スライド、論文の原稿
- 修論の計画、期日、週次の進捗、受けた指摘
- 本人の事実と、進路の判断基準

**冊とはコードを import し合わない。** 冊の図が読むのはデータだけ（設定 JSON と MLflow の run）。
シミュレータは TS の冊にも二重にあってよく、ずれは共通の基準値のテストで見張る。

## Coding Standards

- 一時変数は同じ値を何度も使うときだけ許す（NG: `x = obj.attr; f(x)` / OK: `f(obj.attr)`）
- `__init__.py` に `__all__` を定義しない。過剰な複雑さになる
- `_` 始まりの名前とモジュール名は「外から参照しない」印。外から使うものに付けない
- 大きな改装のあとは `just test` が通ることを確認する。テストは自由に足してよいが 20s 以下に抑える
- 編集のたびにルートの hook が ruff で整形と検査をし、ターンの終わりに mypy を走らせる。返ってきた誤りはその場で直す

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

- 依存の向き: `core ← neurons ← sim.{_current_catalog,spec,result} ← surrogate ← sim.{run,artifacts} ← artifact.bundle`
- `neurosurrogate/` はドメイン層で、MLflow/Hydra を import しない。
- 評価結果は MLflow の波形 run に置き、Python では図を描かない（学習 run が自分について描く図だけは学習のときに `figures/` へ）。形式は `docs/architecture.md`。
- 依存の向き、ドメイン層の import、`__all__` の不在、`_` の綴りは `tests/test_conventions.py` が機械検査する。
- 動的に呼ばれる入口（Hydra entry / `just eval` の main / `neurons/{hh,traub}.py`）はテスト側の免除リストに明記する。
- 検査が落ちたら、テストを緩めずコードを直す。
- 各ディレクトリとファイルの責務、設定ファイルの規約は `docs/architecture.md` が持つ。
- 設計に手を入れる前に `docs/agents/design-principles.md`（どちらへ倒すかの判断基準）を読む。

## Agent skills

- Issue tracker: GitHub Issues (`gh` CLI)。`docs/agents/issue-tracker.md`
- Triage labels: 5 ラベル (`needs-triage`/`needs-info`/`ready-for-agent`/`ready-for-human`/`wontfix`)。`docs/agents/triage-labels.md`
- Domain docs: single-context (`CONTEXT.md` + `docs/adr/`)。`docs/agents/domain.md`

## RTK

`rtk` 前置は出力を縮めるだけで、終了コードは素通しする。出力が変なときだけ `rtk proxy <cmd>` と比べる。
