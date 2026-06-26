# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Standards [WRITTEN BY HUMAN - DO NOT OVERWRITE]
- HH型マルチコンパートメントニューロンの一部ノードをSINDyで抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード
- 一時変数は同じ値を何度も使うような場合にのみ許可
  - NG: `x = obj.attr; f(x)`
  - OK: `f(obj.attr)`
- 実装が終わったら、uv run scripts/main.pyでエラーが出ないことを確認
- 

[以下のセクションは永続メモリとして上書きしても構いません。
ただし、30行以内程度に納め、本当に必要な情報だけを圧縮して書くこと]

## Commands

```bash
uv run scripts/main.py                                        # 実行（変更後は必ず確認）
uv run scripts/main.py sindy=hh_relaxation                    # Hydraプリセット切替
uv run scripts/main.py --multirun sindy.optimizer.alpha=0.01,0.1  # スイープ
just format && just lint   # ruff fix+format / ruff+mypy
just mlflow                # MLflow UI（port 5100、バックエンド: mlflow.db）
just marimo                # インタラクティブノートブック（port 2700）
just marimo-mcp            # Claude Code MCP連携モード（port 2701）
just clean-cache           # キャッシュ/ビルド成果物を削除
just clean-log             # hydra-outputs/mlrunsを削除
```

mypy は `strict` モードだが `scripts/` は除外。

## Architecture

**Data flow**: Hydra conf → `DatasetConfig`/`NeuronGraph` → `unified_simulator()` (JAX Euler+lax.scan) → `SINDyNeuroSurrogate.fit()` (PCA→SINDy→exec動的コンパイル) → `SINDyAnalyzer` → MLflow

**neurosurrogate/model/**: `registry_compartments.py`(HH/passive動力学+`COMPARTMENT_TEMPLATES`) / `registry_neuron.py`(事前定義済み`MCMODELS`) / `model_dataset.py`(`Compartment`,`NeuronGraph`,`DatasetConfig`,`CurrentConfig`) / `model_neurosindy.py`(`SINDyNeuroSurrogate`) / `model_preprocessor.py`(`AutoEncoderPreprocessor`、PCA互換JAX-AE)

**neurosurrogate/calc_engine.py**: `calc_universal_deriv`がhh/passive/surrogateの3種を`vmap`で一括計算、`generic_euler_solver`が`lax.scan`でタイムループ。`IndiceArgs`でノードを3種に分離する設計が重要。

**neurosurrogate/builder/**: `registry_feature_libraries.py`(`LIB_BUILDER_REGISTRY`、type="gate"/"volt"/"identity"/"const") / `builder_feature_libraries.py`(`FeatureLibrary`) / `registry_current.py`(電流波形生成、`FUNC_MAP`) / `build_coords.py`(xarray座標構築)

**neurosurrogate/profiler/**: `profiler_model.py`(`OpCost`代数、`SINDyResult`、`SINDyAnalyzer`) / `profiler_wave.py`(eFELスパイクメトリクス+RMSE/MAE) / `profiler_view.py`(`draw_engine`/`TraceSpec`/`PanelSpec`、波形描画エンジン) / `registry_view.py`(`DRAW_MAP`、可視化スペックレジストリ)

**scripts/**: `main.py`(Hydraエントリポイント) / `io_handler.py`(MLflow I/O。`RunInfo`でrun情報読み込み、`SINDySurrogateMLflowModel`としてサロゲートをpyfunc形式で保存/復元) / `analysis.py`(marimo向け評価ロジック+UI定義。`calc_eval`がサロゲート置換後シミュレーション結果を返す) / `analysis_sweep.py`(振幅スイープ評価+plot) / `marimo.py`(marimoノートブック本体) / `conf/`(Hydra設定: `config.yaml` / `sindy/{base,hh_relaxation,hh_informed}.yaml`)

**Key notes**:
- `SINDyNeuroSurrogate._build_source`: PySINDy特徴量名からJAX関数ソースを生成し`exec`でコンパイル（`^`→`**`変換あり）
- `OpCost`: `+`/`*`をオーバーロードしてコンパートメント全体のコストを代数的に集計（`registry_compartments.py`参照）
- `NeuronGraph.chain(["passive","hh","passive"], weights)`: ノード名を自動生成（"p0","h0","p1"）して辺を接続
- `NeuronGraph.with_surrogates(targets, make_surr)`: 指定ノードをサロゲートコンパートメントに置換した新グラフを返す
