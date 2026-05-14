# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

HH型マルチコンパートメントニューロンの一部ノードをSINDyで抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード。

## Commands

```bash
# Run (after any change)
uv run scripts/main.py

# Format / lint
just format        # ruff fix + format
just lint          # ruff check + mypy

# MLflow UI (port 5100)
just mlflow

# Interactive notebook
just marimo
just marimo-mcp    # Claude Code MCP連携モード

# Cleanup
just clean-cache
just clean-log

# Hydra overrides
uv run scripts/main.py sindy=hh_relaxation
uv run scripts/main.py --multirun sindy.optimizer.alpha=0.01,0.1,1.0
```

mypy は `strict` モードだが `scripts/` は除外。

## Architecture

### Data flow

```
Hydra config (scripts/conf/)
    → DatasetConfig / NeuronGraph
    → unified_simulator()          # JAX Euler + lax.scan
    → SINDyNeuroSurrogate.fit()   # PySINDy学習 + compute_theta をexecでコンパイル
    → SINDyAnalyzer (OpCost計算)
    → MLflow logging
```

### neurosurrogate/model/

| ファイル | 役割 |
|---|---|
| `registry_compartments.py` | JAXで実装したHH/passiveコンパートメントの動力学関数と `COMPARTMENT_TEMPLATES` |
| `registry_neuron.py` | 事前定義済みニューロングラフ `MCMODELS`（"hh", "php", "phhpp", "hh7" など） |
| `model_dataset.py` | `Compartment`, `NeuronGraph`, `DatasetConfig`, `CurrentConfig` のデータクラス群。`NeuronGraph.graph_laplacian` がコンパートメント間電流行列を返す |
| `model_neurosindy.py` | `SINDyNeuroSurrogate`: PCAで前処理→SINDy学習→`compute_theta`をexecでJAX関数に動的コンパイル |

### neurosurrogate/calc_engine.py

シミュレーション本体。`calc_universal_deriv` がhh/passive/surrogateの3種ノードを `jax.vmap` で一括計算し、`generic_euler_solver` が `jax.lax.scan` でタイムループを走らせる。`IndiceArgs` でノードインデックスを3種に分離する設計が重要。

### neurosurrogate/builder/

- `registry_feature_libraries.py` — `LIB_BUILDER_REGISTRY`: `type` キーでライブラリビルダーを選択（"gate", "volt", "identity", "const"）
- `builder_feature_libraries.py` — `FeatureLibrary`: PySINDyの `GeneralizedLibrary` とOpCost対応表をまとめるラッパー
- `registry_current.py` — 刺激電流波形の生成関数群
- `build_coords.py` — xarrayの座標構築とゲートインデックス計算

### neurosurrogate/profiler/

- `profiler_model.py` — `OpCost`（四則演算コストの代数的集計）、`SINDyResult`（学習結果のデータクラス）、`SINDyAnalyzer`（サロゲートのOpCostと各種メトリクス計算）
- `profiler_wave.py` — eFELによるスパイクメトリクス（スパイク数差、レイテンシ誤差、AP振幅誤差、ISI統計など）とRMSE/MAE

### scripts/conf/ (Hydra)

- `config.yaml` — トップレベル設定（sindy:グループのデフォルト選択）
- `sindy/base.yaml` — デフォルト設定（PCA前処理、STLSQ最適化、ライブラリ定義、データセット設定）
- `sindy/hh_relaxation.yaml`, `sindy/hh_informed.yaml` — 実験プリセット

### Key design notes

- **動的コンパイル**: `SINDyNeuroSurrogate._build_source` がPySINDyの特徴量名からJAX関数のソースコードを生成し `exec` でコンパイルする。`^` を `**` に置換する処理がある。
- **OpCost代数**: `OpCost` は `+` と `*` 演算子をオーバーロードしており、コンパートメント全体のコストを宣言的に記述できる（`registry_compartments.py` 参照）。
- **NeuronGraph.chain**: `["passive", "hh", "passive"]` のような型リストからノード名（"p0", "h0", "p1"）と辺を自動生成する便利メソッド。
- **MLflow**: バックエンドはSQLite (`mlflow.db`)。実験結果は `io_handler.py` 経由でログ。
