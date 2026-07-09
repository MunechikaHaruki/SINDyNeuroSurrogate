# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Standards [WRITTEN BY HUMAN - DO NOT OVERWRITE]
- HH型マルチコンパートメントニューロンの一部ノードをSINDyで抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード
- 一時変数は同じ値を何度も使うような場合にのみ許可
  - NG: `x = obj.attr; f(x)`
  - OK: `f(obj.attr)`
- 実装が終わったら、uv run scripts/main.pyでエラーが出ないことを確認
- Hooksで実行されるjust lint、just formatのエラーは都度対処すること

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

**Data flow**: Hydra conf → `DatasetConfig`/`NeuronGraph` → `unified_simulator()` (JAX Euler+`lax.scan`) → `SINDyNeuroSurrogate.fit()` (前処理→PySINDy→`FeatureLibrary`から`compute_theta`直接構築) → `SINDyAnalyzer` → MLflow。`neurosurrogate/__init__.py`で `jax_enable_x64` 強制ON（HHゲートexp数値安定化のため必須）。

**neurosurrogate/core/**: `network.py`(`Compartment`/`CompartmentType`/`NeuronGraph`/`DatasetConfig`/`CurrentConfig`。`chain(...)`でノード自動命名・`with_surrogates(targets, make_surr)`でサロゲート置換グラフ生成) / `simulator.py`(`unified_simulator` + `calc_universal_deriv`がtype名dispatchで`vmap`一括計算、`generic_euler_solver`が`lax.scan`、`IndiceArgs`でtype分離) / `coords.py`(xarray座標)

**neurosurrogate/registry/**: `compartments/{hh,traub,common}.py`(動力学+`COMPARTMENT_TEMPLATES`、HH/Traub/passive分割。`hh.py`に`HH_RATE_ENTRIES`/`HH_GATE_PAIRS`/`HH_GATE_FORWARD`集約) / `traub19.py`(19-comp Traub事前定義、traub.c代数的等価) / `neuron.py`(`MCMODELS`) / `current.py`(電流波形、`FUNC_MAP`) / `feature_libraries.py`(`FIXED_LIB_ENTRIES`={hh_gate,hh_gate_product,hh_gate_forward,hh_gate_forward_product,hh_relaxation_driver,hh_relaxation_decay,volt,gate_poly_volt} + `VARIADIC_LIB_ENTRIES`={basis})

**neurosurrogate/surrogate/**: `neurosindy.py`(`SINDyNeuroSurrogate`。`_build_compute_theta`が`LibraryEntry.func`直接呼び出しでexec不使用。save/load は`surrogate.joblib`単一ファイル、load時sindy=None) / `libraries.py`(`FeatureLibrary`/`SubLibrary`/`LibraryEntry`、`FeatureLibrary.build(specs)`で構築) / `preprocessor.py`(`AutoEncoderPreprocessor`、PCA互換JAX-AE) / `analysis.py`(`SINDyAnalyzer`、`SINDyResult`)

**neurosurrogate/{metrics,view,opcost}**: `metrics/wave.py`(eFELスパイク+RMSE/MAE) / `view/{engine,specs,plots}.py`(`draw_engine`/`TraceSpec`/`PanelSpec`/`DRAW_MAP`) / `opcost.py`(`OpCost`代数、`+`/`*`で演算コスト集計)

**scripts/**: `main.py`(Hydraエントリ) / `mlflow_io.py`(MLflow I/O、`RunInfo`+`SINDySurrogateMLflowModel`でpyfunc save/load 委譲) / `analysis/{single,sweep,ui}.py`(marimo評価。`calc_eval`がサロゲート置換後シミュ結果返却) / `marimo.py` / `conf/`(`config.yaml` + `sindy/{base,base_traub,hh_relaxation,hh_informed}.yaml`)
