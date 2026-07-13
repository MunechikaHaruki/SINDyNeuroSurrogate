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
uv run scripts/main.py                                            # 実行（変更後は必ず確認）
uv run scripts/main.py sindy=hh_relaxation                        # Hydraプリセット切替 (base/base_traub/hh_informed/hh_relaxation/hybrid/hybrid_n2)
uv run scripts/main.py --multirun sindy.fit.optimizer.alpha=0.01,0.1  # スイープ
just format && just lint   # ruff fix+format / ruff+mypy (strict、scripts/ 除外)
just mlflow                # MLflow UI (port 5100、backend: mlflow.db)
just marimo                # marimo notebook (port 2700)
just marimo-mcp            # Claude Code MCP連携 (port 2701)
just clean-cache / clean-log
```

## Architecture

**Data flow**: Hydra conf → `DatasetConfig`/`NeuronGraph` → `unified_simulator()` (JAX Euler + `lax.scan`) → `NeuroSurrogateBase.fit()` (前処理 → PySINDy → `FeatureLibrary` から `compute_theta` 直接構築) → MLflow log。`neurosurrogate/__init__.py` で `jax_enable_x64` 強制 ON (HH ゲート exp 数値安定化に必須)。

**neurosurrogate/core/**: `network.py`(`Compartment`/`CompartmentType`/`NeuronGraph`/`DatasetConfig`/`CurrentConfig`。`chain(...)` ノード自動命名、`with_surrogates(targets, make_surr)` でサロゲート置換グラフ生成) / `simulator.py`(`unified_simulator` + `calc_universal_deriv` が type 名 dispatch で `vmap` 一括計算、`generic_euler_solver` が `lax.scan`) / `coords.py`(xarray 座標) / `opcost.py`(`OpCost` 代数、`+`/`*` で演算コスト集計)

**neurosurrogate/{compartments,models,currents}**: `compartments/{hh,traub,common}.py` 動力学 + `COMPARTMENT_TEMPLATES` (`hh.py` に `HH_RATE_ENTRIES`/`HH_GATE_PAIRS`/`HH_GATE_FORWARD`) / `models/` に `MCMODELS` + `traub19.py`(19-comp、traub.c 代数的等価) / `currents.py` に電流波形 + `CURRENT_MAP`

**neurosurrogate/surrogate/**: `base.py`(`NeuroSurrogateBase`、`_dataset`/`train_comp_id`/`_train_xr` 保持、`save`/`load` は `surrogate.joblib` 単一ファイル) / `sindy.py`(`SINDyNeuroSurrogate`) / `hybrid.py`(`HybridSINDyNeuroSurrogate`、AE preprocessor 利用可、n_components 3 が既定) / `bundle.py`(`PREPROCESSOR_CLS`={"pca","ae"} / `OPTIMIZER_CLS` / `SindyBundle`/`PreprocessorBundle`、`bundle.decode()` 統一) / `libraries.py`(`FeatureLibrary`/`SubLibrary`/`LibraryEntry`、`FIXED_LIB_ENTRIES`={hh_gate,hh_gate_product,hh_gate_forward,hh_gate_forward_product,hh_relaxation_driver,hh_relaxation_decay,volt,gate_poly_volt} + `VARIADIC_LIB_ENTRIES`={basis}、`FeatureLibrary.build(specs)`) / `autoencoder.py`(`AutoEncoderPreprocessor`、PCA 互換 JAX-AE) / `__init__.py` に `SURR_CLS` 辞書 (`sindy`/`hybrid`)

**neurosurrogate/{metrics,view}**: `metrics/wave.py`(eFEL スパイク + RMSE/MAE) / `view/{engine,specs,plots}.py`(`draw_engine`/`TraceSpec`/`PanelSpec`/`DRAW_MAP`)

**scripts/**: `main.py`(Hydra エントリ) / `mlflow_io.py`(MLflow I/O。`log_surrogate_model` は `surrogate.save` + `meta.yaml`(surrogate_type/dataset/train_comp_id) を artifact log、`load_surrogate_model` は `(NeuroSurrogateBase, SurrogateMeta)` tuple 返却) / `analysis/{single,sweep,ui}.py`(marimo 評価、`calc_eval` がサロゲート置換後シミュ結果返却) / `marimo.py` / `conf/`(`config.yaml` に共通 `sindy.fit.optimizer` 集約 + `sindy/*.yaml` プリセット)
