# CLAUDE.md

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード。

## コマンド

パッケージ管理は `uv`、タスクは `just`。

```bash
just format          # ruff check --fix + ruff format
just lint            # ruff + mypy + pylint
just mlflow          # MLflow UI (port 5100, backend: ./mlruns)
just marimo          # marimo edit equations.py
just clean-cache     # __pycache__ / .mypy_cache 等
just clean-log       # hydra-outputs / hydra-multiruns / mlruns

uv run python scripts/main.py                            # single run → static eval
uv run python scripts/main.py -m                         # multirun → Optuna reaction eval
uv run python scripts/main.py sindy.optimizer.alpha=0.1  # Hydra override
```

Marimo: `analysis.py` (MLflow run 閲覧), `analysis_current.py` (電流波形), `equations.py` (LaTeX → PNG)。

## 実行モード分岐

`scripts/main.py` は `HydraConfig.mode.name == "MULTIRUN"` で分岐:

| mode     | 評価関数                     | MLflow experiment         | 用途                                |
| -------- | ---------------------------- | ------------------------- | ----------------------------------- |
| single   | `eval_with_static_datasets`  | `test_static_params`      | catalog スイープ全ケースを詳細評価   |
| multirun | `eval_with_model_reaction`   | `test_dynamic_datasets`   | 定常電流で発火閾値検出 → Optuna     |

multirun は `eval_with_model_reaction` が返す値が Optuna 目的関数。現状 `orig_threshold = 6.5` ハードコード (scripts/eval.py)。

## アーキテクチャ

```
neurosurrogate/          # コアライブラリ（scripts 非依存）
  modeling/
    neuron_core.py       # HH パラメータ / ゲート関数 @njit / FUNC_COST_MAP
    calc_engine.py       # unified_simulator, generic_euler_solver @njit
    xarray_utils.py      # build_indices, set_coords (MultiIndex)
    profiler.py          # コスト静的解析, 波形/スパイク指標
    __init__.py          # SINDySurrogateWrapper, PCAPreProcessorWrapper
  utils/
    current_generators.py  # apply(arr) -> None を返す高階関数
    plots.py               # draw_engine (matplotlib/plotly), spec_*

scripts/                 # Hydra + MLflow オーケストレーション
  main.py                # 学習 (親 run) → 評価 (子 run)
  eval.py                # 評価フロー 2 種
  conf/
    config.yaml
    current_generators/base.yaml
    feature_library_components.py  # LIB_BUILDER_REGISTRY
    neuron_models.py               # MODEL_DEFINITIONS
    style/*.mplstyle
  utils/
    boot.py              # setup_all: proxy無効化 + MLflow + matplotlib
    builder_core.py      # build_simulator_config, build_surrogate
    builder.py  # catalog → sweep cases 展開
    log_model.py         # pyfunc で SINDy モデル保存/復元
    log_utils.py         # メトリクス / 図 / xarray ロギング
```

## 不変条件・規約

### xarray レイアウト

`unified_simulator` 出力の `vars` は `(time, features)`、`features` は MultiIndex `(comp_id, variable, gate)`。

- 電位: `ds["vars"].sel(gate=False, comp_id=i)`
- ゲート / 潜在: `ds["vars"].sel(gate=True, comp_id=i)`
- 外部電流: `ds["I_ext"]` — `(time,)`
- 網内電流: `ds["I_internal"]` — `(time, node_id)`

### SoA シミュレータ

`calc_universal_deriv` は if 分岐なし。ノード種別ごとに `indice["ids"]["hh"|"passive"|"surr"]` で独立ループ。新しいコンパートメント種を足す場合は `COMPARTMENT_TEMPLATES`, `build_indices`, `calc_universal_deriv` の 3 箇所を同期更新。

サロゲート未使用時は `DummySurrogate` が `sindy_args` を埋める。`surr_ids` が空なら実コストゼロ。

### 動的 Numba 関数生成

`SINDySurrogateWrapper._build_source` が pysindy の `feature_names` から Python ソース文字列を組み立てて `exec` で `neuron_core` 名前空間にコンパイル。置換ルール:

- `1` → `1.0` (Numba 型推論のため)
- `^` → `**`

生成関数 `dynamic_compute_theta` に `@njit` が適用される前提で `neuron_core` の名前空間を使っているため、`neuron_core` を改名・移動する場合は `SINDySurrogateWrapper.fit` の `target_module` 引数経路と `log_model.SINDySurrogateMLflowModel.load_context` の両方を更新。

### 演算コスト map の整合性

`profiler.build_feature_cost_map` は SINDy feature 名を regex でパースし `FUNC_COST_MAP` を参照。**未知の基底関数に対して fail-fast する** (`static_calc_cost` 経由)。

新しい `@njit` 関数を SINDy ライブラリに追加したら:
1. `neuron_core.FUNC_COST_MAP` にコスト `{exp, div, pm, mul}` 追加
2. `feature_library_components.FUNC_REGISTRY` に登録

### Hydra catalog → sweep 展開

`config.yaml` の `datasets_settings.catalog.<name>.sweep` サブキーが `build_sweep_cases` で展開される:

- `sweep.current_seed: [v1, v2, ...]` — seed 値で複製
- `sweep.current.variable.<key>: {start, stop, step|num}` — pipeline パラメータの線形スイープ (`pipeline_ind` で対象インデックス指定)

`teaching_catalog` は学習用、`sweep_catalog` は評価用の catalog 名を指定。

### MLflow run 構造

- 親: `Training_run:{hydra_overrides}` — 学習済みモデル、要約指標
- 子: `Eval_{case_name}` — `tags.mlflow.parentRunId` で親と紐付け、`tags.eval_dataset` でケース識別

サロゲートは `mlflow.pyfunc` でログ。`load_surrogate_model(run_id)` で復元 → `unified_simulator` 専用（`predict` は未実装）。

### 新規追加パターン

| 追加対象             | 変更箇所                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| 電流生成関数         | `neurosurrogate/utils/current_generators.py` + `conf/current_generators/base.yaml` |
| ニューロン構造       | `scripts/conf/neuron_models.py::MODEL_DEFINITIONS`                                 |
| SINDy ライブラリ型   | `scripts/conf/feature_library_components.py::LIB_BUILDER_REGISTRY`                 |
| 基底関数             | `neuron_core.py` (@njit) + `FUNC_COST_MAP` + `FUNC_REGISTRY`                       |
| コンパートメント種   | `COMPARTMENT_TEMPLATES` + `build_indices` + `calc_universal_deriv`                 |

### 環境・スタイル

- Python >= 3.11、mypy strict（`scripts/` 除外）、ruff、pylint は `neurosurrogate` と `scripts` のみ
- `boot.setup_proxy()` で `HTTP_PROXY`/`HTTPS_PROXY` を空に上書き（社内プロキシ回避）
- matplotlib は `Agg` バックエンド固定、スタイルは `base.mplstyle` + `{matplotlib_style}.mplstyle`
- Numba 初回呼び出しで遅延コンパイル、初回実行は遅い
- 積分器は固定ステップ Euler（`generic_euler_solver`）