# CLAUDE.md

HH 型マルチコンパートメントニューロンの一部ノードを SINDy で抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード。

## コマンド

パッケージ管理は `uv`、タスクは `just`。

```bash
just format          # ruff check --fix + ruff format
just lint            # ruff + mypy + pylint
just mlflow          # MLflow UI (port 5100, backend: ./mlruns)
just marimo          # marimo edit equations.py
just clean-cache     # __pycache__ / .mypy_cache / .ruff_cache / .pytest_cache 等
just clean-log       # hydra-outputs / hydra-multiruns / mlruns
just lizard          # 複雑度計測 (neurosurrogate, scripts, data_scripts)
just radon           # cc / mi 計測

uv run python scripts/main.py                            # single run
uv run python scripts/main.py -m                         # multirun (Hydra sweeper で alpha スイープ)
uv run python scripts/main.py sindy.optimizer.alpha=0.1  # Hydra override
uv run python scripts/main.py sindy=hh_informed          # SINDy config 切り替え
```

Marimo: `scripts/marimo.py` が MLflow run を読んで SINDy 係数ヒートマップ・方程式・代理モデル評価をインタラクティブに閲覧。

## 実行モード

`scripts/main.py::cli_flow` は学習 1 サイクルのみを担当する：

1. `build_surrogate(cfg_sindy)` — preprocessor / pysindy / library を Hydra `_target_` で instantiate
2. `build_dataset(**cfg_sindy["datasets"])` で学習用 dataset config を組み立て、`build_simulator_config` 経由で `unified_simulator` に渡す
3. `surrogate.fit(train_ds, train_comp_id)` — `train_comp_identifier`（例: `"soma"`）を `name_to_idx_dict` で解決
4. `get_loggable_summary` で要約を作り `log_surrogate_summary` / `log_surrogate_model` で MLflow にログ

`is_multirun` の分岐は現在プレースホルダ（`pass`）。multirun では Hydra Optuna sweeper が `sindy.optimizer.alpha` を振るが、目的関数を返す評価フローはまだ繋がっていない。評価は `scripts/analysis.py`（marimo 経由）で別途インタラクティブに行う運用。

MLflow experiment は `is_multirun` で切替：
- single → `test_static_params`
- multirun → `test_dynamic_datasets`

## アーキテクチャ

```
neurosurrogate/                # コアライブラリ（フラット構造、scripts 非依存）
  neuron_core.py               # HH パラメータ jitclass / ゲート関数 @njit
                               # HH_RATE_COST_MAP, HH_COST, COMPARTMENT_TEMPLATES
  calc_engine.py               # unified_simulator, generic_euler_solver @njit
                               # ModelArgs / IndiceArgs namedtuple
                               # build_indices, calc_graph_laplacian
  xarray_utils.py              # StateAccumulator, set_coords (MultiIndex)
                               # set_i_internal, get_gate_numpy, transform_gate
  model.py                     # SINDyNeuroSurrogate, DummySurrogate
                               # 動的 Numba 関数生成 (_build_source)
  profiler.py                  # コスト静的解析 (build_feature_cost_map, static_calc_cost)
                               # 波形/スパイク指標 (calc_dynamic_metrics)
                               # SINDySummary, get_loggable_summary

scripts/                       # Hydra + MLflow オーケストレーション
  main.py                      # cli_flow: 学習 → MLflow ログ
  analysis.py                  # MLflow run 読み出し、surrogate 復元、評価
  marimo.py                    # marimo インタラクティブ UI
  conf/
    config.yaml                # defaults: sindy/base, sweeper params
    sindy/
      base.yaml                # PCA + STLSQ + gate/volt/identity/const ライブラリ
      hh_informed.yaml         # gate_poly_volt + alpha/beta 関数明示
      hh_relaxation.yaml       # relaxation1/relaxation2 形式
    feature_library_components.py  # LIB_BUILDER_REGISTRY, FUNC_REGISTRY
                                   # make_gate_lib / make_volt_lib /
                                   # make_relaxation_{1,2}var_lib /
                                   # make_gate_poly_volt_lib
    neuron_models.py           # MODEL_DEFINITIONS (hh, php, hhp, ..., hh7)
    current_generators.py      # apply(arr) -> None を返す高階関数
    preprocessor.py            # AutoEncoderPreprocessor (JAX/optax, PCA 代替)
    style/*.mplstyle           # base / paper / presentation
  utils/
    builder.py                 # build_surrogate, build_simulator_config,
                               # build_dataset, build_model
    mlflow_handler.py          # setup_mlflow, log_surrogate_summary,
                               # log_surrogate_model, load_surrogate_model
                               # SINDySurrogateMLflowModel (pyfunc)
    plots.py                   # draw_engine (matplotlib/plotly), spec_simple,
                               # spec_diff, view_model,
                               # plot_2d_attractor_comparison
```

## 不変条件・規約

### xarray レイアウト

`unified_simulator` 出力の `vars` は `(time, features)`、`features` は MultiIndex `(comp_id, variable, gate)`。

- 電位: `ds["vars"].sel(gate=False, comp_id=i)`
- ゲート / 潜在変数: `ds["vars"].sel(gate=True, comp_id=i)`
- 外部電流: `ds["I_ext"]` — `(time,)`
- 網内電流: `ds["I_internal"]` — `(time, node_id)`、グラフラプラシアンと `stim_idx` から `set_i_internal` で計算

`StateAccumulator` が `(comp_id, variable, gate)` の MultiIndex と初期値配列を組み立てる単一の入口。

### SoA シミュレータ

`calc_universal_deriv` は if 分岐なし。ノード種別ごとに `IndiceArgs` の `passive_ids` / `hh_ids` / `surr_ids` で独立ループを回す。`gate_offsets[i]` でノード `i` のゲート変数の状態ベクトル内オフセットを引く（ゲート無しは `-1`）。

新しいコンパートメント種を足す場合は **3 箇所を同期更新**：
1. `neuron_core.COMPARTMENT_TEMPLATES`
2. `calc_engine.build_indices`（ids 収集ループ）
3. `calc_engine.calc_universal_deriv`（独立ループ追加）

サロゲート未使用時は `DummySurrogate` が `sindy_args` を埋め、`build_indices` は `surr_comp=None` で `COMPARTMENT_TEMPLATES` のみを参照する。`surr_ids` が空なら実コストゼロ。

サロゲート使用時のみ `compartments = COMPARTMENT_TEMPLATES | {"surr": surr_comp}` でテンプレートが拡張される。

### 動的 Numba 関数生成

`SINDyNeuroSurrogate._build_source` が pysindy の `feature_names` から Python ソース文字列を組み立てて `exec` で `target_module`（= `feature_library_components` 経由で `neuron_core` の関数群）の名前空間にコンパイル。置換ルール：

- `1` → `1.0`（Numba 型推論のため）
- `^` → `**`

生成関数 `dynamic_compute_theta` は `@njit` で JIT され、`calc_universal_deriv` から `model_args.compute_theta(v, latent, i_int)` として呼ばれる。

`SINDyNeuroSurrogate.fit` は `target_module` を引数で受け取り、`mlflow_handler.SINDySurrogateMLflowModel.load_context` は `inspect.getfile(surrogate.target_module)` で取得したパスを `target_module_path` として保存・復元する。`target_module` の場所を変える場合は両方とも追従する。

### サロゲートの状態構造

`SINDyNeuroSurrogate.surr_comp` は preprocessor の出力次元から動的に作る：

- `init` = preprocessed の最初の時刻のベクトル
- `vars` = `["V", "latent1", "latent2", ...]`
- `gate` = `[False, True, True, ...]`

つまり潜在変数は何次元でも良い（`base.yaml` は PCA `n_components: 1` で 1 次元）。`calc_universal_deriv` の `surr_ids` ループは現状 `latent` を 1 つ前提（`curr_x[g_idx]` のスカラー取得、`xi_matrix[0]`/`[1]` の 2 行のみ）なので、多次元化する場合はここも書き換える。

### 演算コスト map の整合性

`profiler.build_feature_cost_map` は SINDy feature 名を regex でパースし `HH_RATE_COST_MAP` を参照する。

- `np.power(x, k)` → `(k-1)` 回の mul として展開
- 残った `*` は mul、`+`/`-` は pm でカウント
- 名前の長い関数から優先マッチ（部分一致バグ防止）

`static_calc_cost` は **未知の基底関数に対して fail-fast**（`ValueError`）。

新しい `@njit` 関数を SINDy ライブラリに追加したら：
1. `neuron_core.HH_RATE_COST_MAP` にコスト `{exp, div, pm, mul}` 追加
2. `feature_library_components.FUNC_REGISTRY` に登録（`make_gate_lib` で参照される）

`HH_COST` は `_get_original_hh_cost` で `calc_hh_channel` のソースを静的トレースした基準値。HH 微分計算側を変えたらここも更新する。

### Hydra config

`config.yaml` は最小構成：

```yaml
defaults:
  - sindy: base
  - _self_
hydra:
  run:    { dir: hydra-outputs/${now:%Y-%m-%d_%H-%M-%S} }
  sweep:  { dir: hydra-multiruns/${now:%Y-%m-%d_%H-%M-%S} }
  sweeper:
    params:
      sindy.optimizer.alpha: 0.01, 0.05, 0.1, 0.5, 1.0
```

`sindy/*.yaml` 側で preprocessor / optimizer / library_specs / datasets / train_comp_identifier を持つ。`datasets.pipeline` の `_target_` は `conf.current_generators.*` を指す。

### MLflow run 構造

現状はフラット：`train:{cfg_sindy.name}` の単一 run。各 run に以下がログされる：

- metrics: `nonzero_term_num`, `cost/{surrogate,original,diff}/{exp,div,pm,mul}`, `pca/*`
- params: STLSQ optimizer の `get_params()`
- artifacts: `equations.txt`, `coef.txt`, `features.json`, `features_active.json`, `sindy_coef.json`, `dataset.yaml`, `surrogate_model/`（pyfunc）, `misc/source.txt`

`load_surrogate_model(run_id)` で `pyfunc.load_model` → `_model_impl.python_model` 経由で `SINDySurrogateMLflowModel` インスタンスを取得。`unified_simulator` 専用で `predict` は未実装。

### Preprocessor

- `sklearn.decomposition.PCA`（デフォルト、`base.yaml`）
- `conf.preprocessor.AutoEncoderPreprocessor`（JAX/optax 実装、PCA と同一インターフェース）

`get_loggable_summary` は `isinstance(preprocessor, PCA)` で PCA メトリクス（explained variance, reconstruction MSE）の収集を分岐。AutoEncoder の指標を増やす場合はここに追加。

### 新規追加パターン

| 追加対象             | 変更箇所                                                                  |
| -------------------- | ------------------------------------------------------------------------- |
| 電流生成関数         | `scripts/conf/current_generators.py` + 各 `sindy/*.yaml::datasets.pipeline` |
| ニューロン構造       | `scripts/conf/neuron_models.py::MODEL_DEFINITIONS`                        |
| SINDy ライブラリ型   | `scripts/conf/feature_library_components.py::LIB_BUILDER_REGISTRY`        |
| 基底関数             | `neuron_core.py` (@njit) + `HH_RATE_COST_MAP` + `FUNC_REGISTRY`              |
| コンパートメント種   | `COMPARTMENT_TEMPLATES` + `build_indices` + `calc_universal_deriv`        |
| Preprocessor         | `scripts/conf/preprocessor.py` + `sindy/*.yaml::preprocessor._target_`    |

### 環境・スタイル

- Python >= 3.11、mypy strict（`scripts/` は除外）、ruff、pylint は `neurosurrogate` と `scripts` のみ
- `scripts/main.py` 冒頭で `HTTP_PROXY` / `HTTPS_PROXY` を空に上書き（社内プロキシ回避）
- matplotlib は `Agg` バックエンド固定、スタイルは `base.mplstyle` + `{paper|presentation}.mplstyle`
- Numba 初回呼び出しで遅延コンパイル、初回実行は遅い
- 積分器は固定ステップ Euler（`generic_euler_solver`）
- `lin_exp_form` は HH ゲート関数の `x / (exp(x) - 1)` 形を `|x| < 1e-8` で Taylor 展開に切替えて 0 除算を回避