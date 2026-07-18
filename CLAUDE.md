# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Standards [WRITTEN BY HUMAN - DO NOT OVERWRITE]
- HH型マルチコンパートメントニューロンの一部ノードをSINDyで抽出したサロゲート方程式に置換し、演算コスト削減と波形再現性を評価する研究コード
- 一時変数は同じ値を何度も使うような場合にのみ許可
  - NG: `x = obj.attr; f(x)`
  - OK: `f(obj.attr)`
- 大きな改装が終わったら、just test でエラーが出ないことを確認,tests/ 以下のテストは自由に追加して良い ただし、20s以下に抑えること
- Hooksで実行されるjust lint、just formatのエラーは都度対処すること
- surrogate/ 以下は _がプレフィックスで一括実行から除外、__がプレフィックスでgit除外

[以下のセクションは永続メモリとして上書きしても構いません。
ただし、30行以内程度に納め、本当に必要な情報だけを圧縮して書くこと]

## Commands

```bash
uv sync                                                           # 初期セットアップ（依存導入）
uv run scripts/main.py                                            # 実行 (fit+MLflow log のみ。kernel は回さない)
uv run scripts/main.py surrogate=_hh_informed                     # Hydraプリセット切替 (base=hh_full/traub_full、lib違いは _hh_informed/_hh_relaxation)
uv run scripts/main.py --multirun                                 # config.yaml の hydra.sweeper.params 直積 sweep (hh/traub × type{sindy,hybrid} × n_components{1,2} × preprocessor{pca,ae} = 16 run)
just test                  # pytest (tests/、Hydraプリセット読込→fit→置換シミュ→指標/描画) + main.py + marimo export
just format && just lint   # ruff fix+format / ruff+mypy (strict、scripts/ 除外)
just mlflow                # MLflow UI (port 5100、backend: mlflow.db)
just marimo                # marimo notebook (port 2700)
just marimo-mcp            # Claude Code MCP連携 (port 2701)
just clean-cache / clean-log
```

## Architecture

**Data flow**: Hydra conf → `DatasetConfig`/`NeuronGraph` → `unified_simulator()` (JAX Euler + `lax.scan`) → `NeuroSurrogateBase.fit()` (前処理 → PySINDy → `FeatureLibrary` から `compute_theta` 直接構築) → MLflow log。`neurosurrogate/__init__.py` で `jax_enable_x64` 強制 ON (HH ゲート exp 数値安定化に必須)。

**neurosurrogate/core/**: `network.py`(`Compartment`/`CompartmentType`/`NeuronGraph`/`DatasetConfig`/`CurrentConfig`。`chain(...)` ノード自動命名。置換の type 差替は surrogate/replace.py に移管、core は純粋データ構造) / `simulator.py`(`unified_simulator` + `calc_universal_deriv` が type 名 dispatch で `vmap` 一括計算、`generic_euler_solver` が `lax.scan`) / `coords.py`(xarray 座標の **write** 側。features=MultiIndex(comp_id,variable,gate)/gate=False→電位/node_id→I_internal を組立。`set_latent_coords`(V+latent 組立) + `transform_gate`(preprocessor で gate→latent 射影、ansatz/metrics 共用)) / `access.py`(同スキーマの **read** 規約を集約する唯一の場所。生 `sel` は全域禁止しここ経由。`potential`/`gate_matrix`/`comp_matrix`/`potential_matrix`/`i_ext_values`/`i_internal_values`/`time` 等 numpy accessor(計算層) + `trace`/`i_ext`/`i_internal` 等 (t,y) accessor(描画層)) / `opcost.py`(`OpCost` 代数、`+`/`*` で演算コスト集計)

**neurosurrogate/{compartments,models,currents}**: `compartments/{hh,traub,common}.py` 動力学 + `COMPARTMENT_TEMPLATES` (`hh.py` に `HH_RATE_ENTRIES`/`HH_GATE_PAIRS`/`HH_GATE_FORWARD`) / `models/` に `MCMODELS` + `traub19.py`(19-comp、traub.c 代数的等価) / `currents.py` に電流波形 + `CURRENT_MAP`

**neurosurrogate/surrogate/**: `base.py`(`NeuroSurrogateBase`、`__init__(datasets,train_comp_identifier,n_components)`。`n_components`=潜在方程式の次元 (ansatz が保持、fit で preprocessor spec へ注入=単一源)。`_meta`(=`SurrogateMeta`: surrogate_type/dataset/train_comp_id 保持)/bundle 保持、`save`/`load` は `surrogate.joblib` 単一ファイル。`surrogate.apply(dataset)` は `surr_comp_type` (学習結果から構築する abstract property) を渡し `replace.apply` に委譲) / `replace.py`(置換ドメイン一元化。`verdict(meta,comp)` 学習ドメイン照合3値判定 (`REPLACE` 型+params一致 / `MISMATCH` 型一致params違い / `SKIP` 型違い)、`replaceables(meta,ds)` fail first (MISMATCH有 or REPLACE皆無で `ValueError`)、`apply(meta,surr_type,ds)` が REPLACE ノードを `replace_nodes`+`dc_replace` で非破壊 type 差替 (`COMPARTMENT_TYPES` 共有参照で load 跨ぎ一致)。診断 `transform_gate` も verdict ガードで非 REPLACE を明示エラー化) / `sindy.py`(`SINDyNeuroSurrogate`、列構造 [V,latent…,u]) / `hybrid.py`(`HybridSINDyNeuroSurrogate`、列構造 [latent…,V] (u無)、AE preprocessor 利用可、n_components 3 が既定)。各 ansatz が fit 内で `Roles` を n_components から inline 構築 / `roles.py`(`Roles`: V=電位列/g=ゲート列群(=隠れ変数)/u=外電流列。方程式構造=ansatz ドメイン。`bind` 1本で項の args シンボル名 V/g/u を列へ束縛、libraries は duck typed 消費) / `bundle.py`(`PREPROCESSOR_CLS`={"pca","ae"} / `OPTIMIZER_CLS` / `SindyBundle`/`PreprocessorBundle`、`bundle.decode()` 統一) / `libraries/`(ロジック/カタログ分離。項=1つの sympy 式が真実源→func(lambdify)/name(subs)/cost(op_cost)/args(free_symbols を V,g,u 順)/arity 全派生。HH レート関数は数値安定 exp 保持のため未定義 Function シンボル+lambdify 注入。`entry.py`=ロジック(`op_cost` 木辿り→OpCost/`LibraryEntry`(expr,args,func。`argnames`=列束縛キー兼 展開規則の判定源)/`_group_by_argnames` 同一 argnames で 1 束/`SubLibrary.expand(spec,roles)` は spec={type,latents?} のみ受け (番号=latent序数、既定全)、展開は args が決定: g 持ち群=選択 latent へ複製・g 無し群=1束・u 持ち群=u 無し ansatz では脱落/`FeatureLibrary.build(specs,roles)` は catalog を遅延 import) / `catalog.py`=データ(`_entry`(args 自動導出)+`_R` レート式+項カタログ `LIB_ENTRIES` 1本。**全 type は互いに素** (同じ式を2 type が持たない。合成は yaml 側の責務: 例 forward+backward=全レート、forward+relaxation_decay=緩和形)。`_rate_types(prefix,pairs)` が (alpha,beta) ペア列から {gate_forward,gate_backward,gate_forward_product,gate_backward_product,relaxation_decay} を派生 → `hh_*` (3ゲート) / `traub_*` (V依存8ゲート、Q は XI 依存で対象外) + gate_poly_volt (latent 高次・V積。1次は basis) + basis。レート実装注入は `_RATE_NS`=vars(hh)|vars(traub)、コストは entry.py の `_RATE_COST_MAP`=HH|TRAUB。basis=V/u/latent 射影+定数1(V に相乗り) で他項と同じ規則に還元、latents:[] で V/u/1 のみ)) / `autoencoder.py`(`AutoEncoderPreprocessor`、PCA 互換 JAX-AE) / `__init__.py` に `SURR_CLS` 辞書 (`sindy`/`hybrid`)

**neurosurrogate/{metrics,view}**: `metrics/wave.py`(eFEL スパイク + RMSE/MAE、V 抽出は `core.access` 経由) / `view/{engine,specs,model}.py`(`draw_engine`/`TraceSpec`(t,y numpy 保持で Dataset 非依存)/`PanelSpec`/`error_fig` は engine、`plot_2d_attractor_comparison`+`draw_all(original,surrogate,comp_id,get_preprocessed)` (内部 jobs dict で全描画一括、失敗を error_fig に畳み識別子付き fig 列返却) は specs、`view_neuron_graph`/`view_model`+`model_figures` (識別子付き model 図列) は model、`current_preview_fig`/`sweep_fig` (marimo 非依存の純粋描画。analysis は UI 値引き出しのみ担い委譲) は utils。**view が (id,fig) 列を提供し analysis は保存/表示に流すだけ**。Dataset 掘削は `core.access` 経由)

**scripts/**: `main.py`(Hydra エントリ) / `mlflow_io.py`(MLflow I/O。`log_surrogate_model` は `surrogate.save` + `meta.yaml`(surrogate_type/dataset/train_comp_id) を artifact log、`load_surrogate_model` は `(NeuroSurrogateBase, SurrogateMeta)` tuple 返却) / `analysis/`(marimo 層。`ui.py`=widget builder / `actions.py`=calc gate+run load (`LoadedRun` は mlflow_io、`calc_eval` 置換後シミュ) / `view.py`=表示合成 (Panel 組立) / `panel.py`=保存機構 / `mode/{single,sweep}.py`=per-mode compute+draw。依存: view→actions/ui/panel/mode、actions→mode、循環無) / `marimo.py` / `conf/`(`config.yaml` に共通 `surrogate.fit.optimizer` + `init.n_components` 既定 (1) 集約 + `surrogate/*.yaml` プリセット (base=`hh_full`/`traub_full` は lib=full の素体、type/n_components/preprocessor は `hydra.sweeper.params` 既定の `--multirun` で直積に振る。lib 本質差の `_hh_informed`/`_hh_relaxation` のみ個別)。`n_components` は `init` 直下 (preprocessor spec でなく)、library_specs は type+latent序数 `latents` のみ (省略=全latent。V/u/定数は項が固定保有))
