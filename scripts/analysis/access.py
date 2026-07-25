from __future__ import annotations

import marimo as mo

from neurosurrogate.surrogate.meta import SurrogateMeta

# ---------------------------------------------------------------------------
# 設定 cfg (base.json⊕meta.json) / draw_ui / meta の read 規約の集約
# (leaf: marimo と meta のみ依存)。
#
# base_ui/setting_ui widget は廃止 → シミュ入力は cfg (plain dict) から読む。
# 表示設定だけ draw_ui widget から、comp_type は選択 run の SurrogateMeta から。
# ui.py に置くと ui→mode の import と衝突する (循環) ため独立 module に切出す。
# ---------------------------------------------------------------------------


ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢


def preset_of(preset_ui: mo.ui.dropdown) -> str:
    """選択中の preset (surrogate/*.yaml)。ALL_PRESETS なら絞り込まない。
    run_selector の上流フィルタ (preset を変えると出す run 群が変わる)。"""
    return str(preset_ui.value)


# --- cfg (base.json⊕meta.json の解決 dict) → domain 関数の kwargs bundle ------
#
# 単発 scalar getter を並べる代わりに、呼び出し側が `f(**bundle)` で splat できる
# 「意味の単位」を返す。access は cfg のセクション構造 (base/sim/sweep) を domain
# 関数 (build_dataset / CurrentSweepConfig) の引数形へ写す一箇所になる。


def _current_params(cfg: dict) -> dict:
    """current_params 値 (空 = 電流関数の既定を使う)。"""
    return cfg.get("sim", {}).get("current_params") or {}


def current_type_of(cfg: dict) -> str:
    """電流タイプ名 (is_sweepable の判定用に単独で要る)。"""
    return str(cfg["base"]["sim_current_type"])


def dt_of(cfg: dict) -> float:
    """シミュ刻み幅 (evaluate_sweep が CurrentSweepConfig とは別に受ける)。"""
    return float(cfg["base"]["dt"])


def current_inputs(cfg: dict) -> dict:
    """電流シミュ入力を `DatasetConfig.build_dataset` / `current_preview_fig` の
    kwargs 形で返す (model_name だけ呼び出し側が付す)。cfg の base/sim を 1 単位に。"""
    return {
        "current_type": current_type_of(cfg),
        "dt": dt_of(cfg),
        "current_params": _current_params(cfg),
    }


def sweep_config_inputs(cfg: dict) -> dict:
    """`CurrentSweepConfig(**)` の kwargs。cfg の base(電流型)/sim(base_params)/
    sweep(掃引範囲) を domain config の引数形へまとめる。"""
    return {
        "current_type": current_type_of(cfg),
        "base_params": _current_params(cfg),
        **cfg.get("sweep", {}),  # sweep_param/amp_start/amp_stop/amp_steps
    }


# --- 選択 run の meta からの読取 ----------------------------------------------


def comp_type_of(meta: SurrogateMeta) -> str:
    """置換対象のコンパートメント種類名。適用先は選ばず TARGET_MODEL[comp_type] を
    全部 simulate する (single mode)。UI 選択でなく学習成果物 (meta) が単一源。"""
    return meta.comp_type.name


# --- draw 値 (marimo で draw_ui.value 済みの plain dict) の読取 ----------------


def eval_comp_of(draw: dict) -> str:
    """評価対象 comp 名 (diff/指標の比較対象、1 件)。"""
    return str(draw["eval_comp"])


def view_comps_of(draw: dict) -> list[str] | None:
    """全 comp を並べる図 (simple/train_*) に描く comp 名。空選択 = 制限なし (None)。"""
    return [str(c) for c in draw["view_comps"]] or None


def plt_style_of(draw: dict) -> str:
    """matplotlib style 名 (描画設定)。"""
    return str(draw["plt_style"])


def valid_or(value: object, options: object, default: object) -> object:
    """復元値が現 options に含まれれば採用、無ければ default。run 集合変化などで
    無効化した選択 (dropdown 値) を既定へ吸収する共通規約。"""
    return value if value in options else default  # type: ignore[operator]
