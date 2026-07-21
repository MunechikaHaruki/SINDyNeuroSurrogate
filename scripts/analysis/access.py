from __future__ import annotations

import marimo as mo

# ---------------------------------------------------------------------------
# base_ui read 規約の集約 (leaf: marimo のみ依存)。
#
# ui/mode/view が共有する base_ui の掘削をここへ一元化。ui.py に置くと
# ui→mode の import と衝突し mode 側が使えない (循環) ため独立 module に切出す。
# ---------------------------------------------------------------------------


def target_of(base_ui: mo.ui.dictionary) -> str:
    """適用先 MC モデル名 (model_pair の target)。"""
    return str(base_ui["model_pair"].value[1])


def train_of(base_ui: mo.ui.dictionary) -> str:
    """学習元 train_model 名 (model_pair の train)。"""
    return str(base_ui["model_pair"].value[0])


def current_of(base_ui: mo.ui.dictionary) -> str:
    """選択電流タイプ名。"""
    return str(base_ui["sim_current_type"].value)
