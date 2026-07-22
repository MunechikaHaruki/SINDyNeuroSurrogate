# 置換対象のコンパートメント種類 → 適用先 MC モデル候補 (モデルペアの右側)。
# 実際に選べる run は replace の互換判定でさらに絞られる (ui.make_setting_ui /
# sweep_traces.py)。marimo 非依存の leaf に置き、notebook と CLI が同じ対応を見る。
TARGET_MODEL: dict[str, list[str]] = {
    "hh": ["hh", "phhhp"],
    "traub": ["traub19", "traub"],
}
