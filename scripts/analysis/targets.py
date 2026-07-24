# 置換対象のコンパートメント種類 → 適用先 MC モデル候補 (モデルペアの右側)。
# 実際に選べる run は replace の互換判定でさらに絞られる (ui.make_setting_ui /
# sweep_traces.py)。marimo 非依存の leaf に置き、notebook と CLI が同じ対応を見る。
TARGET_MODEL: dict[str, list[str]] = {
    "hh": ["hh", "phhhp"],
    # traub19_soma = soma だけ置換対象に残した 19-comp (dendrite はダミー型)。
    # comp_type=traub の run をこれへ適用すると soma 1 ノードだけ置換される。
    "traub": ["traub19", "traub", "traub19_soma"],
}
