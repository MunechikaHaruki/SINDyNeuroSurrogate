# 置換対象のコンパートメント種類 → 適用先 MC モデル候補。single mode は選択 run の
# comp_type からこのリスト全部を simulate する (target は単一選択しない)。互換性の
# 無い適用先は評価側で error 表示。marimo 非依存の leaf → notebook と CLI 共有。
TARGET_MODEL: dict[str, list[str]] = {
    "hh": ["hh", "phhhp"],
    # traub19_soma = soma だけ置換対象に残した 19-comp (dendrite はダミー型)。
    # comp_type=traub の run をこれへ適用すると soma 1 ノードだけ置換される。
    "traub": ["traub19", "traub", "traub19_soma", "traub19_soma_dendstim"],
}
