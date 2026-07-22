from pathlib import Path

import matplotlib.pyplot as plt

# matplotlib style の適用。marimo 非依存の leaf に置き、notebook と CLI が同じ
# スタイル (conf/style/*.mplstyle) で図を出す。
STYLE_DIR = Path(__file__).resolve().parents[1] / "conf" / "style"
STYLES = ["paper", "presentation"]


def setup_mpl(matplotlib_style: str) -> None:
    plt.style.use(STYLE_DIR / "base.mplstyle")
    plt.style.use(STYLE_DIR / f"{matplotlib_style}.mplstyle")
