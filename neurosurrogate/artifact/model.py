"""保存方式に依存しない成果物 1 件の運搬形。"""

from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Artifact:
    """成果物 1 件 = 保存段で使う名前 + 図または表。"""

    name: str
    obj: Figure | pd.DataFrame
