"""**結果ドメイン**: 回した結果を軸に開き、宣言に従って報告 (図/表 + 由来) へ畳む。

- `results.py` — `SeriesView`/`ResultSet` (点軸 × run 軸。**run 軸を掛ける唯一の場所**)
- `spec.py` — `draw.json` → `ReportSpec`/`DrawSpec`/`CompareSpec` (宣言の型)
- `grid.py` — 軸に沿った図表 (波形格子・点軸メトリクス)
- `save.py` — `SaveEntry` (成果物 1 件 = 表示名 + 中身 + 由来) と書き出し
- `build.py` — 組立と `render_report` (marimo から呼ぶ入口)

**ドメインを横断する唯一の層**: 波形も surrogate の自己記述も互いを知らないので、
1 つの報告に束ねる責務をここへ集める。marimo/mlflow 非依存。
"""

from __future__ import annotations

from .build import eval_report as eval_report
from .build import load_and_render_report as load_and_render_report
from .build import model_report as model_report
from .build import render_report as render_report
from .results import ResultSet as ResultSet
from .results import SeriesView as SeriesView
from .results import run_names as run_names
from .results import series_matrix as series_matrix
from .save import SaveEntry as SaveEntry
from .save import save_entries as save_entries
from .save import slug as slug
from .spec import ALL_KINDS as ALL_KINDS
from .spec import CompareSpec as CompareSpec
from .spec import DrawSpec as DrawSpec
from .spec import ReportSpec as ReportSpec
