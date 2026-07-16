"""描画テストを headless 化。view を import する前に効かせる必要がある。"""

import matplotlib

matplotlib.use("Agg")
