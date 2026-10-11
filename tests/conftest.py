"""`scripts/` を import path へ入れる: 回したい条件のカタログ (`catalog.py`) と
MLflow I/O (`mlflow_io/`) はドメイン層でなくここに住む。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
