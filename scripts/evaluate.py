"""評価の入口: `just eval <系列> <学習 run の id>...`。系列はカタログの名前。

結果は MLflow の波形 experiment に入り、live-textbook の research/ の冊が読んで描く。
"""

import argparse
import logging

from catalog import SERIES
from mlflow_io.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", choices=sorted(SERIES))
    parser.add_argument("run_ids", nargs="+", metavar="run_id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate(args.series, tuple(args.run_ids))


if __name__ == "__main__":
    main()
