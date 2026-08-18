"""experiment の解決と鍵引きの共通手続き。

`series` も `report` も「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」の
4 点セットで、**鍵の組み方と書き方だけが experiment ごとに違う**。残る 2 つをここに 1 つ
置く = 綴りの揃った同型コードを module ごとに写さない。

**書く側だけが experiment を作る** (`exp_id`)。探すだけの経路 (`latest_by_tag`) は
作らない — 鍵を変えるたび空の experiment が生えるのを防ぐ。
"""

import mlflow
from mlflow.entities import Run


def exp_id(exp_name: str) -> str:
    """experiment の id (**無ければ作る** = 書く側の入口)。`set_experiment` は学習側の
    既定を書き換えるので使わず、run ごとに experiment_id を指定するための値。"""
    exp = mlflow.get_experiment_by_name(exp_name)
    return exp.experiment_id if exp else mlflow.create_experiment(exp_name)


def latest_by_tag(exp_name: str, tag: str, value: str) -> Run | None:
    """その tag が一致する最新 run (無ければ None)。experiment 自体が無ければ探索を
    諦める = **読む経路は experiment を作らない**。"""
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        return None
    found = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.{tag} = '{value}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return found[0] if found else None
