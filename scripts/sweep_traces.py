"""preset (surrogate/*.yaml) ごとに MLflow run を集め、電流振幅を掃引した波形格子
図 (sweep_traces) を marimo を介さず PNG 出力する CLI。

marimo の sweep モードと同じ経路を通る (evaluate_sweep → sweep_trace_grid_fig)。
run の絞り込みも UI と同一規則: preset 一致 × 適用先モデルへ実際に置換できる
(`replaced_names` 非空) run のみ。識別ラベルも `sweep_labels` を共有するので、
notebook で出した図と行の並び・名前が一致する。

例:
    uv run scripts/sweep_traces.py                                # 全 preset
    uv run scripts/sweep_traces.py -p traub_useSingleCompForTeachingData
    uv run scripts/sweep_traces.py -p hh_hybrid --current step --steps 6 \
        --current-param duration=200
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from analysis.style import STYLES, setup_mpl
from analysis.targets import TARGET_MODEL
from mlflow_io import get_runs_df, load_runs, setup_mlflow

from neurosurrogate.metrics.eval_sweep import (
    CurrentSweepConfig,
    evaluate_sweep,
    sweep_labels,
    sweepable_params,
)
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.utils import sweep_trace_grid_fig

logger = logging.getLogger(__name__)

RESULT_DIR = Path(__file__).resolve().parent / "conf" / "surrogate" / "result"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-p",
        "--preset",
        nargs="+",
        help="対象 preset (surrogate/*.yaml 名)。既定は run のある全 preset",
    )
    p.add_argument(
        "--target",
        help="適用先 MC モデル。既定は preset の comp_type から TARGET_MODEL の先頭",
    )
    p.add_argument("--current", default="lin&steady", help="電流タイプ")
    p.add_argument(
        "--sweep-param", help="掃引パラメータ。既定は電流の第 1 numeric 引数"
    )
    p.add_argument("--amp-start", type=float, default=-5.0)
    p.add_argument("--amp-stop", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=10, help="掃引点数 (= 図の列数)")
    p.add_argument(
        "--current-param",
        nargs="+",
        default=[],
        metavar="K=V",
        help="掃引パラメータ以外の電流引数 (例 duration=200)",
    )
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--eval-comp", default="soma", help="波形を描く comp 名")
    p.add_argument("--max-runs", type=int, help="preset ごとの run 上限 (新しい順)")
    p.add_argument("--style", default=STYLES[1], choices=STYLES)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=RESULT_DIR,
        help=f"出力先 (既定 {RESULT_DIR}、<preset>_sweep_traces.png で平置き)",
    )
    return p.parse_args()


def _current_params(pairs: list[str]) -> dict:
    """`K=V` 列 → 電流引数 dict。数値化できるものは float に (duration 等)。"""
    params = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--current-param は K=V 形式: {pair}")
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params


def _target_of(runs: pd.DataFrame, preset: str) -> str:
    """preset に属する run の comp_type から適用先モデルを決める (--target 未指定時)。
    種類が混在する preset は自動では選べないので明示を促す。"""
    comp_types = sorted(runs["comp_type"].unique())
    if len(comp_types) != 1:
        raise ValueError(
            f"preset={preset} の comp_type が {comp_types} → --target で適用先を指定"
        )
    targets = TARGET_MODEL.get(comp_types[0])
    if not targets:
        raise ValueError(
            f"comp_type={comp_types[0]} の適用先が TARGET_MODEL に無い "
            "(analysis/targets.py に定義する)"
        )
    return targets[0]


def _replaceable_runs(runs: pd.DataFrame, target: str) -> pd.DataFrame:
    """target モデルへ実際に置換できる run だけ (互換判定は replace ドメイン)。"""
    net = MCMODELS[target]
    return runs[runs["meta"].map(lambda m: bool(replaced_names(m, net)))]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    params = sweepable_params(args.current)
    if not params:
        raise ValueError(f"current={args.current} に掃引できる numeric 引数が無い")
    setup_mpl(args.style)
    setup_mlflow()

    runs_df = get_runs_df()
    presets = args.preset or sorted(runs_df["preset"].dropna().unique())
    for preset in presets:
        in_preset = runs_df[runs_df["preset"] == preset]
        if in_preset.empty:
            logger.warning(f"[{preset}] run が無い → skip")
            continue
        target = args.target or _target_of(in_preset, preset)
        runs = _replaceable_runs(in_preset, target)
        if runs.empty:
            logger.warning(f"[{preset}] {target} へ置換できる run が無い → skip")
            continue
        # runs_df は start_time 降順 → head が最新
        if args.max_runs:
            runs = runs.head(args.max_runs)

        logger.info(f"[{preset}] → {target}: {len(runs)} run で掃引")
        surrogates = load_runs(runs["run_id"].tolist())
        labels = sweep_labels(surrogates)
        sweep_eval = evaluate_sweep(
            dict(zip(labels, surrogates, strict=True)),
            model_name=target,
            dt=args.dt,
            cfg=CurrentSweepConfig(
                current_type=args.current,
                sweep_param=args.sweep_param or params[0],
                amp_start=args.amp_start,
                amp_stop=args.amp_stop,
                amp_steps=args.steps,
                base_params=_current_params(args.current_param),
            ),
        )
        # result 直下へ平置き (preset 名をファイル名に冠する) → marimo の保存先
        # `<preset>_result/` とは別ファイルなので上書きしない。
        args.out_dir.mkdir(parents=True, exist_ok=True)
        dest = args.out_dir / f"{preset}_sweep_traces.png"
        sweep_trace_grid_fig(sweep_eval, args.eval_comp, labels).savefig(
            dest, dpi=300, bbox_inches="tight"
        )
        logger.info(f"[{preset}] 保存: {dest}")


if __name__ == "__main__":
    main()
