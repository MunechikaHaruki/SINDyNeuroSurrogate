"""計算の入口 (`neurosurrogate/run.py`) の smoke: spec.json 1 つ → run のディレクトリ。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは run の置き場と
波形の保存形式を通す。ハッシュは段取り (TS) が組むので、ここではディレクトリ名を
適当に付ける。
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from test_surrogate import preset_config

from neurosurrogate import run
from neurosurrogate.core import access
from neurosurrogate.sim.run import _simulate
from neurosurrogate.sim.spec import EvalSeries


def _series() -> dict:
    """学習と同じ入力を**さらに短く**した 2 点の掃引。見たいのは保存と読込の往復で
    あって波形の質ではないので、シミュ長は最小で足りる。"""
    spec = preset_config("_test_hh_sindy")["spec"]["datasets"]
    return {
        "spec": {**spec, "stim": "soma"},
        "replace_targets": ["soma"],
        "param": "duration",
        "values": [170.0, 190.0],
    }


def _put(tmp: Path, name: str, spec: dict) -> Path:
    """spec を回して runs/<name>/ を埋める。"""
    path = tmp / f"{name}.json"
    path.write_text(json.dumps({"format": run._FORMAT, **spec}))
    run._run(path, tmp / "runs" / name, name)
    return tmp / "runs" / name


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """学習・原系・置換系を 1 本ずつ回した runs/。"""
    tmp = tmp_path_factory.mktemp("runs")
    _put(tmp, "T", {"kind": "train", "surrogate": preset_config("_test_hh_sindy")})
    series = _series()
    original = {k: v for k, v in series.items() if k != "replace_targets"}
    _put(tmp, "O", {"kind": "original", "series": original})
    _put(
        tmp,
        "R",
        {"kind": "replaced", "series": series, "original": "O", "train": "T"},
    )
    return tmp / "runs"


def test_runs_round_trip_and_are_not_rerun(runs: Path) -> None:
    """波形は点の並びごと float32 で往復し、ディレクトリがあれば回し直さない。"""
    series = EvalSeries.from_dict(_series())
    waves = run._read_waves(runs / "O/waves")
    np.testing.assert_allclose(
        access.potential(waves[1], 0),
        access.potential(_simulate(series.points[1], None, ()), 0),
        rtol=1e-5,
    )
    surr = run._read_waves(runs / "R/waves")
    assert not np.allclose(access.potential(surr[1], 0), access.potential(waves[1], 0))
    meta = json.loads((runs / "R/meta.json").read_text())
    assert (meta["kind"], meta["name"], meta["unit"]) == ("replaced", "R", "")
    before = (runs / "O/meta.json").read_text()
    run._run(runs.parent / "O.json", runs / "O", "again")
    assert (runs / "O/meta.json").read_text() == before
    assert not list(runs.glob("*.partial"))


def test_runs_hold_what_the_figures_read(runs: Path) -> None:
    """図が読むものは全部 run の中にある: 学習 run の中身、素の float32 の波形、
    全 comp・全点の指標、置換した comp の射影、学習側の指標。JSON に nan は出さない
    (JS が読めない)。"""
    model = json.loads((runs / "T/description/model.json").read_text())
    n_time = json.loads((runs / "T/description/train/meta.json").read_text())
    x = np.fromfile(runs / "T/description/fit/0/x.f32", dtype="<f4")
    assert x.size == len(model["x_names"]) * n_time["n"][0]
    assert (runs / "T/surrogate/spec.json").exists()

    meta = json.loads((runs / "O/waves/meta.json").read_text())
    assert meta["names"] == {"soma": 0}
    n_vars = sum(1 for comp, _, _ in meta["features"] if comp == 0)
    raw = np.fromfile(runs / "O/waves/1/comp0.f32", dtype="<f4")
    assert raw.size == n_vars * meta["n"][1]

    def strict(path: str) -> dict:
        def reject(token: str) -> None:
            raise ValueError(token)

        text = (runs / "R" / path).read_text()
        return cast(dict, json.loads(text, parse_constant=reject))

    metrics = strict("metrics.json")
    assert list(metrics) == ["soma"] and len(metrics["soma"]) == 2
    assert "nnz" in strict("summary.json")
    projected = json.loads((runs / "R/projected/soma/meta.json").read_text())
    assert [v for _, v, _ in projected["features"]][0] == "V"


def test_unreplaceable_pair_writes_nothing(runs: Path) -> None:
    """置換できない組み合わせ (hh で学習した run で traub の系列) は何も書かず、
    終了コードで段取りに知らせる。"""
    series = {
        "spec": {
            "target": "traub",
            "current_type": "lin&steady",
            "dt": 0.01,
            "current_params": {"duration": 20.0, "value": 3.0},
            "stim": "soma",
        },
        "replace_targets": ["soma"],
        "param": None,
        "values": [],
    }
    spec = {"kind": "replaced", "series": series, "original": "O", "train": "T"}
    with pytest.raises(SystemExit) as raised:
        _put(runs.parent, "X", spec)
    assert raised.value.code == run._NOT_REPLACEABLE
    assert not (runs / "X").exists() and not (runs / "X.partial").exists()
