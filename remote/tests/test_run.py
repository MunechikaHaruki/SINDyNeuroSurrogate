"""計算の入口 (`neurosurrogate/run.py`) の smoke: spec.json 1 つ → run のディレクトリ。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは run の置き場と
波形の保存形式を通す。ハッシュは段取り (TS) が組むので、ここでは置き場の形
(`parts/<学習器>`、`models/<モデル>`、`inputs/<入力>/<original かモデル>`。
学習データは学習の入力の `inputs/<入力>/original`) だけを守って名前を適当に付ける。
"""

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from test_surrogate import declared_config

from neurosurrogate import run
from neurosurrogate.physics.simulate import EvalSeries, potential, simulate


def _series() -> dict:
    """学習と同じ入力を**さらに短く**した 2 点の掃引。見たいのは保存と読込の往復で
    あって波形の質ではないので、シミュ長は最小で足りる。"""
    spec = declared_config("_test_hh_sindy")["data"]
    return {
        "spec": {**spec, "stim": "soma"},
        "replace_targets": ["soma"],
        "param": "duration",
        "values": [170.0, 190.0],
    }


def _put(tmp: Path, place: str, spec: dict) -> Path:
    """spec を回して runs/<place>/ を埋める。"""
    path = tmp / f"{place.replace('/', '_')}.json"
    path.write_text(
        json.dumps({"format": run._FORMATS[spec["kind"]], "device": "cpu", **spec})
    )
    run._run(path, tmp / "runs" / place, place)
    return tmp / "runs" / place


def _train(config: dict) -> dict:
    """学習用の電流を単発で回す学習の入力の、元のモデルの sim の spec。"""
    input_ = {"spec": config["data"], "param": None, "values": []}
    return {"kind": "sim", "input": input_, "model": None}


def _sim(series: dict, model: dict | None) -> dict:
    """系列を入力とモデルに分けた sim の spec。"""
    input_ = {k: v for k, v in series.items() if k != "replace_targets"}
    return {"kind": "sim", "input": input_, "model": model}


_MODEL = {"surrogate": "M", "replace_targets": ["soma"]}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """学習の入力の元のモデル・学習器・それを組んだモデルと、1 つの入力を元のモデルと
    置き換えたモデルで回した runs/。"""
    tmp = tmp_path_factory.mktemp("runs")
    config = declared_config("_test_hh_sindy")
    _put(tmp, "inputs/T/original", _train(config))
    shared = {k: config[k] for k in ("cell", "form", "train_comp")}
    [group] = config["groups"]
    _put(tmp, "parts/P", {"kind": "part", "input": "T", **shared, "group": group})
    model = {"kind": "model", "cell": config["cell"], "form": config["form"]}
    _put(tmp, "models/M", {**model, "parts": ["P"]})
    _put(tmp, "inputs/I/original", _sim(_series(), None))
    _put(tmp, "inputs/I/R", _sim(_series(), _MODEL))
    return tmp / "runs"


def test_runs_round_trip_and_are_not_rerun(runs: Path) -> None:
    """波形は点の並びごと float64 で往復し、ディレクトリがあれば回し直さない。"""
    series = EvalSeries.from_dict(_series())
    waves = run._read_waves(runs / "inputs/I/original/waves")
    np.testing.assert_allclose(
        potential(waves[1], 0),
        potential(simulate(series.points[1].run_input()), 0),
        rtol=1e-5,
    )
    surr = run._read_waves(runs / "inputs/I/R/waves")
    assert not np.allclose(potential(surr[1], 0), potential(waves[1], 0))
    meta = json.loads((runs / "inputs/I/R/meta.json").read_text())
    assert (meta["kind"], meta["name"], meta["unit"]) == ("replaced", "inputs/I/R", "")
    original = json.loads((runs / "inputs/I/original/meta.json").read_text())
    assert original["kind"] == "original"
    before = (runs / "inputs/I/original/meta.json").read_text()
    spec = runs.parent / "inputs_I_original.json"
    run._run(spec, runs / "inputs/I/original", "again")
    assert (runs / "inputs/I/original/meta.json").read_text() == before
    assert not list(runs.glob("**/*.partial"))


def test_runs_hold_what_the_figures_read(runs: Path) -> None:
    """図が読むものは全部 run の中にある: 学習データ、学習器の中身、素の float64 の
    波形、置換した comp の射影、学習側の指標。JSON に nan は出さない
    (JS が読めない)。"""
    decode = json.loads((runs / "parts/P/compressor/decode.json").read_text())
    latents = [axis["latent"] for axis in decode["axes"]]
    n_time = json.loads((runs / "inputs/T/original/waves/meta.json").read_text())
    latent = np.fromfile(runs / "parts/P/fit/0/latent.f64", dtype="<f8")
    assert latent.size == len(latents) * n_time["n"][0]
    recon = np.fromfile(runs / "parts/P/fit/0/recon_rmse.f64", dtype="<f8")
    assert recon.size == n_time["n"][0]
    assert [p.name for p in (runs / "parts/P/fit").iterdir()] == ["0"]
    assert (runs / "parts/P/learner.joblib").exists()
    kinds = [
        json.loads((runs / p / "meta.json").read_text())["kind"]
        for p in ("inputs/T/original", "parts/P", "models/M")
    ]
    assert kinds == ["original", "part", "model"]

    meta = json.loads((runs / "inputs/I/original/waves/meta.json").read_text())
    assert meta["names"] == {"soma": 0}
    n_vars = sum(1 for comp, _, _ in meta["features"] if comp == 0)
    raw = np.fromfile(runs / "inputs/I/original/waves/1/comp0.f64", dtype="<f8")
    assert raw.size == n_vars * meta["n"][1]

    def strict(path: str) -> dict:
        def reject(token: str) -> None:
            raise ValueError(token)

        text = (runs / path).read_text()
        return cast(dict, json.loads(text, parse_constant=reject))

    # run は生データだけを持つ。スパイクの特徴量は API が頼まれるたびに計算する
    assert not any(runs.glob("inputs/I/*/spikes.json"))
    assert "cost/surr-orig/mul" in strict("models/M/summary.json")
    assert strict("parts/P/rhs/own.json")["kind"] == "sindy"
    assert strict("parts/P/compressor/own.json")["kind"] == "pca"
    assert "reconstruction_mse_ratio" in strict("parts/P/fit/0/summary.json")
    projected = json.loads((runs / "inputs/I/R/projected/soma/meta.json").read_text())
    names = [v for _, v, _ in projected["features"]]
    assert names[0] == "V" and names[1:] == latents
    assert all(
        m["elapsed"] >= 0
        for m in map(
            lambda p: json.loads((runs / p / "meta.json").read_text()),
            ("inputs/T/original", "parts/P", "models/M"),
        )
    )


def test_a_kam_model_holds_what_kepler_draws(tmp_path: Path) -> None:
    """KAM のモデルは学習器を持たず、学習の入力を指して、条件 (kam.json) と、
    元のモデルの等価電位 (equivalent/) と、同じ電流で回した縮約 (reduced/) を書く。"""
    config = declared_config("_test_traub_kam")
    _put(tmp_path, "inputs/T/original", _train(config))
    kam = {"groups": config["groups"], "input": "T", "train_comp": None}
    model = {"kind": "model", "cell": config["cell"], "form": config["form"]}
    root = _put(tmp_path, "models/K", {**model, "parts": [], "kam": kam})
    files = json.loads((root / "kam.json").read_text())
    assert [g["gates"] for g in files["groups"]] == [["N", "H"], ["A", "C"]]
    equivalent = json.loads((root / "equivalent/meta.json").read_text())
    assert equivalent["names"] == {"soma": 0}
    assert [v for _, v, _ in equivalent["features"]][:3] == ["V", "N", "H"]
    reduced = json.loads((root / "reduced/meta.json").read_text())
    data = json.loads((tmp_path / "runs/inputs/T/original/waves/meta.json").read_text())
    assert reduced["n"] == data["n"]
    assert "cost/surr-orig/mul" in json.loads((root / "summary.json").read_text())


def _unreplaceable() -> dict:
    """hh で学習した run では置換できない traub の系列。"""
    return {
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


def test_unreplaceable_pair_writes_nothing(runs: Path) -> None:
    """置換できない組み合わせ (hh で学習した run で traub の系列) は何も書かず、
    終了コードで段取りに知らせる。"""
    with pytest.raises(SystemExit) as raised:
        _put(runs.parent, "inputs/J/X", _sim(_unreplaceable(), _MODEL))
    assert raised.value.code == run._NOT_REPLACEABLE
    assert not (runs / "inputs/J/X").exists()
    assert not (runs / "inputs/J/X.partial").exists()


def test_a_run_being_written_is_not_written_twice(runs: Path) -> None:
    """書きかけの run を生きたプロセスが書いていれば終了コードで知らせて触らず、
    書き手が死んだ残骸なら片付けて回し直す。"""
    partial = runs / "inputs/I/Y.partial"
    partial.mkdir()
    (partial / "pid").write_text(str(os.getpid()))
    with pytest.raises(SystemExit) as raised:
        _put(runs.parent, "inputs/I/Y", _sim(_series(), _MODEL))
    assert raised.value.code == run._RUNNING and (partial / "pid").exists()
    (partial / "pid").write_text("999999999")
    with pytest.raises(SystemExit):
        _put(runs.parent, "inputs/I/Y", _sim(_unreplaceable(), _MODEL))
    assert not partial.exists()
