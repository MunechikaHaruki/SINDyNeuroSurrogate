from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import xarray as xr
from tqdm import tqdm

CHUNK_SIZE = 10000


def run_simulation(
    fp, NT, dset_shape, dset_chunks, initialize_func, step_chunk, get_I_ext_chunk
):
    initialize_func()

    dset = fp.create_dataset(
        "vars",
        shape=dset_shape,
        dtype="float64",
        chunks=dset_chunks,
    )

    chunk_template = np.empty(dset_chunks, dtype=np.float64)

    chunk_num = NT // CHUNK_SIZE

    for cn in tqdm(range(chunk_num)):
        start_index = CHUNK_SIZE * cn
        end_index = start_index + CHUNK_SIZE

        chunk = chunk_template.copy()
        i_ext_chunk = get_I_ext_chunk(start_index, end_index)
        step_chunk(chunk, i_ext_chunk)
        dset[start_index:end_index] = chunk

    remaining_steps = NT % CHUNK_SIZE
    if remaining_steps > 0:
        start_index = CHUNK_SIZE * chunk_num
        end_index = start_index + remaining_steps

        remaining_chunk = chunk_template[:remaining_steps].copy()
        i_ext_chunk = get_I_ext_chunk(start_index, end_index)
        step_chunk(remaining_chunk, i_ext_chunk)
        dset[start_index:end_index] = remaining_chunk


MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "hh": {
        "features": ["V", "M", "H", "N"],
        "dims": (["time", "features"], ["time"]),
        "compartments": False,
    },
    "hh3": {
        "features": ["V", "M", "H", "N", "V_pre", "V_post"],
        "dims": (["time", "features"], ["time"]),
        "compartments": False,
    },
    "traub": {
        "features": ["V", "XI", "M", "S", "N", "C", "A", "H", "R", "B", "Q"],
        "dims": (["time", "features", "compartments"], ["time", "compartments"]),
        "compartments": True,
    },
}


def calc_ThreeComp_internal(dataset, params):
    I_pre = params.G_12 * (
        dataset["vars"].sel(features="V_pre") - dataset["vars"].sel(features="V")
    )
    I_post = params.G_23 * (
        dataset["vars"].sel(features="V") - dataset["vars"].sel(features="V_post")
    )
    I_soma = I_pre - I_post

    dataset["I_internal"] = xr.concat(
        [I_pre, I_post, I_soma], dim="direction"
    ).assign_coords(direction=["pre", "post", "soma"])


def preprocess_dataset(model_type: str, file_path: Path, params):
    config = MODEL_CONFIG[model_type]

    with h5py.File(file_path, "r") as f:
        coords: Dict[str, Any] = {
            "time": f["time"],
            "features": config["features"],
        }
        if config["compartments"]:
            coords["compartments"] = np.arange(f["vars"].shape[-1])

        dataset = xr.Dataset(
            {
                "vars": (config["dims"][0], f["vars"]),
                "I_ext": (config["dims"][1], f["I_ext"]),
            },
            coords=coords,
        )

        if model_type == "hh3":
            calc_ThreeComp_internal(dataset, params)

    return dataset
