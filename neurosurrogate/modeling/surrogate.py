# mypy: ignore-errors
import importlib
import os
import sys
from datetime import datetime

import numpy as np
import pysindy as ps
import pyximport
import xarray as xr
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from ..config import PYX_DATA_DIR


class SINDySurrogate:
    """
    単純なSINDyによるシングルコンパートメント・ニューロンサロゲーター
    """

    def __init__(self, feature_lib, optimizer, params):
        self.feature_lib = feature_lib
        self.optimizer = optimizer
        self.params = params

    def fit(self, train: np.ndarray, u: np.ndarray, t: np.ndarray):
        self.init = train[0]

        self.sindy = ps.SINDy(
            feature_library=self.feature_lib,
            optimizer=self.optimizer,
            # feature_names=feature_names,
        )
        try:
            self.sindy.fit(
                train,
                u=u,
                t=t,
            )
        except ValueError as e:
            raise ValueError(
                f"SINDyモデルへのフィッティングで、ValueErrorが発生: {e}"
            ) from e
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"LinAlgError:SINDyモデルへのフィッテングの際、LinAlgErrorが発生しました: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"SINDyモデルへのフィッティングの際、予期せぬエラーが発生しました: {e}"
            ) from e

        context = {
            "derivative_x0": self.sindy.equations(precision=50)[0],
            "derivative_x1": self.sindy.equations(precision=50)[1],
            "G_LEAK": self.params.G_LEAK,
            "E_LEAK": self.params.E_LEAK,
            "C": self.params.C,
            "G_12": self.params.G_12,
            "G_23": self.params.G_23,
            "DT": self.params.DT,
        }
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        )
        template = env.get_template("predict_cy.pyx.j2")
        output_context = template.render(context)

        self.module_name = f"predict_cy_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        with open(PYX_DATA_DIR / f"{self.module_name}.pyx", "w", encoding="utf-8") as f:
            f.write(output_context)

    def predict(self, init, dt, iter, u, mode=None):
        try:
            pyximport.install(setup_args={"include_dirs": np.get_include()})
            sys.path.append(PYX_DATA_DIR)
            predict_cy = importlib.import_module(self.module_name)

        except Exception as e:
            logger.critical("MODULE IMPORT ERROR HAPPENED")
            raise ImportError("Import Error") from e
        if hasattr(init, "to_numpy"):
            init = init.to_numpy()
        if hasattr(u, "to_numpy"):
            u = u.to_numpy()
        # ensure they are numpy arrays
        init = np.asarray(init)
        if u is not None:
            u = np.asarray(u)

        if mode == "ThreeComp":
            logger.info("ThreeCompモードで予測を実行します")
            init = np.array([init[0], init[1], -65, -65])  # v,隠れ変数,v_pre,v_post
            return xr.Dataset(
                {
                    "vars": (
                        ("time", "features"),
                        predict_cy.predict_cython_threecomp(init, dt, iter, u),
                    ),
                    "I_ext": (("time"), u),
                },
                coords={
                    "time": np.arange(0, iter * dt, dt),
                    "features": ["V", "latent1", "V_pre", "V_post"],
                },
            )
        elif mode == "SingleComp":
            logger.info("SingleCompモードで予測を実行します")
            return xr.Dataset(
                {
                    "vars": (
                        ("time", "features"),
                        predict_cy.predict_cython(init, dt, iter, u),
                    ),
                    "I_ext": (("time"), u),
                },
                coords={
                    "time": np.arange(0, iter * dt, dt),
                    "features": ["V", "latent1"],
                },
            )
        else:
            raise ValueError(f"未知のmodeが指定されました: {mode}")
