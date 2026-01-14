# mypy: ignore-errors

import numpy as np
import pysindy as ps
import xarray as xr
from loguru import logger


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

    def predict(self, init, dt, iter, u, mode=None):
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
            from .simulate_numba import simulate_three_comp_numba

            return xr.Dataset(
                {
                    "vars": (
                        ("time", "features"),
                        simulate_three_comp_numba(
                            init,
                            u,
                            self.sindy.coefficients(),
                            dt,
                            G_12=self.params["G_12"],
                            G_23=self.params["G_23"],
                            G_LEAK=self.params["G_LEAK"],
                            E_LEAK=self.params["E_LEAK"],
                            C=self.params["C"],
                        ),
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
            from .simulate_numba import simulate_sindy

            return xr.Dataset(
                {
                    "vars": (
                        ("time", "features"),
                        simulate_sindy(init, u, self.sindy.coefficients(), dt),
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
