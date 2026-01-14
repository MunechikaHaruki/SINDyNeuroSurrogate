# mypy: ignore-errors

import numpy as np
from loguru import logger

from ..utils.data_processing import create_xr


class SINDySurrogate:
    """
    単純なSINDyによるシングルコンパートメント・ニューロンサロゲーター
    """

    def __init__(self, sindy, params):
        self.sindy = sindy
        self.params = params

    def predict(self, init, dt, iter, u, data_type=None):
        if hasattr(init, "to_numpy"):
            init = init.to_numpy()
        if hasattr(u, "to_numpy"):
            u = u.to_numpy()
        # ensure they are numpy arrays
        init = np.asarray(init)
        if u is not None:
            u = np.asarray(u)

        if data_type == "hh3":
            logger.info("hh3のサロゲートモデルをテストします")
            init = np.array([init[0], init[1], -65, -65])  # v,隠れ変数,v_pre,v_post
            from .surrogate_numba import simulate_three_comp_numba

            var = simulate_three_comp_numba(
                init,
                u,
                self.sindy.coefficients(),
                dt,
                G_12=self.params["G_12"],
                G_23=self.params["G_23"],
                G_LEAK=self.params["G_LEAK"],
                E_LEAK=self.params["E_LEAK"],
                C=self.params["C"],
            )
            features = ["V", "latent1", "V_pre", "V_post"]
        elif data_type == "hh":
            logger.info("hhのサロゲートモデルをテストします")
            from .surrogate_numba import simulate_sindy

            var = simulate_sindy(init, u, self.sindy.coefficients(), dt)
            features = ["V", "latent1"]
        else:
            raise ValueError(f"未知のmodeが指定されました: {data_type}")
        time = np.arange(0, iter * dt, dt)
        return create_xr(var, time, u, features)
