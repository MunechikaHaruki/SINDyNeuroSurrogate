# mypy: ignore-errors

import numpy as np
from loguru import logger
from numba import njit

from ..utils.base_hh import compute_theta
from ..utils.data_processing import create_xr
from . import instantiate_OmegaConf_params


@njit
def simulate_sindy(init, u, xi_matrix, dt):
    """
    xi_matrix: SINDyで得られた model.coefficients() (shape: [n_vars, n_features])
    """
    n_steps = len(u)
    n_vars = len(init)
    x_history = np.empty((n_steps, n_vars))

    curr_x = init.copy()
    for t in range(n_steps):
        # x0=V, x1=m, u0=I_ext と仮定
        # 特徴量ベクトルを生成
        theta = compute_theta(curr_x[0], curr_x[1], u[t])

        # 微係数の計算: dx/dt = Xi @ Theta
        # numpy.dot は Numba 内で高度に最適化される
        dot_product = xi_matrix @ theta

        # 更新（オイラー法）
        curr_x += dot_product * dt
        x_history[t] = curr_x

    return x_history


@njit
def simulate_three_comp_numba(init, u, xi_matrix, dt, params):
    """
    init: [v_soma, latent, v_pre, v_post]
    xi_matrix: SINDyの係数行列
    """
    n_steps = len(u)
    n_vars = init.shape[0]
    x = np.zeros((n_steps, n_vars))
    x[0] = init

    for nt in range(n_steps - 1):
        # 状態変数の展開
        v_soma = x[nt, 0]
        latent = x[nt, 1]
        v_pre = x[nt, 2]
        v_post = x[nt, 3]

        # 1. コンパートメント間の電流計算
        I_pre = params.G_12 * (v_pre - v_soma)
        I_post = params.G_23 * (v_soma - v_post)

        # 2. SINDyモデルによる微分値の計算 (dx = Xi @ Theta)
        theta = compute_theta(v_soma, latent, I_pre - I_post)

        # Euler法による更新
        x[nt + 1, 0] = v_soma + dt * xi_matrix[0] @ theta
        x[nt + 1, 1] = latent + dt * xi_matrix[1] @ theta
        x[nt + 1, 2] = (
            v_pre
            + dt
            * (-params.hh.G_LEAK * (v_pre - params.hh.E_LEAK) - I_pre + u[nt])
            / params.hh.C
        )
        x[nt + 1, 3] = (
            v_post
            + dt
            * (-params.hh.G_LEAK * (v_post - params.hh.E_LEAK) + I_post)
            / params.hh.C
        )

    return x


def predict(init, dt, iter, u, sindy, data_type=None, params_dict=None):
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

        params = instantiate_OmegaConf_params(params_dict, data_type=data_type)

        var = simulate_three_comp_numba(
            init=init,
            u=u,
            xi_matrix=sindy.coefficients(),
            params=params,
            dt=dt,
        )
        features = ["V", "latent1", "V_pre", "V_post"]
    elif data_type == "hh":
        logger.info("hhのサロゲートモデルをテストします")

        var = simulate_sindy(init, u, sindy.coefficients(), dt)
        features = ["V", "latent1"]
    else:
        raise ValueError(f"未知のmodeが指定されました: {data_type}")
    time = np.arange(0, iter * dt, dt)
    return create_xr(var, time, u, features)
