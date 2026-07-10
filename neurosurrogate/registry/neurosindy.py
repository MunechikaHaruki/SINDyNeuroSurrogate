import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import jax.numpy as jnp
import joblib
import numpy as np
import pysindy as ps

from ..core.libraries import FeatureLibrary
from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.simulator import unified_simulator
from ..metrics.opcost import OpCost, calc_sindy_opcost
from ..metrics.result_bundle import PCABundle, SINDyBundle
from .compartments.hh import HH_DV_COST, HHParams

_BUNDLE_FILE = "surrogate.joblib"

logger = logging.getLogger(__name__)


def get_gate_numpy(train_xr, target_comp_id):
    return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()


def transform_gate(preprocessor, xr_data, target_comp_id):
    from ..core.coords import StateAccumulator, set_coords

    transformed_gate = preprocessor.transform(get_gate_numpy(xr_data, target_comp_id))
    n_latent = transformed_gate.shape[1]

    return set_coords(
        raw=np.concatenate(
            (
                xr_data["vars"]
                .sel(gate=False, comp_id=target_comp_id)
                .to_numpy()
                .reshape(-1, 1),
                transformed_gate,
            ),
            axis=1,
        ),
        u=xr_data["I_internal"].sel(node_id=target_comp_id).to_numpy(),
        coords=StateAccumulator(
            comp_id=[target_comp_id] * (n_latent + 1),
            variable=["V"] + [f"latent{i + 1}" for i in range(n_latent)],
            gate=[False] + [True] * n_latent,
        ).to_coords(),
        dt=float(xr_data.time[1] - xr_data.time[0]),
    )


def _build_compute_theta(feature_lib: FeatureLibrary):
    """FeatureLibrary の sub_libraries から compute_theta 関数を直接構築。
    LibraryEntry.func を直接呼び出す → exec / target_module 依存なし。"""
    subs = feature_lib.sub_libraries

    def compute_theta(*inputs):
        values = []
        for sub in subs:
            bound = [inputs[i] for i in sub.inputs]
            for entry in sub.entries:
                values.append(entry.func(*bound))
        return jnp.array(values, dtype=jnp.float64)

    return compute_theta


def _instantiate_if_dict(obj):
    if isinstance(obj, dict):
        import hydra

        return hydra.utils.instantiate(obj)
    return obj


class NeuroSurrogateBase(ABC):
    def __init__(self, datasets: dict, train_comp_identifier: str):
        self._dataset = DatasetConfig.build_dataset(**datasets)
        self._target_comp_id: int = self._dataset.net.name_to_idx(train_comp_identifier)
        self.train_comp_id: int = self._target_comp_id
        self._train_xr = unified_simulator(self._dataset)

    @abstractmethod
    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None: ...

    @abstractmethod
    def make_surr_comp(self, name: str, **kwargs) -> Compartment: ...

    @property
    @abstractmethod
    def result(self) -> SINDyBundle: ...

    @property
    @abstractmethod
    def opcost(self) -> OpCost: ...

    def save(self, dir: Path | str) -> None:
        bundle = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        joblib.dump(bundle, Path(dir) / _BUNDLE_FILE)

    @classmethod
    def load(cls, dir: Path | str) -> Self:
        self = cls.__new__(cls)
        self.__dict__.update(joblib.load(Path(dir) / _BUNDLE_FILE))
        return self


class SINDyNeuroSurrogate(NeuroSurrogateBase):
    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        self.preprocessor = _instantiate_if_dict(preprocessor)
        feature_lib = FeatureLibrary.build(library_specs)
        self._sindy = ps.SINDy(
            feature_library=feature_lib.library,
            optimizer=_instantiate_if_dict(optimizer),
        )
        self._train_gate_data = get_gate_numpy(self._train_xr, self._target_comp_id)
        self.preprocessor.fit(self._train_gate_data)
        preprocessed_xr = transform_gate(
            self.preprocessor, self._train_xr, target_comp_id=self._target_comp_id
        )
        self.gate_inits: list = preprocessed_xr["vars"].to_numpy()[0][1:].tolist()
        target_names = preprocessed_xr.variable.values.tolist()
        self._sindy.fit(
            preprocessed_xr["vars"].sel(comp_id=self._target_comp_id).to_numpy(),
            u=preprocessed_xr["I_ext"].to_numpy(),
            t=self._train_xr["time"].to_numpy(),
            feature_names=target_names + ["u"],
        )
        self.sindy_result = SINDyBundle.from_sindy(
            self._sindy, target_names, library_specs
        )
        self.original_opcost: OpCost | None = self._dataset.net.nodes[
            self._target_comp_id
        ].type.opcost

    @property
    def result(self) -> SINDyBundle:
        return self.sindy_result

    def make_surr_comp(self, name: str, **kwargs) -> Compartment:
        xi = self.sindy_result.xi
        compute_theta = _build_compute_theta(
            FeatureLibrary.build(self.sindy_result.library_specs)
        )

        def surr_kernel(params, i_t, v, state):
            theta = compute_theta(v, state[0], i_t)
            return xi[0] @ theta, jnp.stack([xi[1] @ theta])

        return Compartment(
            name=name,
            type=CompartmentType(
                name="surr",
                kernel=surr_kernel,
                param_cls=None,
                default_params=None,
                gate_names=[f"latent{i + 1}" for i in range(len(self.gate_inits))],
                default_gate_inits=self.gate_inits,
                v_init=-65,
                opcost=None,
            ),
        )

    @property
    def opcost(self) -> OpCost:
        cost_map = FeatureLibrary.build(self.sindy_result.library_specs).to_base_cost(
            self.sindy_result.target_names + ["u"]
        )
        return calc_sindy_opcost(self.sindy_result, cost_map)


class HybridSINDyNeuroSurrogate(NeuroSurrogateBase):
    """
    Hybrid: 物理HH回路方程式 (HHParams 陽) + 学習ゲート dynamics。
      dV/dt   = HH 物理式 (E_LEAK, G_*, E_*, C, gates)
      gates   = state @ pca_components + pca_mean    (線形 decoder)
      d(latent)/dt = f(V, latent) via SINDy
    SINDy 入力順: (latent_1, ..., V) → library_specs は index 0=latent, 末尾=V。
    """

    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        self.preprocessor = _instantiate_if_dict(preprocessor)
        feature_lib = FeatureLibrary.build(library_specs)
        self._sindy = ps.SINDy(
            feature_library=feature_lib.library,
            optimizer=_instantiate_if_dict(optimizer),
        )
        gate_data = get_gate_numpy(self._train_xr, self._target_comp_id)
        self.preprocessor.fit(gate_data)
        latent = self.preprocessor.transform(gate_data)
        v = (
            self._train_xr["vars"]
            .sel(gate=False, comp_id=self._target_comp_id)
            .to_numpy()
        )
        n_latent = latent.shape[1]
        latent_names = [f"latent{i + 1}" for i in range(n_latent)]
        self._sindy.fit(
            latent,
            u=v,
            t=self._train_xr["time"].to_numpy(),
            feature_names=latent_names + ["V"],
        )
        self.sindy_result = SINDyBundle.from_sindy(
            self._sindy, latent_names + ["V"], library_specs
        )
        self.pca_bundle = PCABundle.from_preprocessor(self.preprocessor)
        self.gate_inits: list = latent[0].tolist()
        self.original_opcost: OpCost | None = self._dataset.net.nodes[
            self._target_comp_id
        ].type.opcost

    @property
    def result(self) -> SINDyBundle:
        return self.sindy_result

    def make_surr_comp(
        self, name: str, params: HHParams | None = None, **kwargs
    ) -> Compartment:
        xi = jnp.asarray(self.sindy_result.xi)
        pca_components = jnp.asarray(self.pca_bundle.components)
        pca_mean = jnp.asarray(self.pca_bundle.mean)
        compute_theta = _build_compute_theta(
            FeatureLibrary.build(self.sindy_result.library_specs)
        )
        n_latent = len(self.gate_inits)

        def hybrid_kernel(p: HHParams, u_t, v, state):
            gates = state @ pca_components + pca_mean
            m, h, n = gates[0], gates[1], gates[2]
            i_na = p.G_NA * m**3 * h * (v - p.E_NA)
            i_k = p.G_K * n**4 * (v - p.E_K)
            i_leak = p.G_LEAK * (v - p.E_LEAK)
            dv = (-i_leak - i_na - i_k + u_t) / p.C
            theta_inputs = [state[i] for i in range(n_latent)] + [v]
            theta = compute_theta(*theta_inputs)
            dlatent = xi @ theta
            return dv, dlatent

        return Compartment(
            name=name,
            type=CompartmentType(
                name="hybrid_surr",
                kernel=hybrid_kernel,
                param_cls=HHParams,
                default_params=HHParams(),
                gate_names=[f"latent{i + 1}" for i in range(n_latent)],
                default_gate_inits=self.gate_inits,
                v_init=-65,
                opcost=None,
            ),
            params=params if params is not None else HHParams(),
        )

    @property
    def opcost(self) -> OpCost:
        cost_map = FeatureLibrary.build(self.sindy_result.library_specs).to_base_cost(
            self.sindy_result.target_names + ["u"]
        )
        return calc_sindy_opcost(self.sindy_result, cost_map) + HH_DV_COST
