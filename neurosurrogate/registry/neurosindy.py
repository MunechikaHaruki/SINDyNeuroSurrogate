from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import jax.numpy as jnp
import joblib
import numpy as np

from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.simulator import unified_simulator
from ..metrics.opcost import OpCost
from ..metrics.result_bundle import PreprocessorBundle, SINDyBundle
from .compartments.hh import HH_DV_COST, HHParams

_BUNDLE_FILE = "surrogate.joblib"


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


class NeuroSurrogateBase(ABC):
    _sindy_bundle: SINDyBundle
    _preprocessor_bundle: PreprocessorBundle

    def __init__(self, datasets: dict, train_comp_identifier: str):
        self._dataset = DatasetConfig.build_dataset(**datasets)
        self.train_comp_id: int = self._dataset.net.name_to_idx(train_comp_identifier)
        self._train_xr = unified_simulator(self._dataset)

    @abstractmethod
    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None: ...

    @abstractmethod
    def make_surr_comp(self, name: str, **kwargs) -> Compartment: ...

    @property
    @abstractmethod
    def opcost(self) -> OpCost: ...

    @property
    def sindy_bundle(self) -> SINDyBundle:
        return self._sindy_bundle

    @property
    def preprocessor_bundle(self) -> PreprocessorBundle:
        return self._preprocessor_bundle

    def _set_bundles(
        self,
        sindy_bundle: SINDyBundle,
        preprocessor_bundle: PreprocessorBundle,
    ) -> None:
        self._sindy_bundle = sindy_bundle
        self._preprocessor_bundle = preprocessor_bundle

    @property
    def original_opcost(self) -> OpCost | None:
        return self._dataset.net.nodes[self.train_comp_id].type.opcost

    def metrics(self) -> dict:
        return {
            **self.sindy_bundle.xi_metrics(),
            **self.preprocessor_bundle.metrics(),
            **self.opcost.diff_dict(self.original_opcost),
        }

    def save(self, dir: Path | str) -> None:
        joblib.dump(
            {
                "sindy_bundle": self.sindy_bundle,
                "preprocessor_bundle": self.preprocessor_bundle,
            },
            Path(dir) / _BUNDLE_FILE,
        )

    @classmethod
    def load(cls, dir: Path | str) -> Self:
        self = cls.__new__(cls)
        self._set_bundles(**joblib.load(Path(dir) / _BUNDLE_FILE))
        return self


class SINDyNeuroSurrogate(NeuroSurrogateBase):
    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        train_gate = get_gate_numpy(self._train_xr, self.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(preprocessor, train_gate)
        preprocessed_xr = transform_gate(
            preprocessor_bundle.preprocessor,
            self._train_xr,
            target_comp_id=self.train_comp_id,
        )
        target_names = preprocessed_xr.variable.values.tolist()
        self._set_bundles(
            sindy_bundle=SINDyBundle.from_sindy(
                library_specs=library_specs,
                optimizer_spec=optimizer,
                x=preprocessed_xr["vars"].sel(comp_id=self.train_comp_id).to_numpy(),
                u=preprocessed_xr["I_ext"].to_numpy(),
                t=self._train_xr["time"].to_numpy(),
                target_names=target_names,
                input_names=["u"],
            ),
            preprocessor_bundle=preprocessor_bundle,
        )

    def make_surr_comp(self, name: str, **kwargs) -> Compartment:
        xi = self.sindy_bundle.xi
        compute_theta = self.sindy_bundle.compute_theta()

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
                gate_names=[
                    f"latent{i + 1}"
                    for i in range(len(self.preprocessor_bundle.gate_inits))
                ],
                default_gate_inits=self.preprocessor_bundle.gate_inits,
                v_init=-65,
                opcost=None,
            ),
        )

    @property
    def opcost(self) -> OpCost:
        return self.sindy_bundle.opcost()


class HybridSINDyNeuroSurrogate(NeuroSurrogateBase):
    """
    Hybrid: 物理HH回路方程式 (HHParams 陽) + 学習ゲート dynamics。
      dV/dt   = HH 物理式 (E_LEAK, G_*, E_*, C, gates)
      gates   = state @ pca_components + pca_mean    (線形 decoder)
      d(latent)/dt = f(V, latent) via SINDy
    SINDy 入力順: (latent_1, ..., V) → library_specs は index 0=latent, 末尾=V。
    """

    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        gate_data = get_gate_numpy(self._train_xr, self.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(preprocessor, gate_data)
        latent = preprocessor_bundle.preprocessor.transform(gate_data)
        v = (
            self._train_xr["vars"]
            .sel(gate=False, comp_id=self.train_comp_id)
            .to_numpy()
        )
        latent_names = [f"latent{i + 1}" for i in range(latent.shape[1])]
        self._set_bundles(
            sindy_bundle=SINDyBundle.from_sindy(
                library_specs=library_specs,
                optimizer_spec=optimizer,
                x=latent,
                u=v,
                t=self._train_xr["time"].to_numpy(),
                target_names=latent_names,
                input_names=["V"],
            ),
            preprocessor_bundle=preprocessor_bundle,
        )

    def make_surr_comp(
        self, name: str, params: HHParams | None = None, **kwargs
    ) -> Compartment:
        xi = jnp.asarray(self.sindy_bundle.xi)
        assert self.preprocessor_bundle.bundle is not None
        pca_components = jnp.asarray(self.preprocessor_bundle.bundle.components)
        pca_mean = jnp.asarray(self.preprocessor_bundle.bundle.mean)
        compute_theta = self.sindy_bundle.compute_theta()
        n_latent = len(self.preprocessor_bundle.gate_inits)

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
                default_gate_inits=self.preprocessor_bundle.gate_inits,
                v_init=-65,
                opcost=None,
            ),
            params=params or HHParams(),
        )

    @property
    def opcost(self) -> OpCost:
        return self.sindy_bundle.opcost() + HH_DV_COST


SURR_CLS: dict[str, type[NeuroSurrogateBase]] = {
    "sindy": SINDyNeuroSurrogate,
    "hybrid": HybridSINDyNeuroSurrogate,
}
