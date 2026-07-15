from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pysindy as ps
from sklearn.decomposition import PCA

from ..core.opcost import OpCost
from .autoencoder import AutoEncoderPreprocessor, decoder

if TYPE_CHECKING:
    from collections.abc import Callable

    from .ansatz.roles import Roles
    from .libraries.entry import FeatureLibrary

PREPROCESSOR_CLS: dict[str, type] = {
    "pca": PCA,
    "ae": AutoEncoderPreprocessor,
}

OPTIMIZER_CLS: dict[str, type] = {
    "stlsq": ps.optimizers.STLSQ,
}

# tanh(x) = 1 - 2 / (exp(2x) + 1)
TANH_COST = OpCost(exp=1, div=1, pm=2, mul=1)


def _instantiate(spec: dict, registry: dict[str, type]):
    spec = dict(spec)
    return registry[spec.pop("type")](**spec)


def _reconstruction_mse(preprocessor, train_gate: np.ndarray) -> tuple[float, float]:
    reconstructed = preprocessor.inverse_transform(preprocessor.transform(train_gate))
    mse = float(np.mean((train_gate - reconstructed) ** 2))
    return mse, mse / float(np.var(train_gate))


@dataclass
class PCABundle:
    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    reconstruction_mse: float
    reconstruction_mse_ratio: float

    @classmethod
    def from_preprocessor(cls, preprocessor, train_gate: np.ndarray) -> "PCABundle":
        mse, ratio = _reconstruction_mse(preprocessor, train_gate)
        return cls(
            components=np.asarray(preprocessor.components_),
            mean=np.asarray(preprocessor.mean_),
            explained_variance=np.asarray(preprocessor.explained_variance_),
            explained_variance_ratio=np.asarray(preprocessor.explained_variance_ratio_),
            reconstruction_mse=mse,
            reconstruction_mse_ratio=ratio,
        )

    def metrics(self) -> dict:
        return {
            "pca/explained_variance_ratio": float(self.explained_variance_ratio[0]),
            "pca/explained_variance": float(self.explained_variance[0]),
            "pca/reconstruction_mse": self.reconstruction_mse,
            "pca/reconstruction_mse_ratio": self.reconstruction_mse_ratio,
        }

    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        return state @ jnp.asarray(self.components) + jnp.asarray(self.mean)

    def opcost(self) -> OpCost:
        # decode: gate ごとに latent 数の積 + (latent-1 加算 + mean 1 加算)。
        n_latent, n_gates = self.components.shape
        return OpCost(mul=n_latent * n_gates, pm=n_latent * n_gates)


@dataclass
class AutoEncoderBundle:
    n_components: int
    epochs: int
    lr: float
    reconstruction_mse: float
    reconstruction_mse_ratio: float
    dec_params: dict[str, np.ndarray]
    x_mean: np.ndarray
    x_std: np.ndarray

    @classmethod
    def from_preprocessor(
        cls, preprocessor: AutoEncoderPreprocessor, train_gate: np.ndarray
    ) -> "AutoEncoderBundle":
        assert preprocessor._params is not None
        assert preprocessor._mean is not None
        assert preprocessor._std is not None
        mse, ratio = _reconstruction_mse(preprocessor, train_gate)
        return cls(
            n_components=preprocessor.n_components,
            epochs=preprocessor.epochs,
            lr=preprocessor.lr,
            reconstruction_mse=mse,
            reconstruction_mse_ratio=ratio,
            dec_params={
                k: np.asarray(v) for k, v in preprocessor._params["dec"].items()
            },
            x_mean=np.asarray(preprocessor._mean),
            x_std=np.asarray(preprocessor._std),
        )

    def metrics(self) -> dict:
        return {
            "ae/reconstruction_mse": self.reconstruction_mse,
            "ae/reconstruction_mse_ratio": self.reconstruction_mse_ratio,
        }

    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        jax_params = {k: jnp.asarray(v) for k, v in self.dec_params.items()}
        x_hat = decoder(jax_params, state)
        return jnp.asarray(x_hat * jnp.asarray(self.x_std) + jnp.asarray(self.x_mean))

    def opcost(self) -> OpCost:
        n_latent, hidden = self.dec_params["W1"].shape
        n_gates = int(self.dec_params["W2"].shape[1])
        return (
            OpCost(mul=n_latent * hidden, pm=n_latent * hidden)  # z @ W1 + b1
            + TANH_COST * int(hidden)
            + OpCost(mul=hidden * n_gates, pm=hidden * n_gates)  # h @ W2 + b2
            + OpCost(mul=n_gates, pm=n_gates)  # 標準化の逆変換 (* std + mean)
        )


def _build_bundle(preprocessor, train_gate: np.ndarray):
    if isinstance(preprocessor, PCA):
        return PCABundle.from_preprocessor(preprocessor, train_gate)
    if isinstance(preprocessor, AutoEncoderPreprocessor):
        return AutoEncoderBundle.from_preprocessor(preprocessor, train_gate)
    return None


@dataclass
class PreprocessorBundle:
    preprocessor: Any
    bundle: PCABundle | AutoEncoderBundle | None
    gate_inits: list

    @classmethod
    def from_spec(cls, spec: dict, train_gate: np.ndarray) -> "PreprocessorBundle":
        preprocessor = _instantiate(spec, PREPROCESSOR_CLS)
        preprocessor.fit(train_gate)
        return cls(
            preprocessor=preprocessor,
            bundle=_build_bundle(preprocessor, train_gate),
            gate_inits=preprocessor.transform(train_gate)[0].tolist(),
        )

    def metrics(self) -> dict:
        return self.bundle.metrics() if self.bundle is not None else {}

    def opcost(self) -> OpCost:
        """decode を実行時に呼ぶ ansatz (hybrid) 用の decode コスト。"""
        if self.bundle is None:
            raise ValueError(
                f"opcost 未対応 preprocessor: {type(self.preprocessor).__name__}"
            )
        return self.bundle.opcost()


@dataclass
class SINDyBundle:
    xi: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    input_names: list[str]
    equations: str
    library_specs: list[dict]
    roles: "Roles"

    @cached_property
    def feature_library(self) -> "FeatureLibrary":
        """役割束縛済み FeatureLibrary (compute_theta/opcost 共用)。lambdify 関数は
        pickle 不能 → field でなく cache 化し、__getstate__ で保存対象から除外する。"""
        from .libraries.entry import FeatureLibrary

        return FeatureLibrary.build(self.library_specs, self.roles)

    def __getstate__(self) -> dict:
        # feature_library の cache (pickle 不能な lambdify 関数) を落として保存。
        # load 後は library_specs+roles から lazy 再構築される。
        return {k: v for k, v in self.__dict__.items() if k != "feature_library"}

    @classmethod
    def from_sindy(
        cls,
        library_specs: list[dict],
        optimizer_spec: dict,
        x: np.ndarray,
        u: np.ndarray,
        t: np.ndarray,
        target_names: list[str],
        input_names: list[str],
        roles: "Roles",
    ) -> "SINDyBundle":
        bundle = cls(
            xi=np.empty(0),
            feature_names=[],
            target_names=target_names,
            input_names=input_names,
            equations="",
            library_specs=library_specs,
            roles=roles,
        )
        sindy = ps.SINDy(
            feature_library=bundle.feature_library.library,
            optimizer=_instantiate(optimizer_spec, OPTIMIZER_CLS),
        )
        sindy.fit(x, u=u, t=t, feature_names=target_names + input_names)
        bundle.xi = sindy.coefficients()
        bundle.feature_names = sindy.get_feature_names()
        bundle.equations = "\n".join(sindy.equations(precision=3))
        return bundle

    def xi_metrics(self) -> dict[str, float]:
        nnz = int((self.xi != 0).sum())
        return {"nnz": nnz, "nnz_ratio": nnz / self.xi.size}

    def compute_theta(self) -> "Callable":
        subs = self.feature_library.sub_libraries

        def compute_theta(*inputs):
            values = []
            for sub in subs:
                bound = [inputs[i] for i in sub.inputs]
                for entry in sub.entries:
                    values.append(entry.func(*bound))
            return jnp.array(values, dtype=jnp.float64)

        return compute_theta

    def opcost(self) -> OpCost:
        cost_map = self.feature_library.to_base_cost(
            self.target_names + self.input_names
        )
        active_mask = np.any(self.xi != 0, axis=0)
        active_features = [
            f for i, f in enumerate(self.feature_names) if active_mask[i]
        ]
        nnz = np.count_nonzero(self.xi).item()
        cost = OpCost(mul=nnz, pm=max(0, nnz - int(self.xi.shape[0])))
        for feature in active_features:
            if feature not in cost_map:
                raise ValueError(f"Found Unknown base func: '{feature}'")
            cost = cost + cost_map[feature]
        return cost
