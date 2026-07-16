from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pysindy as ps
import sympy as sp
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
    targets: list[sp.Symbol]
    inputs: list[sp.Symbol]
    library_specs: list[dict]
    roles: "Roles"

    @property
    def columns(self) -> list[sp.Symbol]:
        """SINDy 入力行列の列シンボル (roles の列 index が指す先)。"""
        return self.targets + self.inputs

    @property
    def feature_exprs(self) -> list[sp.Expr]:
        """xi の列に対応する feature 式 (列順 = compute_theta = pysindy の feature
        順。from_sindy が pysindy 名との一致を検証済み)。feature の同一性はこの式で
        表され、コスト (op_cost) も表示 (str/latex) もここから派生する。"""
        return self.feature_library.bound_exprs(self.columns)

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
        targets: list[sp.Symbol],
        inputs: list[sp.Symbol],
        roles: "Roles",
    ) -> "SINDyBundle":
        bundle = cls(
            xi=np.empty(0),
            targets=targets,
            inputs=inputs,
            library_specs=library_specs,
            roles=roles,
        )
        sindy = ps.SINDy(
            feature_library=bundle.feature_library.library,
            optimizer=_instantiate(optimizer_spec, OPTIMIZER_CLS),
        )
        sindy.fit(x, u=u, t=t, feature_names=[str(s) for s in bundle.columns])
        bundle.xi = sindy.coefficients()
        # xi の列は pysindy が並べたもの、feature_exprs は自前展開。両者が同順・同表記
        # であることが opcost/compute_theta/描画すべての前提 → fit 時に照合する。
        if (names := sindy.get_feature_names()) != [
            str(e) for e in bundle.feature_exprs
        ]:
            raise ValueError(f"pysindy の feature 順が展開結果と不一致: {names}")
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
        """ξ の積和コスト + 生き残った feature 式の評価コスト (式木から直接算出)。"""
        from .libraries.entry import op_cost

        nnz = np.count_nonzero(self.xi).item()
        return sum(
            (
                op_cost(expr)
                for expr, active in zip(
                    self.feature_exprs, np.any(self.xi != 0, axis=0), strict=True
                )
                if active
            ),
            OpCost(mul=nnz, pm=max(0, nnz - int(self.xi.shape[0]))),
        )
