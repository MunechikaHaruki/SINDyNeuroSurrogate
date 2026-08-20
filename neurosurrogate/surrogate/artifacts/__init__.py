"""学習 run 1 本が**自己記述できる**成果物の集合。

置換シミュを回さずに描けるものだけ = run をロードしただけで出る図 (中身は
`model.py` = 静的な構造と係数 / `train.py` = 学習データ)。何を出すかはここが持ち、
合流点 (`artifact.bundle`) はどの段へ書くかだけを決める。再 export はしない
(個々の Artifact は submodule から直接引く)。
"""

from __future__ import annotations

from ...artifact.model import Artifacts
from ..model import Surrogate
from .model import closure_artifact, neuron_graph_artifact, preprocessor_artifact
from .train import (
    train_manifold_artifact,
    train_preprocessed_artifact,
    train_raw_artifact,
    train_recon_artifact,
    train_v_coverage_artifact,
)


def surrogate_artifacts(
    surrogate: Surrogate, view_comps: tuple[str, ...] = ()
) -> Artifacts:
    """学習 run 1 本が自己記述できる成果物をまとめる。

    表現の型に対応する図が無ければ `closure_artifact` / `preprocessor_artifact` は
    None を返すので、その 2 つだけ落として並べる。
    """
    net = surrogate.spec.dataset.net
    comps = [net.name_to_idx(comp) for comp in view_comps] or None
    return Artifacts(
        (
            *(
                artifact
                for artifact in (
                    closure_artifact(surrogate.closure),
                    preprocessor_artifact(surrogate.preprocessor),
                )
                if artifact is not None
            ),
            neuron_graph_artifact(
                net,
                surrogate.spec.dataset.stim,
                {n.name for n in net.nodes if surrogate.spec.in_train_domain(n)},
            ),
            train_raw_artifact(surrogate, comps),
            train_preprocessed_artifact(surrogate, comps),
            train_recon_artifact(surrogate, comps),
            train_v_coverage_artifact(surrogate, comps),
            train_manifold_artifact(surrogate, comps),
        )
    )
