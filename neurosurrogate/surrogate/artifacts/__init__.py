"""表現の種類によって有無が変わる surrogate 自己記述成果物。"""

from ...artifact.model import Artifact
from ..closure.base import Closure
from ..closure.sindy import SINDyBundle
from ..preprocessor.base import Preprocessor
from ..preprocessor.impl.pca import PCAPreprocessor
from .model import pca_scree_artifact, sindy_coef_artifact


def closure_artifact(closure: Closure) -> Artifact | None:
    """閉包項に固有の成果物。対応する図がなければ None。"""
    if isinstance(closure, SINDyBundle):
        return sindy_coef_artifact(closure)
    return None


def preprocessor_artifact(preprocessor: Preprocessor) -> Artifact | None:
    """前処理器に固有の成果物。対応する図がなければ None。"""
    if isinstance(preprocessor, PCAPreprocessor):
        return pca_scree_artifact(preprocessor)
    return None
