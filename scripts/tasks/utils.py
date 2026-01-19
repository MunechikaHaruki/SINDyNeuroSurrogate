import io
import subprocess
from collections.abc import Mapping

import matplotlib.pyplot as plt
import mlflow
from omegaconf import OmegaConf
from PIL import Image


def recursive_to_dict(obj):
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, (list, tuple)):
        return [recursive_to_dict(x) for x in obj]
    if isinstance(obj, Mapping):
        return {k: recursive_to_dict(v) for k, v in obj.items()}
    return obj


def get_commit_id():
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        commit_id = "unknown"
    return commit_id


def log_plot_to_mlflow(img_bytes, artifact_path):
    img = Image.open(io.BytesIO(img_bytes))
    mlflow.log_image(img, artifact_path)


def fig_to_buff(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
