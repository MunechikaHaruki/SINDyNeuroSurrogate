import os

import matplotlib

matplotlib.use("Agg")
import logging
import subprocess

import hydra
import matplotlib.pyplot as plt
import mlflow
from flow import main_flow
from omegaconf import DictConfig, OmegaConf

# Prefectのインポートより前に環境変数を設定する
os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = "neurosurrogate"

logger = logging.getLogger(__name__)


# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


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


def get_hydra_overrides():
    try:
        run_name_prefix = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        run_name_prefix = "OverrideError"
    if run_name_prefix == "":
        run_name_prefix = "noOverride"
    return run_name_prefix


def build_full_datasets(cfg):
    # 1. 既存の datasets (random_hh3, random_hh 等) を辞書として取得
    # resolve=True にすることで、内部の変数参照を解決した状態で取得できます
    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)

    # 共通デフォルト値の抽出
    default_seed = cfg.get("seed", 42)
    default_dt = cfg.get("simulater_dt", 0.01)

    # 内部関数：デフォルト値をセットする
    def apply_defaults(ds_dict):
        ds_dict.setdefault("seed", default_seed)
        ds_dict.setdefault("dt", default_dt)
        return ds_dict

    # 2. HH 用の定電流データを追加
    if "hh_steady_values" in cfg.dataset_profiles:
        for v in cfg.dataset_profiles.hh_steady_values:
            key = f"steady_hh_{v}"
            # すでにYAML側で定義されている場合はスキップ、または上書きを選択
            if key not in datasets:
                datasets[key] = {
                    "data_type": "hh",
                    "current": {
                        "_target_": "neurosurrogate.utils.current_generators.hh_steady",
                        "value": float(v),
                    },
                }

    # 3. HH3 用の定電流データを追加
    if "hh3_steady_values" in cfg.dataset_profiles:
        for v in cfg.dataset_profiles.hh3_steady_values:
            key = f"steady_hh3_{v}"
            if key not in datasets:
                datasets[key] = {
                    "data_type": "hh3",
                    "current": {
                        "_target_": "neurosurrogate.utils.current_generators.hh_steady",
                        "value": float(v),
                    },
                }

    for key in datasets:
        datasets[key] = apply_defaults(datasets[key])

    return datasets


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    OmegaConf.resolve(cfg)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_STYLE_PATH = os.path.join(BASE_DIR, "./conf/style/base.mplstyle")
    plt.style.use(BASE_STYLE_PATH)
    STYLE_PATH = os.path.join(BASE_DIR, f"./conf/style/{cfg.matplotlib_style}.mplstyle")
    plt.style.use(STYLE_PATH)

    run_name_prefix = get_hydra_overrides()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.experiment_name)
    # Create run to generate ID
    with mlflow.start_run(run_name=run_name_prefix) as run:
        logger.info(f"run_id:{run.info.run_id}")
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        mlflow.set_tag(
            "mlflow.runName",
            f"{cfg.selected}_{run_name_prefix}_commit-{get_commit_id()}",
        )
        # Prefect flow
        dataset_cfg = build_full_datasets(cfg)
        main_flow(dataset_cfg)
    logger.info("Script ended")


if __name__ == "__main__":
    main()
