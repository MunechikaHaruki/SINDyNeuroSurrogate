import hydra
import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from prefect import flow

from scripts.tasks.data import (
    compute_task_seed,
    generate_single_dataset,
    log_single_dataset,
)
from scripts.tasks.eval import log_diff_eval, log_single_eval, single_eval
from scripts.tasks.train import (
    log_single_preprocess_data,
    log_train_model,
    preprocess_single_data,
    train_model,
    train_preprocessor,
)
from scripts.tasks.utils import get_commit_id, get_hydra_overrides, log_plot_to_mlflow


def eval_flow(
    name: str,
    dataset_cfg: DictConfig,
    neuron_cfg: DictConfig,
    base_seed: int,
    preprocessor,
    surrogate_model,
):
    data_type = dataset_cfg.data_type
    task_seed = compute_task_seed(
        dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg, base_seed=base_seed
    )
    ds = generate_single_dataset(
        dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg, task_seed=task_seed
    )
    buf = log_single_dataset(
        data_type=data_type,
        xr_data=ds,
        task_seed=task_seed,
    )
    log_plot_to_mlflow(buf, f"original/{data_type}/{name}.png")

    transformed_ds = preprocess_single_data(
        dataset_name=name, preprocessor=preprocessor, xr_data=ds
    )
    buf = log_single_preprocess_data(
        dataset_key=name,
        dataset_type=data_type,
        xr_data=transformed_ds,
    )
    log_plot_to_mlflow(buf, f"preprocessed/{data_type}/{name}.png")

    eval_result = single_eval(
        data_type=data_type,
        params=neuron_cfg["params"],
        preprocessed_ds=transformed_ds,
        surrogate_model=surrogate_model,
    )
    buf = log_single_eval(data_type=data_type, surrogate_result=eval_result)
    log_plot_to_mlflow(buf, f"surrogate/{data_type}/{name}.png")
    buf = log_diff_eval(
        surrogate_result=eval_result,
        preprocessed_result=transformed_ds,
    )
    log_plot_to_mlflow(buf, f"compare/{data_type}/{name}.png")


def train_flow(cfg):
    # 1. Generate Train Dataset
    train_dataset_cfg = cfg.datasets["train"]
    train_data_type = train_dataset_cfg.data_type
    train_neuron_cfg = cfg.neurons.get(train_data_type)
    if train_neuron_cfg is None:
        logger.error(f"Neuron config for {train_data_type} not found.")
        return
    base_seed = cfg.seed
    task_seed = compute_task_seed(
        dataset_cfg=train_dataset_cfg,
        neuron_cfg=train_neuron_cfg,
        base_seed=base_seed,
    )
    train_ds = generate_single_dataset(
        dataset_cfg=train_dataset_cfg,
        neuron_cfg=train_neuron_cfg,
        task_seed=task_seed,
    )
    buf = log_single_dataset(
        data_type=train_data_type,
        xr_data=train_ds,
        task_seed=task_seed,
    )
    log_plot_to_mlflow(buf, f"train_{train_data_type}.png")

    # 2. Train Preprocessor
    preprocessor = train_preprocessor(train_xr_dataset=train_ds)

    # 3. Train Model
    surrogate_model = train_model(
        train_xr_dataset=train_ds,
        preprocessor=preprocessor,
        surrogate_model_cfg=cfg.models["surrogate"],
        train_dataset_type=train_data_type,
    )
    log_train_model(surrogate=surrogate_model)
    return preprocessor, surrogate_model


@flow
def main_flow(cfg: DictConfig):
    preprocessor, surrogate_model = train_flow(cfg)

    # 4. Process other datasets
    for name, dataset_cfg in cfg.datasets.items():
        data_type = dataset_cfg.data_type
        neuron_cfg = cfg.neurons.get(data_type)
        if neuron_cfg is None:
            logger.warning(f"Neuron config for {data_type} not found.")
            continue
        eval_flow(
            name=name,
            dataset_cfg=dataset_cfg,
            neuron_cfg=neuron_cfg,
            base_seed=cfg.seed,
            preprocessor=preprocessor,
            surrogate_model=surrogate_model,
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    run_name_prefix = get_hydra_overrides()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.experiment_name)
    # Create run to generate ID
    with mlflow.start_run(run_name=run_name_prefix) as run:
        logger.info(f"run_id:{run.info.run_id}")
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        mlflow.set_tag("mlflow.runName", f"{run_name_prefix}_commit-{get_commit_id()}")
        # Prefect flow
        main_flow(cfg)


if __name__ == "__main__":
    main()
