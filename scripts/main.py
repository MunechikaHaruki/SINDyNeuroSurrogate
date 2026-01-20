import hydra
import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

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


@flow(task_runner=ConcurrentTaskRunner())
def main_flow(cfg: DictConfig):
    # 1. Generate Datasets
    generated_datasets = {}
    for name, dataset_cfg in cfg.datasets.items():
        data_type = dataset_cfg.data_type
        neuron_cfg = cfg.neurons.get(data_type)
        if neuron_cfg is None:
            logger.warning(f"Neuron config for {data_type} not found.")
            continue
        base_seed = cfg.seed
        task_seed = compute_task_seed(
            dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg, base_seed=base_seed
        )
        ds = generate_single_dataset(
            dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg, task_seed=task_seed
        )
        generated_datasets[name] = ds

        buf = log_single_dataset(
            data_type=data_type,
            xr_data=ds,
            task_seed=task_seed,
        )
        log_plot_to_mlflow(buf, f"original/{data_type}/{name}.png")

    # 2. Train Preprocessor
    if "train" not in generated_datasets:
        logger.error("Train dataset not found in generated datasets.")
        return
    train_ds = generated_datasets["train"]
    preprocessor = train_preprocessor(train_xr_dataset=train_ds)

    # 3. Train Model
    surrogate_model = train_model(
        train_xr_dataset=train_ds,
        preprocessor=preprocessor,
        surrogate_model_cfg=cfg.models["surrogate"],
        train_dataset_type=cfg.datasets["train"]["data_type"],
    )
    log_train_model(surrogate=surrogate_model)

    # 4. Preprocess all datasets
    preprocess_futures = {}
    for name, ds in generated_datasets.items():
        future = preprocess_single_data.submit(
            dataset_name=name, preprocessor=preprocessor, xr_data=ds
        )
        preprocess_futures[name] = future

    preprocessed_datasets = {}
    for name, future in preprocess_futures.items():
        transformed_ds = future.result()
        preprocessed_datasets[name] = transformed_ds
        dataset_type = cfg.datasets[name]["data_type"]
        buf = log_single_preprocess_data(
            dataset_key=name,
            dataset_type=dataset_type,
            xr_data=transformed_ds,
        )
        log_plot_to_mlflow(buf, f"preprocessed/{dataset_type}/{name}.png")

    # 5. Evaluate
    for name, transformed_ds in preprocessed_datasets.items():
        data_type = cfg.datasets[name].data_type
        eval_result = single_eval(
            data_type=data_type,
            params=cfg.neurons[data_type]["params"],
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
