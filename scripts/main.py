import hydra
import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

from scripts.tasks.data import (
    compute_task_seed,
    generate_single_dataset,
    log_plot_to_mlflow,
    log_single_dataset,
)
from scripts.tasks.eval import log_single_eval, single_eval
from scripts.tasks.train import (
    log_single_preprocess_data,
    log_train_model,
    preprocess_single_data,
    train_model,
    train_preprocessor,
)
from scripts.tasks.utils import get_commit_id


@flow(task_runner=ConcurrentTaskRunner())
def main_flow(cfg: DictConfig, run_id: str, run_name_prefix: str):
    with mlflow.start_run(run_id=run_id):
        # Log config
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        commit_id = get_commit_id()
        run_name = f"{run_name_prefix}_commit-{commit_id}"
        mlflow.set_tag("mlflow.runName", run_name)

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
            log_single_preprocess_data(
                dataset_key=name,
                dataset_type=cfg.datasets[name]["data_type"],
                xr_data=transformed_ds,
            )

        # 5. Evaluate
        eval_futures = {}
        for name, transformed_ds in preprocessed_datasets.items():
            dataset_cfg = cfg.datasets[name]
            neuron_cfg = cfg.neurons.get(dataset_cfg.data_type)
            future = single_eval.submit(
                dataset_key=name,
                dataset_cfg=dataset_cfg,
                neuron_cfg=neuron_cfg,
                preprocessed_ds=transformed_ds,
                surrogate_model=surrogate_model,
            )
            eval_futures[name] = future

        for name, future in eval_futures.items():
            prediction = future.result()
            log_single_eval(
                dataset_key=name,
                dataset_cfg=cfg.datasets[name],
                surrogate_result=prediction,
                preprocessed_result=preprocessed_datasets[name],
            )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    try:
        run_name_prefix = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        run_name_prefix = "default_run"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.experiment_name)

    # Create run to generate ID
    with mlflow.start_run(run_name=run_name_prefix) as run:
        run_id = run.info.run_id
        logger.info(f"run_id:{run_id}")

    main_flow(cfg, run_id, run_name_prefix)


if __name__ == "__main__":
    main()
