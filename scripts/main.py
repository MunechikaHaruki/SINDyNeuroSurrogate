import hydra
import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from prefect import flow

from scripts.tasks.data import generate_single_dataset, log_single_dataset
from scripts.tasks.train import (
    train_preprocessor,
    train_model,
    log_train_model,
    preprocess_single_data,
    log_single_preprocess_data,
)
from scripts.tasks.eval import single_eval, log_single_eval
from scripts.tasks.utils import get_commit_id


@flow
def main_flow(cfg: DictConfig, run_id: str, run_name_prefix: str):
    # Log config
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        commit_id = get_commit_id()
        run_name = f"{run_name_prefix}_commit-{commit_id}"
        mlflow.set_tag("mlflow.runName", run_name)

    # 1. Generate Datasets
    generated_datasets = {}

    for dataset_name, dataset_cfg in cfg.datasets.items():
        data_type = dataset_cfg.data_type
        # Access neuron config safely
        neuron_cfg = cfg.neurons.get(data_type)
        if neuron_cfg is None:
            # Fallback or error? Original code assumed it exists.
            logger.warning(f"Neuron config for {data_type} not found.")
            continue

        ds = generate_single_dataset(
            dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg, seed=cfg.seed
        )
        generated_datasets[dataset_name] = ds

        log_single_dataset(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            run_id=run_id,
            xr_data=ds,
        )

    # 2. Train Preprocessor (using "train" dataset)
    if "train" not in generated_datasets:
        logger.error("Train dataset not found in generated datasets.")
        return

    train_ds = generated_datasets["train"]
    preprocessor = train_preprocessor(train_xr_dataset=train_ds)

    # 3. Train Model
    surrogate_model_cfg = cfg.models["surrogate"]
    train_dataset_cfg = cfg.datasets["train"]
    train_dataset_type = train_dataset_cfg["data_type"]

    surrogate_model = train_model(
        train_xr_dataset=train_ds,
        preprocessor=preprocessor,
        surrogate_model_cfg=surrogate_model_cfg,
        train_dataset_type=train_dataset_type,
    )

    log_train_model(surrogate=surrogate_model, run_id=run_id)

    # 4. Preprocess all datasets
    preprocessed_datasets = {}
    for dataset_name, ds in generated_datasets.items():
        transformed_ds = preprocess_single_data(
            dataset_name=dataset_name, preprocessor=preprocessor, xr_data=ds
        )
        preprocessed_datasets[dataset_name] = transformed_ds

        log_single_preprocess_data(
            dataset_key=dataset_name,
            dataset_type=cfg.datasets[dataset_name]["data_type"],
            run_id=run_id,
            xr_data=transformed_ds,
        )

    # 5. Evaluate
    for dataset_name, transformed_ds in preprocessed_datasets.items():
        dataset_cfg = cfg.datasets[dataset_name]
        data_type = dataset_cfg.data_type
        neuron_cfg = cfg.neurons.get(data_type)

        prediction = single_eval(
            dataset_key=dataset_name,
            dataset_cfg=dataset_cfg,
            neuron_cfg=neuron_cfg,
            preprocessed_ds=transformed_ds,
            surrogate_model=surrogate_model,
        )

        log_single_eval(
            dataset_key=dataset_name,
            dataset_cfg=dataset_cfg,
            run_id=run_id,
            surrogate_result=prediction,
            preprocessed_result=transformed_ds,
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # If tqdm is installed, configure loguru with tqdm.write
    try:
        from tqdm import tqdm

        logger.remove(0)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:
        pass

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