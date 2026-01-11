import gokart
import luigi
import matplotlib.pyplot as plt
import mlflow
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate import PLOTTER_REGISTRY
from neurosurrogate.config import (
    DATA_DIR,
)
from neurosurrogate.plots import plot_diff
from neurosurrogate.utils.data_processing import _get_control_input

from .train import PreProcessDataTask, TrainModelTask
from .utils import CommonConfig, recursive_to_dict


class SingleEvalTask(gokart.TaskOnKart):
    dataset_key = luigi.Parameter()

    def requires(self):
        return {
            "preprocess_task": PreProcessDataTask(),
            "trainmodel_task": TrainModelTask(),
        }

    def run(self):
        loaded_data = self.load()
        k = self.dataset_key
        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(recursive_to_dict(conf.datasets_dict))
        neurons_cfg = OmegaConf.create(recursive_to_dict(conf.neurons_dict))

        # PreProcessDataTask now returns the dictionary directly
        preprocessed_datasets = loaded_data["preprocess_task"]
        ds = preprocessed_datasets[k]

        data_type = datasets_cfg[k].data_type
        neuron_cfg = neurons_cfg[data_type]
        logger.info(f"{ds} started to process")

        u = _get_control_input(ds, data_type=data_type)
        if data_type == "hh":
            mode = "SingleComp"
            if conf.eval_cfg["onlyThreeComp"] is True:
                logger.info(f"{k} is passed")
                self.dump(None)
                return
        elif data_type == "hh3":
            mode = "ThreeComp"
            if conf.eval_cfg["direct"] is True:
                logger.info("Using direct ThreeComp mode")
                u = _get_control_input(ds, data_type=data_type, direct=True)
                mode = "SingleComp"

        input_data = {
            "init": ds["vars"][0],
            "dt": 0.01,
            "iter": len(ds["time"].to_numpy()),
            "u": u,
            "mode": mode,
        }
        logger.info(f"input:{input_data}")

        logger.critical(f"{k}")
        # TrainModelTask returns the surrogate model directly
        prediction = loaded_data["trainmodel_task"].predict(**input_data)
        logger.info(f"key:{k} prediction_result:{prediction}")
        if mode == "ThreeComp":
            from neurosurrogate.dataset_utils._base import (
                calc_ThreeComp_internal,
            )

            calc_ThreeComp_internal(prediction, neuron_cfg.params)

        logger.trace(prediction)
        self.dump(prediction)


class EvalTask(gokart.TaskOnKart):
    """
    PreProcessDataTaskの結果を受け取り、データセットごとにSingleEvalTaskを実行して集計する。
    """

    # 必要に応じてパラメータ化し、requiresで利用できるようにする
    # model_name = gokart.Parameter()

    def requires(self):
        return PreProcessDataTask()

    def run(self):
        preprocessed_datasets = self.load()

        tasks = {k: SingleEvalTask(dataset_key=k) for k in preprocessed_datasets.keys()}
        yield list(tasks.values())
        data_dict = {k: task.output().load() for k, task in tasks.items()}

        self.dump(data_dict)


class LogEvalTask(gokart.TaskOnKart):
    run_id = luigi.Parameter(default=CommonConfig().run_id)

    def requires(self):
        return {"eval_task": EvalTask(), "preprocess_task": PreProcessDataTask()}

    def run(self):
        loaded_data = self.load()
        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(recursive_to_dict(conf.datasets_dict))
        with mlflow.start_run(run_id=self.run_id):
            # EvalTask dumps {"path_dict": ...}
            for k, surrogate_result in loaded_data["eval_task"].items():
                data_type = datasets_cfg[k].data_type
                # PreProcessDataTask dumps the dict directly
                preprocessed_result = loaded_data["preprocess_task"][k]

                u = preprocessed_result["I_ext"].to_numpy()
                fig = plot_diff(
                    u, preprocessed_result["vars"], surrogate_result["vars"]
                )
                mlflow.log_figure(fig, f"compare/{data_type}/{k}.png")

                self._debug_show_image(fig)

                plt.close(fig)
                fig = PLOTTER_REGISTRY[data_type](
                    surrogate_result,
                    surrogate=True,
                )

                mlflow.log_figure(
                    fig,
                    f"surrogate_result/{data_type}/{k}.png",
                )
                plt.close(fig)

        self.dump(True)

    def _debug_show_image(self, fig):
        import subprocess

        TMP = DATA_DIR / "debug.png"
        fig.savefig(TMP)
        subprocess.run(["wezterm", "imgcat", TMP])
