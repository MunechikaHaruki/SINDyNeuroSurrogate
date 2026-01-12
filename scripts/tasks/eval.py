import gokart
import luigi
import matplotlib.pyplot as plt
import mlflow
from loguru import logger

from neurosurrogate import PLOTTER_REGISTRY
from neurosurrogate.config import (
    DATA_DIR,
)
from neurosurrogate.plots import plot_diff
from neurosurrogate.utils.data_processing import _get_control_input

from .train import PreProcessSingleDataTask, TrainModelTask
from .utils import CommonConfig


class SingleEvalTask(gokart.TaskOnKart):
    dataset_key = luigi.Parameter()
    dataset_cfg = luigi.DictParameter()
    neuron_cfg = luigi.DictParameter(default=CommonConfig().neurons_dict["hh3"])
    eval_cfg = luigi.DictParameter(default=CommonConfig().eval_cfg)

    def requires(self):
        return {
            "preprocess_single_task": PreProcessSingleDataTask(
                dataset_name=self.dataset_key
            ),
            "trainmodel_task": TrainModelTask(),
        }

    def run(self):
        loaded_data = self.load()
        # PreProcessSingleDataTask returns the dataset directly
        ds = loaded_data["preprocess_single_task"]
        logger.info(f"{ds} started to process")

        data_type = self.dataset_cfg["data_type"]
        u = _get_control_input(ds, data_type=data_type)
        if data_type == "hh":
            mode = "SingleComp"
            if self.eval_cfg["onlyThreeComp"] is True:
                self.dump(None)
                return
        elif data_type == "hh3":
            mode = "ThreeComp"
            if self.eval_cfg["direct"] is True:
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
        # TrainModelTask returns the surrogate model directly
        prediction = loaded_data["trainmodel_task"].predict(**input_data)
        logger.info(f"prediction_result:{prediction}")
        if mode == "ThreeComp":
            from neurosurrogate.dataset_utils._base import (
                calc_ThreeComp_internal,
            )

            calc_ThreeComp_internal(
                prediction,
                self.neuron_cfg["params"]["G_12"],
                self.neuron_cfg["params"]["G_23"],
            )

        logger.trace(prediction)
        self.dump(prediction)


class EvalTask(gokart.TaskOnKart):
    """
    CommonConfigからキーを取得し、データセットごとにSingleEvalTaskを実行して集計する。
    """

    def requires(self):
        conf = CommonConfig()
        tasks = {}
        for k, dataset_cfg in conf.datasets_dict.items():
            tasks[k] = SingleEvalTask(
                dataset_key=k,
                dataset_cfg=dataset_cfg,
            )
        return tasks

    def run(self):
        targets = self.input()
        data_dict = {
            k: target.load()
            for k, target in targets.items()
            if target.load() is not None
        }
        self.dump(data_dict)


class LogSingleEvalTask(gokart.TaskOnKart):
    dataset_key = luigi.Parameter()
    dataset_cfg = luigi.DictParameter()
    run_id = luigi.Parameter(default=CommonConfig().run_id)

    def requires(self):
        return {
            "eval_result": SingleEvalTask(
                dataset_key=self.dataset_key,
                dataset_cfg=self.dataset_cfg,
            ),
            "preprocess_result": PreProcessSingleDataTask(
                dataset_name=self.dataset_key
            ),
        }

    def run(self):
        inputs = self.load()
        surrogate_result = inputs["eval_result"]

        if surrogate_result is None:
            self.dump(True)
            return

        preprocessed_result = inputs["preprocess_result"]
        data_type = self.dataset_cfg["data_type"]

        with mlflow.start_run(run_id=self.run_id):
            u = preprocessed_result["I_ext"].to_numpy()
            fig = plot_diff(u, preprocessed_result["vars"], surrogate_result["vars"])
            mlflow.log_figure(fig, f"compare/{data_type}/{self.dataset_key}.png")

            self._debug_show_image(fig)

            plt.close(fig)
            fig = PLOTTER_REGISTRY[data_type](
                surrogate_result,
                surrogate=True,
            )

            mlflow.log_figure(
                fig,
                f"surrogate_result/{data_type}/{self.dataset_key}.png",
            )
            plt.close(fig)

        self.dump(True)

    def _debug_show_image(self, fig):
        import subprocess

        TMP = DATA_DIR / "debug.png"
        fig.savefig(TMP)
        subprocess.run(["wezterm", "imgcat", TMP])


class LogEvalTask(gokart.TaskOnKart):
    run_id = luigi.Parameter(default=CommonConfig().run_id)

    def requires(self):
        conf = CommonConfig()
        return {
            k: LogSingleEvalTask(
                dataset_key=k,
                dataset_cfg=conf.datasets_dict[k],
            )
            for k in conf.datasets_dict.keys()
        }

    def run(self):
        self.dump(True)
