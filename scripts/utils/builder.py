import importlib
import logging

from neurosurrogate.build_current import PIPE_FUNCS, build_current_pipeline
from neurosurrogate.build_neuron import build_model

logger = logging.getLogger(__name__)


def build_simulator_config(dataset_cfg):

    u = build_current_pipeline(dataset_cfg["current"])
    dt = dataset_cfg["dt"]
    parsed_dict = {"u": u, "dt": dt, "net": dataset_cfg["net"]}
    return parsed_dict


def build_dataset(
    dt=0.01,
    silence_duration=80,
    duration=800,
    model_name="hh",
    pipeline=None,
    current_type=None,
    value=None,
) -> dict:
    """yamlとの境界"""

    if pipeline is None:
        pipeline = PIPE_FUNCS[current_type](value)

    neuron_module = importlib.import_module("neurosurrogate.build_neuron")
    neuron_spec = getattr(neuron_module, model_name)
    return {
        "data_type": model_name,
        "dt": dt,
        "current": {
            # フラットアクセスではなく、case_cfg["current"] のネストを参照する
            "iteration": int(duration / dt),
            "pipeline": pipeline,
            "silence_steps": int(silence_duration / dt),
        },
        "net": build_model(neuron_spec),
    }
