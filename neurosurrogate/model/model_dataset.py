from dataclasses import dataclass

from neurosurrogate.builder.build_current import CurrentConfig
from neurosurrogate.model.model_neuron import MCMODELS, NeuronGraph


@dataclass
class DatasetConfig:
    model_name: str
    dt: float
    current: CurrentConfig
    net: NeuronGraph

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dt": self.dt,
            "current": self.current.to_dict(),
            "net": self.net.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            model_name=d["model_name"],
            dt=d["dt"],
            current=CurrentConfig.from_dict(d["current"]),
            net=NeuronGraph.from_dict(d["net"]),
        )

    @classmethod
    def build_dataset(
        cls,
        dt: float,
        silence_duration: float,
        duration: float,
        model_name: str,
        pipeline: dict,
    ) -> "DatasetConfig":
        """yamlとの境界"""
        return DatasetConfig(
            model_name=model_name,
            dt=dt,
            current=CurrentConfig(
                iteration=int(duration / dt),
                silence_steps=int(silence_duration / dt),
                pipeline=pipeline,
            ),
            net=MCMODELS[model_name],
        )
