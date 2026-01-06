import luigi


class CommonConfig(luigi.Config):
    datasets_cfg_yaml = luigi.Parameter()
    neurons_cfg_yaml = luigi.Parameter()
