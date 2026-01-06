import luigi


class CommonConfig(luigi.Config):
    datasets_cfg_yaml = luigi.Parameter()
    neurons_cfg_yaml = luigi.Parameter()
    model_cfg_yaml = luigi.Parameter()
    eval_cfg_yaml = luigi.Parameter()
    seed = luigi.IntParameter()
    experiment_name = luigi.Parameter()
