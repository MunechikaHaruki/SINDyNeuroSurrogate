from neurosurrogate.build_current import build_current_pipeline


def build_simulator_config(dataset_cfg):

    u = build_current_pipeline(dataset_cfg["current"])
    dt = dataset_cfg["dt"]
    parsed_dict = {"u": u, "dt": dt, "net": dataset_cfg["net"]}
    return parsed_dict
