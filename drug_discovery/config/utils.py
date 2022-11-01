import json
import os
import time


def get_config_from_json(config_path):
    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)
    return config_dict


def process_config(config_path):
    config = get_config_from_json(config_path)
    config["config_path"] = config_path
    config["experiment_dir"] = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["experiment_name"],
    )
    config["tensorboard_log_dir"] = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["experiment_name"],
        "logs/",
    )
    config["checkpoint_dir"] = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["experiment_name"],
        "checkpoints/",
    )
    return config
