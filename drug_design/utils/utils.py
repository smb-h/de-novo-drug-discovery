import json
import os
import sys
import time

from drug_design.config.settings import Settings

settings = Settings()


def create_dirs(dirs):
    if os.path.exists(dirs[0]):
        print("Experiment path already exists! Deleting it...")
        os.system(f"rm -rf {dirs[0]}")

    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print(f"Creating directories error: {err}")
        sys.exit()


def get_config_from_json(config_path):
    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)
    return config_dict


def process_config(config_path):
    config = get_config_from_json(config_path)
    config["config_path"] = config_path
    config["experiment_path"] = os.path.join(
        settings.REPORTS_PATH,
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["experiment_name"],
    )
    config["logs_path"] = os.path.join(
        settings.REPORTS_PATH,
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["experiment_name"],
        "logs/",
    )
    config["checkpoint_path"] = os.path.join(
        settings.REPORTS_PATH,
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["experiment_name"],
        "checkpoints/",
    )
    # create the experiments dirs
    create_dirs(
        [config.get("experiment_path"), config.get("logs_path"), config.get("checkpoint_path")]
    )
    print("Experiment, logs and checkpoints path are created!")
    return config
