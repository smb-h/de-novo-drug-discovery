import json
import logging
import os
import sys
import time

from drug_discovery.config.settings import Settings

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


def process_config(config_path, mode):
    config = get_config_from_json(config_path)
    if mode == "r":
        return config
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
    if mode == "w":
        create_dirs([config["experiment_path"], config["logs_path"], config["checkpoint_path"]])
    return config


def get_logger(logger_name, logs_path, level=logging.DEBUG):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # file handler
    # fh = logging.FileHandler(logs_path + f"/{logger_name}.log")
    fh = logging.FileHandler(logs_path + f"/reports.log")
    fh.setLevel(level)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
