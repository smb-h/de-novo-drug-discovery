from copy import copy

from drug_design.config.settings import Settings
from drug_design.data.data_loader_molinf import DataLoader as DataLoader_molinf
from drug_design.utils.utils import get_logger, process_config

from .fine_tuner import FineTuner
from .model_bpmoe_c import Model as Model_bpmoe_c
from .model_bpmoe_m import Model as Model_bpmoe_m
from .model_bpmoe_s import Model as Model_bpmoe_s
from .model_pmoe_c import Model as Model_pmoe_c
from .model_pmoe_m import Model as Model_pmoe_m
from .model_pmoe_s import Model as Model_pmoe_s

settings = Settings()


def main():
    config = process_config("./reports/2022-12-05/test/config.json", "r")
    logger = get_logger(config["experiment_name"], config["experiment_path"])

    x_train = DataLoader_molinf(config, data_type="fine_tune", logger=logger)
    x_train, y_train = x_train.__getitem__()

    config["input_shape"] = x_train.shape

    models = [
        Model_bpmoe_c,
        Model_bpmoe_m,
        Model_bpmoe_s,
        Model_pmoe_c,
        Model_pmoe_m,
        Model_pmoe_s,
    ]

    active_models = [key.split("model_")[1] for key in config.keys() if key.startswith("model_")]

    for model in models:
        model = model(config, session="fine_tune", logger=logger)
        if model.name in active_models:
            logger.info(f"\n")
            logger.info(f"************************************")
            model.load(
                checkpoint_path=config.get(f"model_{model.name}").get("best_weight_path"),
                model=model.model,
            )
            fine_tuner_obj = FineTuner(model, [x_train, y_train], logger)
            fine_tuner_obj.fine_tune()


if __name__ == "__main__":
    main()
