from copy import copy

from dd.config.settings import Settings
from dd.data.data_loader_molinf import DataLoader as DataLoader_molinf
from dd.utils.utils import get_logger, process_config

from .model_bpmoe_c import Model as Model_bpmoe_c
from .model_bpmoe_m import Model as Model_bpmoe_m
from .model_bpmoe_s import Model as Model_bpmoe_s
from .model_pmoe_c import Model as Model_pmoe_c
from .model_pmoe_m import Model as Model_pmoe_m
from .model_pmoe_s import Model as Model_pmoe_s
from .predictor import Predictor

settings = Settings()


def main():
    config = process_config("./reports/2022-12-05/test/config.json", "r")
    logger = get_logger(config["experiment_name"], config["experiment_path"])

    x_train = DataLoader_molinf(config, data_type="train", logger=logger)
    x_test = copy(x_train)
    x_test.data_type = "test"
    x_test, y_test = x_test.__getitem__()
    x_train, y_train = x_train.__getitem__()

    config["input_shape"] = x_test.shape

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
        model = model(config, session="test", logger=logger)
        if model.name in active_models:
            logger.info(f"\n")
            logger.info(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            model.load(
                checkpoint_path=config.get(f"model_{model.name}").get("best_weight_path"),
                model=model.model,
            )
            predictor = Predictor(
                config,
                model.name,
                model.model,
                x_train,
                [x_test, y_test],
                plot=True,
                logger=logger,
            )
            predictor.predict()


if __name__ == "__main__":
    main()
