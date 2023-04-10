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
from .model_bpmoe_Mean import Model as Model_bpmoe_Mean
from .model_pmoe_Mean import Model as Model_pmoe_Mean
from .predictor import Predictor
from .trainer import Trainer

settings = Settings()


def main():
    config = process_config(settings.CONFIG_PATH, "w")
    logger = get_logger(config["experiment_name"], config["experiment_path"])
    logger.info("=====" * 6)
    logger.info("Train")

    x_train = DataLoader_molinf(config, data_type="train", logger=logger)
    x_validation = copy(x_train)
    x_validation.data_type = "validation"
    x_test = copy(x_train)
    x_test.data_type = "test"

    x_train, y_train = x_train.__getitem__()
    x_validation, y_validation = x_validation.__getitem__()
    x_test, y_test = x_test.__getitem__()

    config["input_shape"] = x_validation.shape

    models = [
        Model_bpmoe_c,
        Model_pmoe_c,
        Model_bpmoe_Mean,
        Model_pmoe_Mean,
    ]

    for model in models:
        model = model(config, session="train", logger=logger)
        logger.info(f"\n")
        logger.info(f"#####" * 6)
        trainer = Trainer(model, [x_train, y_train], [x_validation, y_validation], "train", logger)
        trainer.train()
        predictor = Predictor(
            config, model.name, trainer.model, x_train, [x_test, y_test], plot=False, logger=logger
        )
        predictor.predict()


if __name__ == "__main__":
    main()
