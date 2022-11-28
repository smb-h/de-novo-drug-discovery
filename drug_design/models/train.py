from copy import copy

from drug_design.config.settings import Settings
from drug_design.data.data_loader_molinf import DataLoader as DataLoader_molinf
from drug_design.models.model_bpmoe_c import Model as Model_bpmoe_c
from drug_design.models.model_bpmoe_m import Model as Model_bpmoe_m
from drug_design.models.model_bpmoe_s import Model as Model_bpmoe_s
from drug_design.models.model_pmoe_c import Model as Model_pmoe_c
from drug_design.models.model_pmoe_m import Model as Model_pmoe_m
from drug_design.models.model_pmoe_s import Model as Model_pmoe_s
from drug_design.models.predict import Predictor
from drug_design.models.trainer import Trainer
from drug_design.utils.utils import get_logger, process_config

settings = Settings()


def main():
    config = process_config(settings.CONFIG_PATH)
    logger = get_logger(config["experiment_name"], config["logs_path"])
    logger.info("Start training...")

    x_train = DataLoader_molinf(config, data_type="train")
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
        Model_bpmoe_m,
        Model_bpmoe_s,
        Model_pmoe_c,
        Model_pmoe_m,
        Model_pmoe_s,
    ]

    for model in models:
        model = model(config, session="train")
        trainer = Trainer(model, [x_train, y_train], [x_validation, y_validation])
        trainer.train()
        predictor = Predictor(config, model.name, trainer.model, [x_test, y_test], plot=True)
        predictor.predict()


if __name__ == "__main__":
    main()
