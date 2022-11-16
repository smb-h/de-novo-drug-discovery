from copy import copy

from drug_design.config.settings import Settings
from drug_design.data.data_loader_molinf import DataLoader as DataLoader_molinf
from drug_design.models.model_1 import Model
from drug_design.models.trainer import Trainer
from drug_design.utils.utils import process_config

settings = Settings()


def main():
    config = process_config(settings.CONFIG_PATH)

    x_train = DataLoader_molinf(config, data_type="train")
    x_validation = copy(x_train)
    x_validation.data_type = "validation"
    x_sampled, y_sampled = x_train.__getitem__(0)
    config["input_shape"] = [x_sampled.shape[1], x_sampled.shape[2]]

    model = Model(config, session="train")
    trainer = Trainer(model, x_train, x_validation)
    trainer.train()


if __name__ == "__main__":
    main()
