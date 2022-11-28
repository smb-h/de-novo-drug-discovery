import json
import os
from glob import glob

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class Trainer(object):
    def __init__(self, model, train_data, validation_data):
        self.model = model.model
        self.model_name = model.name
        self.config = model.config
        self.x_train = train_data[0]
        self.y_train = train_data[1]
        self.x_validation = validation_data[0]
        self.y_validation = validation_data[1]
        self.callbacks = []
        self.logs_path = self.config.get("logs_path") + f"/{self.model_name}"
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.checkpoint_path = self.config.get("checkpoint_path") + f"{self.model_name}"
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.checkpoint_path,
                    "{epoch}-{val_loss:.2f}.hdf5",
                ),
                monitor=self.config.get("checkpoint_monitor"),
                mode=self.config.get("checkpoint_mode"),
                save_best_only=self.config.get("checkpoint_save_best_only"),
                save_weights_only=self.config.get("checkpoint_save_weights_only"),
                verbose=self.config.get("checkpoint_verbose"),
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.logs_path,
                write_graph=self.config.get("tensorboard_write_graph"),
            )
        )

    def train(self):
        history = self.model.fit(
            {
                "Input_Ex1": self.x_train,
                "polarizer": self.x_train,
                "Input_EX3": self.x_train,
            },
            self.y_train,
            epochs=self.config.get("num_epochs"),
            verbose=self.config.get("verbose_training"),
            validation_data=(
                {
                    "Input_Ex1": self.x_validation,
                    "polarizer": self.x_validation,
                    "Input_EX3": self.x_validation,
                },
                self.y_validation,
            ),
            use_multiprocessing=True,
            shuffle=True,
            callbacks=self.callbacks,
        )

        last_weight_file = glob(
            os.path.join(
                self.checkpoint_path,
                f"{self.config.get('num_epochs')}*.hdf5",
            )
        )[0]
        assert os.path.exists(last_weight_file)

        # add models checkpoint & logs path to config
        self.config[f"model_{self.model_name}"] = {
            "weight_path": last_weight_file,
            "checkpoint_path": self.checkpoint_path,
            "logs_path": self.logs_path,
        }

        with open(os.path.join(self.config.get("experiment_path"), "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
