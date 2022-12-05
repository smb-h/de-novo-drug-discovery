import json
import os
from glob import glob

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class FineTuner(object):
    def __init__(self, model, train_data, logger):
        self.model = model.model
        self.model_name = model.name
        self.config = model.config
        self.x_train = train_data[0]
        self.y_train = train_data[1]
        self.logger = logger

    def fine_tune(self):
        self.logger.info(f"Fine tunning model {self.model_name}...")
        history = self.model.fit(
            {
                "Input_Ex1": self.x_train,
                "polarizer": self.x_train,
                "Input_EX3": self.x_train,
            },
            self.y_train,
            epochs=self.config.get("num_epochs"),
            verbose=self.config.get("verbose_training"),
            use_multiprocessing=True,
            shuffle=True,
        )

        return history
