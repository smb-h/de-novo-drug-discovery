import os
from glob import glob

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class Trainer(object):
    def __init__(self, model, train_data_loader, validation_data_loader):
        self.model = model.model
        self.config = model.config
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.get("checkpoint_path"),
                    "%s-{epoch:02d}-{val_loss:.2f}.hdf5" % self.config.get("experiment_path"),
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
                log_dir=self.config.get("logs_path"),
                write_graph=self.config.get("tensorboard_write_graph"),
            )
        )

    def train(self):
        # history = self.model.fit_generator(
        history = self.model.fit(
            {
                "Input_Ex1": self.train_data_loader,
                "polarizer": self.train_data_loader,
                "Input_EX3": self.train_data_loader,
            },
            self.train_data_loader,
            steps_per_epoch=self.train_data_loader.__len__(),
            epochs=self.config.get("num_epochs"),
            verbose=self.config.get("verbose_training"),
            validation_data=(
                {
                    "Input_Ex1": self.validation_data_loader,
                    "polarizer": self.validation_data_loader,
                    "Input_EX3": self.validation_data_loader,
                },
                self.validation_data_loader,
            ),
            validation_steps=self.validation_data_loader.__len__(),
            use_multiprocessing=True,
            shuffle=True,
            callbacks=self.callbacks,
        )

        last_weight_file = glob(
            os.path.join(
                f"{self.config.get('checkpoint_path')}",
                f"{self.config.get('experiment_name')}-{self.config.get('num_epochs'):02}*.hdf5",
            )
        )[0]

        assert os.path.exists(last_weight_file)
        self.config["model_weight_filename"] = last_weight_file

        with open(os.path.join(self.config.get("experiment_path"), "config.json"), "w") as f:
            f.write(self.config.toJSON(indent=4))
