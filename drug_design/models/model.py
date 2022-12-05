from abc import ABC, abstractmethod

from tensorflow.keras.models import model_from_json


# Model
class BaseModel(ABC):
    # init
    def __init__(self, config, session="train", logger=None) -> None:
        self.name = "Model"
        assert session in ["train", "test", "fine_tune"], "One of {train, test, fine_tune}"
        self.config = config
        self.session = session
        self.logger = logger
        self.model = None

        self.logger.info(f"Initializing model with {self.session} session.")
        if self.session == "train":
            self.logger.info("Building model...")
            self.build()
        else:
            self.model = self.load(
                self.config.get(f"model_{self.name}_path"),
                self.config.get(f"model_{self.name}_weight_path"),
            )

    # build
    @abstractmethod
    def build(self):
        pass

    # save
    def save(self, checkpoint_path):
        self.logger.info(f"Saving model checkpoint to {checkpoint_path} ...")
        if not self.model:
            self.logger.error("Model is not built yet.")
            raise ValueError("Model is not built yet.")
        self.model.save_weights(checkpoint_path)
        self.logger.info(f"Model checkpoint saved to {checkpoint_path}.")

    # load
    def load(self, model_path, checkpoint_path):
        self.logger.info(f"Loading model architecture from {model_path} ...")
        with open(model_path) as f:
            model = model_from_json(f.read())
        self.logger.info(f"Loading model weights from {checkpoint_path} ...")
        model.load_weights(checkpoint_path)
        self.logger.info(f"Model loaded from {model_path} and {checkpoint_path}.")
        return model
