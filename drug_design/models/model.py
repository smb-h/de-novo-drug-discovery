from abc import ABC, abstractmethod

from tensorflow.keras.models import model_from_json


# Model
class BaseModel(ABC):
    # init
    def __init__(self, config, session="train") -> None:
        self.name = "Model"
        assert session in ["train", "test", "fine_tune"], "One of {train, test, fine_tune}"
        self.config = config
        self.session = session
        self.model = None

        if self.session == "train":
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
        assert self.model, "You have to build the model first."
        print("Saving model ...")
        self.model.save_weights(checkpoint_path)
        print("model saved.")

    # load
    def load(self, model_path, checkpoint_path):
        print(f"Loading model architecture from {model_path} ...")
        with open(model_path) as f:
            model = model_from_json(f.read())
        print(f"Loading model checkpoint from {checkpoint_path} ...")
        model.load_weights(checkpoint_path)
        print("Loaded the Model.")
        return model
