from abc import ABC, abstractmethod

from tensorflow.keras.models import model_from_json


# Model
class BaseModel(ABC):
    # init
    def __init__(self, config, session="train") -> None:
        self.model_name = "Model"
        assert session in ["train", "fine_tune"], "One of {train, fine_tune}"
        self.config = config
        self.session = session
        self.model = None

        if self.session == "train":
            self.build()
        else:
            self.model = self.load(
                self.config.get(f"model_{self.model_name}_arch_filepath"),
                self.config.get(f"model_{self.model_name}_weight_filepath"),
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
    def load(self, model_arch_file, checkpoint_file):
        print(f"Loading model architecture from {model_arch_file} ...")
        with open(model_arch_file) as f:
            model = model_from_json(f.read())
        print(f"Loading model checkpoint from {checkpoint_file} ...")
        model.load_weights(checkpoint_file)
        print("Loaded the Model.")
        return model
