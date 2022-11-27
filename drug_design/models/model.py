from abc import ABC, abstractmethod

from tensorflow.keras.models import model_from_json


# Model
class BaseModel(ABC):
    # init
    def __init__(self, config, session="train") -> None:
        self.model_name = "Model"
        assert session in ["train", "test", "fine_tune"], "One of {train, test, fine_tune}"
        self.config = config
        self.session = session
        self.model = None

        if self.session == "train":
            self.build()
        else:
            self.model = self.load(
                self.config.get(f"model_{self.model_name}_path"),
                self.config.get(f"model_{self.model_name}_weight_path"),
            )

    # build
    @abstractmethod
    def build(self):
        pass

    # save
    def save(self, checkpoint_path):
        print(f"Saving model architecture to {checkpoint_path} ...")
        assert self.model, "You have to build the model first."
        print("Saving model ...")
        # save model as json
        model_json = self.model.to_json()
        with open(f"{checkpoint_path}.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(checkpoint_path)

        print("model saved.")

    # load
    def load(self, model_file, checkpoint_file):
        print(f"Loading model architecture from {model_file} ...")
        with open(model_file) as f:
            model = model_from_json(f.read())
        print(f"Loading model checkpoint from {checkpoint_file} ...")
        model.load_weights(checkpoint_file)
        print("Loaded the Model.")
        return model
