class Predictor(object):
    # init
    def __init__(self, config, model, test_data):
        self.model = model
        self.config = config
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.y_pred = self.predict()

    # predict
    def predict(self):
        # predict
        # TODO: discuss this iteration with Arash
        # for iter in self.config.get("prediction_iterations", 10):
        y_pred = self.model.predict(
            {
                "Input_Ex1": self.x_test,
                "polarizer": self.x_test,
                "Input_EX3": self.x_test,
            },
            use_multiprocessing=True,
            verbose=True,
        )
        return y_pred
