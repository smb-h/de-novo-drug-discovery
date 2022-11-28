from drug_design.visualization.visualize import plot_scatter_org_vs_pred, plot_violin_org_vs_pred


class Predictor(object):
    # init
    def __init__(self, config, model_name, model, test_data, plot=True):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.y_pred = None
        if plot:
            self.plot_reports()

    # predict
    def predict(self):
        # predict
        # TODO: discuss this iteration with Arash
        # for iter in self.config.get("prediction_iterations", 10):
        self.y_pred = self.model.predict(
            {
                "Input_Ex1": self.x_test,
                "polarizer": self.x_test,
                "Input_EX3": self.x_test,
            },
            use_multiprocessing=True,
            verbose=True,
        )

    def plot_reports(self):
        # plot scatter
        plot_scatter_org_vs_pred(self.config, self.model_name, self.y_test, self.y_pred)
        # plot violin
        plot_violin_org_vs_pred(self.config, self.model_name, self.y_test, self.y_pred)
