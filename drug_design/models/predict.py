from drug_design.data.smiles_tokenizer_molinf import SmilesTokenizer
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
        self.plot = plot

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
        if self.plot:
            self.plot_reports()

    def plot_reports(self):
        st = SmilesTokenizer()

        y_test_decoded = st.one_hot_decode(self.y_test)
        y_test_decoded = st.remove_paddings(y_test_decoded)
        y_pred_decoded = st.one_hot_decode(self.y_pred)
        y_pred_decoded = st.remove_paddings(y_pred_decoded)

        # TODO: Delete this (it's just for testing)
        y_pred_decoded = y_pred_decoded[: int(len(y_pred_decoded) / 2)]
        y_test_decoded = y_test_decoded[int(len(y_test_decoded) / 2) :]

        # plot scatter
        plot_scatter_org_vs_pred(self.config, self.model_name, y_test_decoded, y_pred_decoded)
        # plot violin
        plot_violin_org_vs_pred(self.config, self.model_name, y_test_decoded, y_pred_decoded)
