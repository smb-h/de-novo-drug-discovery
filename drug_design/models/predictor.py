from rdkit import Chem

from drug_discovery.data.smiles_tokenizer_molinf import SmilesTokenizer
from drug_discovery.visualization.visualize import plot_scatter_org_vs_pred, plot_violin_org_vs_pred


class Predictor(object):
    # init
    def __init__(self, config, model_name, model, test_data, plot=True, logger=None):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.y_test_decoded = None
        self.plot = plot
        self.logger = logger
        self.y_pred = None
        self.y_pred_mols = None
        self.y_pred_decoded = None

    # predict
    def predict(self):
        self.logger.info(f"Start predicting ({self.model_name})...")
        self.logger.info(f"Test shape: {self.x_test.shape}")
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

        st = SmilesTokenizer()
        y_pred_decoded = st.one_hot_decode(self.y_pred)
        self.y_pred_decoded = st.remove_paddings(y_pred_decoded)
        y_test_decoded = st.one_hot_decode(self.y_test)
        self.y_test_decoded = st.remove_paddings(y_test_decoded)

        if self.plot:
            self.plot_reports()

        # check validity
        self.check_validity()
        # check uniqueness
        self.check_uniqueness()

    # plot reports
    def plot_reports(self):
        self.logger.info("Plotting reports...")
        # plot scatter
        plot_scatter_org_vs_pred(
            self.config, self.model_name, self.y_test_decoded, self.y_pred_decoded
        )
        # plot violin
        plot_violin_org_vs_pred(
            self.config, self.model_name, self.y_test_decoded, self.y_pred_decoded
        )

    # check prediction validity
    def check_validity(self):
        """
        Check y_pred high validity
        """
        self.y_pred_mols = []
        for smi in self.y_pred_decoded:
            mol = Chem.MolFromSmiles(smi)
            # mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                self.y_pred_mols.append(mol)

        # print(f"Validity Ratio: {len(self.y_pred_mols) / len(self.y_pred_decoded):.2%}")
        self.logger.info(f"Validity Ratio: {len(self.y_pred_mols) / len(self.y_pred_decoded):.2%}")
        # low validity
        # print(f"Low Validity: {len(self.y_pred_decoded) / 30000:.2%}")
        self.logger.info(f"Low Validity: {len(self.y_pred_decoded) / 30000:.2%}")

    def check_uniqueness(self):
        """
        Check y_pred high uniqueness
        """
        y_pred_smiles = [Chem.MolToSmiles(smi) for smi in self.y_pred_mols]

        # high uniqueness
        # print(f"High Uniqueness: {len(set(y_pred_smiles)) / len(y_pred_smiles):.2%}")
        self.logger.info(f"High Uniqueness: {len(set(y_pred_smiles)) / len(y_pred_smiles):.2%}")
