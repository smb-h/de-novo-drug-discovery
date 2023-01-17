from rdkit import Chem

from drug_design.data.smiles_tokenizer_molinf import SmilesTokenizer
from drug_design.visualization.visualize import plot_scatter_org_vs_pred, plot_violin_org_vs_pred


class Predictor(object):
    # init
    def __init__(self, config, model_name, model, train_data, test_data, plot=True, logger=None):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.x_train = train_data
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.y_test_decoded = []
        self.plot = plot
        self.logger = logger
        self.y_pred = None
        self.y_pred_mols = None
        self.y_pred_decoded = []
        self.validity = []
        self.uniqueness = []
        self.originality = []

    # predict
    def predict(self):
        self.logger.info(f"Start predicting ({self.model_name})...")
        self.logger.info(f"Test shape: {self.x_test.shape}")
        iterations = self.config.get("prediction_iterations", 10)
        for iter in range(iterations):
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
            self.y_pred_decoded.extend(st.remove_paddings(y_pred_decoded))
            y_test_decoded = st.one_hot_decode(self.y_test)
            self.y_test_decoded.extend(st.remove_paddings(y_test_decoded))

            # check validity
            self.check_validity()
            # check uniqueness
            self.check_uniqueness()
            # check originality
            self.check_originality()

        # Plot only last iteration
        if self.plot:
            self.plot_reports()

        # mean self.validity
        self.validity = sum(self.validity) / len(self.validity)
        self.logger.info(f"Mean Low Validity({iterations}): {self.validity:.2%}")
        # mean self.uniqueness
        self.uniqueness = sum(self.uniqueness) / len(self.uniqueness)
        self.logger.info(f"Mean High Uniqueness({iterations}): {self.uniqueness:.2%}")
        # mean self.originality
        self.originality = sum(self.originality) / len(self.originality)
        self.logger.info(f"Mean High Originality({iterations}): {self.originality:.2%}")

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

    # check validity
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

        # low validity
        validity = len(self.y_pred_decoded) / 30000
        self.validity.append(validity)
        # self.logger.info(f"Validity Ratio: {len(self.y_pred_mols) / len(self.y_pred_decoded):.2%}")
        # self.logger.info(f"Low Validity: {validity:.2%}")

    # check uniqueness
    def check_uniqueness(self):
        """
        Check y_pred high uniqueness
        """
        y_pred_smiles = [Chem.MolToSmiles(smi) for smi in self.y_pred_mols]
        # high uniqueness
        uniqueness = len(set(y_pred_smiles)) / len(y_pred_smiles)
        self.uniqueness.append(uniqueness)
        # self.logger.info(f"High Uniqueness: {uniqueness:.2%}")

    # check originality
    def check_originality(self):
        """
        Check y_pred high originality
        """
        y_pred_smiles = [Chem.MolToSmiles(smi) for smi in self.y_pred_mols]
        # originals = predicted mols that are not in the training set
        originals = [smi for smi in y_pred_smiles if smi not in self.x_train]
        # high originality
        originality = len(originals) / len(y_pred_smiles)
        self.originality.append(originality)
        # self.logger.info(f"High Originality: {originality:.2%}")
