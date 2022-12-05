import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from .smiles_tokenizer_molinf import SmilesTokenizer


# Data loader
class DataLoader(Sequence):
    # init
    def __init__(self, config, data_type="train", logger=None):
        self.config = config
        self.data_type = data_type
        self.logger = logger
        if not self.data_type in ["train", "validation", "test", "fine_tune"]:
            self.logger.error("data_type must be train, validation, test or fine_tune.")
            raise ValueError("data_type must be train, validation, test or fine_tune.")

        self.max_len = 0

        if self.data_type == "train":
            self.logger.info("loading train data...")
            self.smiles = self._load(self.config.get("data_path"))
        elif self.data_type == "fine_tune":
            self.logger.info("loading fine_tune data...")
            self.smiles = self._load(self.config.get("fine_tune_data_path"))
        else:
            pass

        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict

        self.tokenized_smiles = self._tokenize(self.smiles)

        if self.data_type in ["train", "validation", "test"]:
            self.logger.info(f"Calculating {self.data_type} data statistics...")
            self.idx = np.arange(len(self.tokenized_smiles))
            self.valid_size = int(
                np.ceil(len(self.tokenized_smiles) * self.config.get("validation_split"))
            )
            self.test_size = int(
                np.ceil(len(self.tokenized_smiles) * self.config.get("test_split"))
            )
            np.random.seed(self.config.get("seed"))
            np.random.shuffle(self.idx)

    def _set_data(self):
        self.logger.info(f"Setting {self.data_type} data...")
        if self.data_type == "train":
            ret = [self.tokenized_smiles[self.idx[i]] for i in self.idx[self.valid_size :]]
        elif self.data_type == "validation":
            ret = [self.tokenized_smiles[self.idx[i]] for i in self.idx[: self.valid_size]]
        elif self.data_type == "test":
            ret = [self.tokenized_smiles[self.idx[i]] for i in self.idx[: self.test_size]]
        else:
            ret = self.tokenized_smiles
        self.logger.info(f"len of {self.data_type} data: {len(ret)}")
        return ret

    def _load(self, data_filename):
        self.logger.info(f"loading {data_filename}...")
        with open(data_filename) as f:
            smiles = [s.rstrip() for s in f]
        self.config["data_len"] = len(smiles)
        self.logger.info(f"len of {data_filename}: {len(smiles)}")
        return smiles

    def _tokenize(self, smiles):
        self.logger.info("tokenizing SMILES...")
        if not isinstance(smiles, list):
            self.logger.error("smiles must be list to tokenize.")
            return None
        tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(smiles)]

        if self.data_type == "train":
            for tokenized_smi in tokenized_smiles:
                length = len(tokenized_smi)
                if self.max_len < length:
                    self.max_len = length
            self.config["train_smiles_max_len"] = self.max_len

        self.logger.info(f"Tokenized SMILES length: {len(tokenized_smiles)}")
        self.logger.info(f"Tokenized SMILES max length: {self.max_len}")
        return tokenized_smiles

    def __len__(self):
        self.logger.info(f"Calculating {self.data_type} data length...")
        target_tokenized_smiles = self._set_data()
        if self.data_type in ["train", "validation"]:
            ret = int(np.ceil(len(target_tokenized_smiles) / float(self.config.get("batch_size"))))
        else:
            ret = int(
                np.ceil(
                    len(target_tokenized_smiles) / float(self.config.get("fine_tune_batch_size"))
                )
            )
        self.logger.info(f"len of {self.data_type} data: {ret} / {len(target_tokenized_smiles)}")
        return ret

    def __getitem__(self, idx=None):
        self.logger.info(f"Getting {self.data_type} data...")
        target_tokenized_smiles = self._set_data()
        # if self.data_type in ["train", "validation"]:
        #     data = target_tokenized_smiles[
        #         idx * self.config.get("batch_size") : (idx + 1) * self.config.get("batch_size")
        #     ]
        # else:
        #     data = target_tokenized_smiles[
        #         idx
        #         * self.config.get("fine_tune_batch_size") : (idx + 1)
        #         * self.config.get("fine_tune_batch_size")
        #     ]
        # data = self._padding(data)
        data = self._padding(target_tokenized_smiles)

        self.x, self.y = [], []
        for tp_smi in data:
            X = [self.one_hot_dict[symbol] for symbol in tp_smi[:-1]]
            self.x.append(X)
            y = [self.one_hot_dict[symbol] for symbol in tp_smi[1:]]
            self.y.append(y)

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        self.logger.info(f"X shape: {self.x.shape}")
        self.logger.info(f"y shape: {self.y.shape}")
        return self.x, self.y

    def _pad(self, tokenized_smi):
        return (
            ["G"] + tokenized_smi + ["E"] + ["A" for _ in range(self.max_len - len(tokenized_smi))]
        )

    def _padding(self, data):
        self.logger.info("padding SMILES...")
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles
