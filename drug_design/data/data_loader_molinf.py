import json
import os

import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from .smiles_tokenizer_molinf import SmilesTokenizer


class DataLoader(Sequence):
    def __init__(self, config, data_type="train"):
        self.config = config
        self.data_type = data_type
        assert self.data_type in ["train", "validation", "test", "fine_tune"]

        self.max_len = 0

        if self.data_type == "train":
            self.smiles = self._load(self.config.get("data_path"))
        elif self.data_type == "fine_tune":
            self.smiles = self._load(self.config.get("fine_tune_data_path"))
        else:
            pass

        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict

        self.tokenized_smiles = self._tokenize(self.smiles)

        if self.data_type in ["train", "validation", "test"]:
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
        if self.data_type == "train":
            ret = [self.tokenized_smiles[self.idx[i]] for i in self.idx[self.valid_size :]]
        elif self.data_type == "validation":
            ret = [self.tokenized_smiles[self.idx[i]] for i in self.idx[: self.valid_size]]
        elif self.data_type == "test":
            ret = [self.tokenized_smiles[self.idx[i]] for i in self.idx[: self.test_size]]
        else:
            ret = self.tokenized_smiles
        return ret

    def _load(self, data_filename):
        with open(data_filename) as f:
            smiles = [s.rstrip() for s in f]
        print("done.")
        self.config["data_len"] = len(smiles)
        return smiles

    def _tokenize(self, smiles):
        assert isinstance(smiles, list)
        print("tokenizing SMILES...")
        tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(smiles)]

        if self.data_type == "train":
            for tokenized_smi in tokenized_smiles:
                length = len(tokenized_smi)
                if self.max_len < length:
                    self.max_len = length
            self.config["train_smiles_max_len"] = self.max_len
        print("done.")
        return tokenized_smiles

    def __len__(self):
        target_tokenized_smiles = self._set_data()
        if self.data_type in ["train", "validation"]:
            ret = int(np.ceil(len(target_tokenized_smiles) / float(self.config.get("batch_size"))))
        else:
            ret = int(
                np.ceil(
                    len(target_tokenized_smiles) / float(self.config.get("fine_tune_batch_size"))
                )
            )
        return ret

    def __getitem__(self, idx=None):
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

        self.X, self.y = [], []
        for tp_smi in data:
            X = [self.one_hot_dict[symbol] for symbol in tp_smi[:-1]]
            self.X.append(X)
            y = [self.one_hot_dict[symbol] for symbol in tp_smi[1:]]
            self.y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        return self.X, self.y

    def _pad(self, tokenized_smi):
        return (
            ["G"] + tokenized_smi + ["E"] + ["A" for _ in range(self.max_len - len(tokenized_smi))]
        )

    def _padding(self, data):
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles
