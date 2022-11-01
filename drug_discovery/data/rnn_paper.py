#!/usr/bin/env python

import argparse
import collections
import os
import re
import time

import numpy as np
from rdkit import Chem

from .utils import hp_write_in_file


class Preprocessor(object):
    def __init__(self):
        self.one_hot = [
            "C",
            "N",
            "O",
            "H",
            "F",
            "Cl",
            "P",
            "B",
            "Br",
            "S",
            "I",
            "Si",
            "#",
            "(",
            ")",
            "+",
            "-",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "=",
            "[",
            "]",
            "@",
            "c",
            "n",
            "o",
            "s",
            "X",
            ".",
        ]

    def load_data(self, data_path, min_len, max_len, verbose=True):
        """
        Function to load a .txt file of SMILES,
        prune SMILES by length and check that they
        are convertible to RDKit mol format.

        Parameters:
        - data_path (string): path to the dataset.
        - min_len (int): minimum length of SMILES to be kept in the dataset.
        - max_len (int): maximum length of SMILES to be kept in the dataset.

        return:
        data -> a list with SMILES in string format
        data_rdkit -> a list with molecules in RDKit mol format
        """
        data = []
        data_rdkit = []

        with open(data_path) as f:
            for line in f:
                newline = line.rstrip("\r\n")
                if len(newline) <= max_len and len(newline) >= min_len:
                    # convert to RDKit mol format
                    mol = Chem.MolFromSmiles(newline)
                    if mol is not None:
                        data.append(newline)
                        data_rdkit.append(mol)

        if verbose:
            print(f"Size of the dataset after pruning by length and check with RDKit: {len(data)}")

        return data, data_rdkit

    # def one_hot_encode(self, token_lists, n_chars):
    #     output = np.zeros((len(token_lists), len(token_lists[0]), n_chars))
    #     for i, token_list in enumerate(token_lists):
    #         for j, token in enumerate(token_list):
    #             output[i, j, int(token)] = 1
    #     return output

    def one_hot_encode(self, smiles, pad_len=-1):
        encoded_smiles = []
        for smile in smiles:
            this_smile = smile + "."
            if pad_len < 0:
                vec = np.zeros((len(this_smile), len(self.one_hot)))
            else:
                vec = np.zeros((pad_len, len(self.one_hot)))
            cont = True
            j = 0
            i = 0
            while cont:

                try:
                    if this_smile[i + 1] in ["r", "i", "l"]:
                        sym = this_smile[i : i + 2]
                        i += 2
                    else:
                        sym = this_smile[i]
                        i += 1
                except:
                    print(f"this_smile[i + 1] not working, value this_smile {this_smile}")
                if sym in self.one_hot:
                    vec[j, self.one_hot.index(sym)] = 1
                else:
                    vec[j, self.one_hot.index("X")] = 1
                j += 1
                if this_smile[i] == "." or j >= (pad_len - 1) and pad_len > 0:
                    vec[j, self.one_hot.index(".")] = 1
                    cont = False
            encoded_smiles.append(vec)
        print(f"count of encoded smiles: {len(encoded_smiles)}")
        return encoded_smiles

    def one_hot_decode(self, encoded_smiles):
        decoded_smiles = []
        for smile in encoded_smiles:
            this_smile = ""
            for token in smile:
                this_smile += self.one_hot[np.argmax(token)]
            # remove the padding
            this_smile = this_smile.replace(".", "")
            decoded_smiles.append(this_smile)
        print(f"count of decoded smiles: {len(decoded_smiles)}")
        return decoded_smiles

    def process(self, data_path, min_len, max_len, save_path, verbose=True):
        """
        Function to process a dataset.

        Parameters:
        - split (float): value used to split the dataset between
        the training set and the validation set. E.g., if split is 0.8,
        80% of the data will go in the training set, and 20% in the
        validation set.
        - data_path (string): path to the dataset.
        - augmentation (int): value to augment the dataset. E.g., if augmentation
        is 10, the SMILES enumeration will be done to add 10 different
        SMILES encoding for each SMILES (i.e. resulting in a total of 11 representations)
        for a given SMILES in the dataset.
        - min_len (int): minimum length of SMILES to be kept in the dataset.
        - max_len (int): maximum length of SMILES to be kept in the dataset.
        - save_path (string): path to save the processed dataset.
        """

        # load the data with right SMILES limits,
        # both in a list and in rdkit mol format
        data_ori, data_rdkit = self.load_data(data_path, min_len, max_len, verbose=verbose)

        return data_ori


def main(input_file, output_file, **kwargs):
    start = time.time()

    min_len = 1
    max_len = 140

    print("\nSTART PROCESSING")
    print("Current data being processed...")

    app = Preprocessor()
    data = app.process(input_file, min_len, max_len, output_file, verbose=True)

    print("---------------------")
    encoded = app.one_hot_encode(data)
    # print(f"Sampled data: {data[0]}")
    # print(f"length of sampled data: {len(data[0])}")
    # print(f"encoded: {encoded[0]}", f" ~ length: {len(encoded[0])}")
    # decoded = app.one_hot_decode(encoded)
    # print(f"decoded: {decoded[0]}", f" ~ length: {len(decoded[0])}")

    # print(encoded)
    print(len(encoded))
    for i in encoded:
        print(i.shape)

    hp_write_in_file(output_file, data)

    end = time.time()
    print(f"PROCESSING DONE in {end - start:.04} seconds")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run data processing")
    # parser.add_argument(
    #     "-fn",
    #     "--filename",
    #     type=str,
    #     help="Path to the fine-tuning txt file",
    #     required=True,
    # )
    # parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)
    main("./src/data/dataset_sampled.smi", f"./src/data/tmp.txt")
