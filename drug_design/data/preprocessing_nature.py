import time

import numpy as np
from rdkit import Chem

from .utils import hp_write_in_file


class Preprocessor(object):
    def __init__(self):
        pass

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

    min_len = 70
    max_len = 140

    print("Current data being processed...")

    app = Preprocessor()
    data = app.process(input_file, min_len, max_len, output_file, verbose=True)

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
    main("./data/raw/dataset_sampled.smi", f"./data/interim/tmp.txt")
