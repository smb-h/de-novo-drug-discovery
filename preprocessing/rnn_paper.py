#!/usr/bin/env python

import argparse
import collections
import os
import re
import time
from datetime import datetime

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, MolStandardize
from tqdm import tqdm

from config.settings import FP_DESCRIPTORS

from .utils import (
    hp_chem_extract_murcko_scaffolds,
    hp_chem_fingerprint_calc,
    hp_chem_get_rdkit_desc_functions,
    hp_chem_rdkit_desc,
    hp_load_obj,
    hp_save_obj,
    hp_write_in_file,
)


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
            print(
                f"Size of the dataset after pruning by length and check with RDKit: {len(data)}"
            )

        return data, data_rdkit

    def analyze_data(self, data_rdkit, descriptor_name, save_path, verbose=False):
        """
        Function to analize a dataset. Will compute: descritpor as specify in
        descriptors_name, Morgan fingerprint, Murcko and generic scaffolds.

        Parameters:
        - data_rdkit: list of RDKit mol.
        - descriptor_name (string): contain name of descriptor to compute.
        - save_path (string): Path to save the output of the analysis.
        """

        # Compute the descriptors with rdkit
        # as defined in the fixed parameter file
        desc_names = re.compile(FP_DESCRIPTORS["names"])
        functions, names = hp_chem_get_rdkit_desc_functions(desc_names)
        descriptors = hp_chem_rdkit_desc(data_rdkit, functions, names)
        hp_save_obj(descriptors, f"{save_path}/desc.pkl")

        # Compute fingerprints
        fingerprint = hp_chem_fingerprint_calc(data_rdkit, verbose=verbose)
        fp_dict = {"fingerprint": fingerprint}
        hp_save_obj(fp_dict, f"{save_path}/fp.pkl")

        # Extract Murcko and generic scaffolds
        scaf, generic_scaf = hp_chem_extract_murcko_scaffolds(data_rdkit)
        desc_scaf = {"scaffolds": scaf, "generic_scaffolds": generic_scaf}
        hp_save_obj(desc_scaf, f"{save_path}/scaf.pkl")
        hp_write_in_file(f"{save_path}/generic_scaffolds.txt", generic_scaf)
        hp_write_in_file(f"{save_path}/scaffolds.txt", scaf)

    def draw_scaffolds(self, top_common, path):
        """
        Function to draw scaffolds with rdkit.

        Parameters:
        - dict_scaf: dictionnary with scaffolds.
        - top_common (int): how many of the most common
        scaffolds to draw.
        - path (string): Path to save the scaffolds picture
        and to get the scaffolds data.
        """

        path_scaffolds = f"{path}/scaf"
        data_ = hp_load_obj(path_scaffolds)

        for name_data, data in data_.items():
            # Note that some molecules are put as a list
            # with a string error; we remove them for drawing
            # Note 2: they occur very rarely
            data = [x for x in data if type(x) is str]
            counter = collections.Counter(data)
            common = counter.most_common(top_common)

            total = sum(counter.values())
            mols = [Chem.MolFromSmiles(x[0]) for x in common[:top_common]]
            repet = [
                str(x[1]) + f"({100*x[1]/total:.2f}%)" for x in common[:top_common]
            ]

            molsPerRow = 5
            common_top = Draw.MolsToGridImage(
                mols, molsPerRow=molsPerRow, subImgSize=(150, 150), legends=repet
            )

            common_top.save(f"{path}/top_{top_common}_{name_data}.png")

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
        data_ori, data_rdkit = self.load_data(
            data_path, min_len, max_len, verbose=verbose
        )

        # we save the data without augmentation if it was
        # not already saved. We will need it to check the novelty
        # of the generated SMILES
        if os.path.isfile(f"{save_path}/pruned.txt"):
            hp_write_in_file(f"{save_path}/pruned.txt", data_ori)

        if verbose:
            print("Start data analysis")
        self.analyze_data(data_rdkit, FP_DESCRIPTORS["names"], save_path)

        # draw top scaffolds
        if verbose:
            print("Start drawing scaffolds")
        top_common = 20
        self.draw_scaffolds(top_common, save_path)

        return


def main(input_file, output_file, **kwargs):
    start = time.time()

    min_len = 1
    max_len = 140

    print("\nSTART PROCESSING")
    print("Current data being processed...")

    app = Preprocessor()
    app.process(input_file, min_len, max_len, output_file, verbose=True)

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
    experiment_name = "test"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    # make experiment directory
    os.makedirs(os.path.join("experiments", timestamp, experiment_name), exist_ok=True)
    main("./data/dataset_sampled.smi", f"./experiments/{timestamp}/{experiment_name}")
