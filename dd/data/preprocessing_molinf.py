import argparse
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from .smiles_tokenizer_molinf import SmilesTokenizer

# RDLogger.DisableLog('rdApp.*')


class Preprocessor(object):
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            # mol = self.lfc.choose(mol)
            # mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return smi
        else:
            return None


def main(input_file, output_file, **kwargs):
    assert os.path.exists(input_file), f"{input_file} does not exists!"

    pp = Preprocessor()

    with open(input_file, "r") as f:
        smiles = [l.rstrip() for l in f]

    print(f"input SMILES num: {len(smiles)}")
    print("start to clean up")

    pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
    cl_smiles = list(set([s for s in pp_smiles if s]))

    # token limits (34 to 74)
    out_smiles = []
    st = SmilesTokenizer()

    # if kwargs["finetune"]:
    #     for cl_smi in cl_smiles:
    #         tokenized_smi = st.tokenize(cl_smi)
    #         if 34 <= len(tokenized_smi) <= 74:
    #             out_smiles.append(cl_smi)
    # else:
    #     out_smiles = cl_smiles
    out_smiles = cl_smiles

    print("done.")
    print(f"output SMILES num: {len(out_smiles)}")

    with open(output_file, "w") as f:
        for smi in out_smiles:
            f.write(smi + "\n")

    return


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="remove salts and stereochemical infomation from SMILES"
    # )
    # parser.add_argument('input', help='input file')
    # parser.add_argument('output', help='output file')
    # parser.add_argument('-ft',
    #                     '--finetune',
    #                     action='store_false',
    #                     help='for finetuning. ignore token length limitation.')
    # args = parser.parse_args()
    # main(args.input, args.output, finetune=args.finetune)
    main("./data/raw/dataset_isomeric_true.smi", "./data/processed/dataset.smi", finetune=True)
