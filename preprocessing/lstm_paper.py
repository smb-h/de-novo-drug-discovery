#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
import numpy as np


# RDLogger.DisableLog('rdApp.*')


class SmilesTokenizer(object):
    def __init__(self):
        atoms = [
            'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N',
            'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te'
        ]
        special = [
            '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
        ]
        padding = ['G', 'A', 'E']

        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        table_len = len(self.table)

        self.table_2_chars = list(filter(lambda x: len(x) == 2, self.table))
        self.table_1_chars = list(filter(lambda x: len(x) == 1, self.table))

        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec

    def tokenize(self, smiles):
        smiles = smiles + ' '
        N = len(smiles)
        token = []
        i = 0
        while (i < N):
            c1 = smiles[i]
            c2 = smiles[i:i + 2]

            if c2 in self.table_2_chars:
                token.append(c2)
                i += 2
                continue

            if c1 in self.table_1_chars:
                token.append(c1)
                i += 1
                continue

            i += 1

        return token

    def one_hot_encode(self, tokenized_smiles):
        result = np.array(
            [self.one_hot_dict[symbol] for symbol in tokenized_smiles],
            dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result


class Preprocessor(object):
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None


def main(input_file, output_file, **kwargs):
    assert os.path.exists(input_file), f'{input_file} does not exists!'
    # assert not os.path.exists(output_file), f'{output_file} already exists.'

    pp = Preprocessor()

    with open(input_file, 'r') as f:
        smiles = [l.rstrip() for l in f]


    # test on first 1000 records
    smiles = smiles[:1000]


    print(f'input SMILES num: {len(smiles)}')
    print('start to clean up')

    pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
    cl_smiles = list(set([s for s in pp_smiles if s]))

    # token limits (34 to 74)
    out_smiles = []
    st = SmilesTokenizer()

    if kwargs['finetune']:
        for cl_smi in cl_smiles:
            tokenized_smi = st.tokenize(cl_smi)
            if 34 <= len(tokenized_smi) <= 74:
                out_smiles.append(cl_smi)
    else:
        out_smiles = cl_smiles

    print('done.')
    print(f'output SMILES num: {len(out_smiles)}')

    with open(output_file, 'w') as f:
        for smi in out_smiles:
            f.write(smi + '\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='remove salts and stereochemical infomation from SMILES')
    # parser.add_argument('input', help='input file')
    # parser.add_argument('output', help='output file')
    # parser.add_argument('-ft',
    #                     '--finetune',
    #                     action='store_false',
    #                     help='for finetuning. ignore token length limitation.')
    # args = parser.parse_args()
    # main(args.input, args.output, finetune=args.finetune)
    main('./data/dataset_sampled.smi', './data/tmp.txt', finetune=True)
