# Copyright (c) 2019 ETH Zurich

import os, sys
import argparse
import configparser
import time
import re
import numpy as np
import random
import collections
from random import shuffle
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import pandas as  pd
from rdkit.Chem import AllChem as Chem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# parser = argparse.ArgumentParser(description='Run data processing')
# parser.add_argument('-fn','--filename', type=str, help='Path to the fine-tuning txt file', required=True)
# parser.add_argument('-v','--verbose', type=bool, help='Verbose', required=True)

# Constants
FP_PROCESSING_FIXED = {'start_char': 'G', 
                    'end_char': 'E', 
                    'pad_char': 'A'}

FP_INDICES_TOKEN = {"0": 'H', "1": '9', "2": 'D', "3": 'r', "4": 'T', "5": 'R', "6": 'V', "7": '4',
                 "8": 'c', "9": 'l', "10": 'b', "11": '.', "12": 'C', "13": 'Y', "14": 's', "15": 'B',
                 "16": 'k', "17": '+', "18": 'p', "19": '2', "20": '7', "21": '8', "22": 'O',
                 "23": '%', "24": 'o', "25": '6', "26": 'N', "27": 'A', "28": 't', "29": '$',
                 "30": '(', "31": 'u', "32": 'Z', "33": '#', "34": 'M', "35": 'P', "36": 'G',
                 "37": 'I', "38": '=', "39": '-', "40": 'X', "41": '@', "42": 'E', "43": ':',
                 "44": '\\', "45": ')', "46": 'i', "47": 'K', "48": '/', "49": '{', "50": 'h',
                 "51": 'L', "52": 'n', "53": 'U', "54": '[', "55": '0', "56": 'y', "57": 'e',
                 "58": '3', "59": 'g', "60": 'f', "61": '}', "62": '1', "63": 'd', "64": 'W',
                 "65": '5', "66": 'S', "67": 'F', "68": ']', "69": 'a', "70": 'm'}
FP_TOKEN_INDICES = {v: k for k, v in FP_INDICES_TOKEN.items()}


FP_PAPER_FONT = {'tick_font_sz': 15,
              'label_font_sz': 18,
              'legend_sz': 16,
              'title_sz': 22}

# Number of molecules to use to
# make the UMAP plot.
FP_UMAP_PLOT = {'n_dataset': 1000,
             'n_gen': 1000}

# Number of molecules to use 
# to compute the Fréchet distance.
# This is an upper bound.
FP_FRECHET = {'n_data': 5000}

# Color palette for UMAP
FP_COLOR_PAL_CB = {'source': '#1575A4',
                'target': '#D55E00',
                'e_start': '#A0E0FF',
                'e_end': '#FFAE6E'}

FP_DESCRIPTORS = {'names': '(FractionCSP3)'}

# Helper
def hp_read_with_pd(path, delimiter='\t', header=None):
    data_pd = pd.read_csv(path, delimiter=delimiter, header=header)
    return data_pd[0].tolist() 

def hp_save_obj(obj, name):
    """save obj with pickle"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def hp_load_obj(name):
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def hp_write_in_file(path_to_file, data):
    with open(path_to_file, 'w+') as f:
        for item in data:
            f.write("%s\n" % item)


# Helper chem
def hp_chem_extract_murcko_scaffolds(mols, verbose=True):
    """ Extract Bemis-Murcko scaffolds from a smile string.

    :param mols: molecule data set in rdkit mol format.
    :return: smiles string of a scaffold and a framework.
    """
    scaf = []
    scaf_unique = []
    generic_scaf = []
    generic_scaf_unique = []
    start = time.time()
    for mol in mols:
        if mol is None:
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            scaf.append(Chem.MolToSmiles(core, isomericSmiles=True))
            generic_scaf.append(Chem.MolToSmiles(fw, isomericSmiles=True))
        except ValueError as e:
            print(e)
            scaf.append(['error'])
            generic_scaf.append(['error'])
    if verbose:
        print('Extracted', len(scaf), 'scaffolds in', time.time() - start, 'seconds.')
    return scaf, generic_scaf

def hp_chem_get_rdkit_desc_functions(desc_names):
    """
    Allows to define RDKit Descriptors by regex wildcards. 
    :return: Descriptor as functions for calculations and names.
    """
    functions = []
    descriptors = []

    for descriptor, function in Descriptors._descList:
        if (desc_names.match(descriptor) != None):
            descriptors = descriptors + [descriptor]
            functions = functions + [function]
    return functions, descriptors

def hp_chem_rdkit_desc(mols, functions, names, verbose=True):
    """
    Calculate RDKit descriptors for a set of molecules.
    Returns calculated descriptors in a dict by their name
    """
    start = time.time()
    descriptors = {}
    
    for function, name in zip(functions, names):
        desc = [function(mol) for mol in mols]
        descriptors[name] = desc
        
    if verbose:
        print(f'{len(functions)} descriptors for {len(mols)} mols were calculated in {time.time()-start:.03} seconds.')
    
    return descriptors

def hp_chem_fingerprint_calc(mols, verbose=True):
    """
    Calculate Morgan fingerprints (circular fingerprint) for set of molecules
    :param mols: input dataset in rdkit mol format.
    :return: Morgan fingerprint for each mol.
    """
    start = time.time()
    morganfp = [Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False) for mol in mols]
    if verbose:
        print(f'Fingerprint calculation for {len(mols)} mols took {time.time()-start:.03} seconds.')
    return morganfp

def hp_chem_cleanup_smiles(all_mols, salt_removal=True, stereochem=False, canonicalize=True):
    """ Clean up for SMILES input file. Function sent by Lukas and mofidied.
    to be used for seq2seq like model (e.g. we don't remove duplicates).
    
    :param all_mols: INPUT data file with SMILES strings. One SMILES string per line.
    :param salt_removal: Check for salts (.) and removes smaller fragment. (default = TRUE)
    :param stereochem: Keep stereochemical information (@, /, \).
    :return: cleaned SMILES files.
    """
    cleaned_mols = []
    
    for c, smi in enumerate(all_mols):
        if not stereochem:
            stereo_smi = smi
            chars_stereochem = ['\\', '/', '@']
            smi = stereo_smi.translate(str.maketrans('','', ''.join(chars_stereochem)))
        if salt_removal:
            maxlen = 0
            max_smi = ''
            if '.' in smi:
                smi_list = smi.split('.')
                for m in smi_list:
                    if len(m) > maxlen:
                        maxlen = len(m)
                        max_smi = m
                smi = max_smi
        cleaned_mols += [smi]
        
    if canonicalize:
        canon_mols = []
        for c, m in enumerate(cleaned_mols):
            mol = Chem.MolFromSmiles(m)
            if mol is None:
                continue
            canon_mols.append(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
            
    return canon_mols



def load_data(data_path, min_len, max_len, verbose=False):
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
            newline = line.rstrip('\r\n')
            if len(newline)<=max_len and len(newline)>=min_len:
                # convert to RDKit mol format
                mol = Chem.MolFromSmiles(newline)
                if mol is not None:
                    data.append(newline)
                    data_rdkit.append(mol)
    
    if verbose: print(f'Size of the dataset after pruning by length and check with RDKit: {len(data)}')
    
    return data, data_rdkit

def randomSmiles(mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)

def smile_augmentation(smile, augmentation, min_len, max_len):
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        smiles = randomSmiles(mol)
        if len(smiles)<=max_len:
            s.add(smiles)
            if len(s)==augmentation:
                break
    
    return list(s)

def augment_dataset(data_ori, augmentation, min_len, max_len, verbose=False):
    """ 
    Function to augment a dataset. 
    
    Parameters:
    - data_ori (list): list of SMILES string to augment.
    - augmentation (int): number of alternative SMILES to create.
    - min_len (int): minimum length of alternative SMILES.
    - max_len (int): maximum length of alternative SMILES.
    
    return: a list alternative SMILES representations of data_ori
    """
    all_alternative_smi = []    
    for i,x in enumerate(data_ori):
        alternative_smi = smile_augmentation(x, augmentation, min_len, max_len)
        all_alternative_smi.extend(alternative_smi)
        if verbose and i%50000:
            print(f'augmentation is at step {i}')
    if verbose:
        print('data augmentation done; number of new SMILES: {len(n_new)}')
        
    return all_alternative_smi


def do_data_analysis(data_rdkit, descriptor_name, save_dir, verbose=False):
    """
    Function to analize a dataset. Will compute: descritpor as specify in
    descriptors_name, Morgan fingerprint, Murcko and generic scaffolds.
    
    Parameters:
    - data_rdkit: list of RDKit mol.
    - descriptor_name (string): contain name of descriptor to compute.
    - save_dir (string): Path to save the output of the analysis.
    """
    
    # Compute the descriptors with rdkit
    # as defined in the fixed parameter file
    desc_names = re.compile(FP_DESCRIPTORS['names'])
    functions, names = hp_chem_get_rdkit_desc_functions(desc_names)
    descriptors = hp_chem_rdkit_desc(data_rdkit, functions, names)
    hp_save_obj(descriptors, f'{save_dir}desc')
    
    # Compute fingerprints
    fingerprint = hp_chem_fingerprint_calc(data_rdkit, verbose=verbose)
    fp_dict = {'fingerprint': fingerprint}
    hp_save_obj(fp_dict, f'{save_dir}fp')  
    
    # Extract Murcko and generic scaffolds
    scaf, generic_scaf = hp_chem_extract_murcko_scaffolds(data_rdkit)
    desc_scaf = {'scaffolds': scaf, 'generic_scaffolds': generic_scaf}
    hp_save_obj(desc_scaf, f'{save_dir}scaf')
    hp_write_in_file(f'{save_dir}generic_scaffolds.txt', generic_scaf)
    hp_write_in_file(f'{save_dir}scaffolds.txt', scaf)

    
def draw_scaffolds(top_common, path):
    """ 
    Function to draw scaffolds with rdkit. 
    
    Parameters:
    - dict_scaf: dictionnary with scaffolds.
    - top_common (int): how many of the most common
    scaffolds to draw.
    - path (string): Path to save the scaffolds picture
    and to get the scaffolds data.
    """
    
    path_scaffolds =  f'{path}scaf'
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
        repet = [str(x[1]) + f'({100*x[1]/total:.2f}%)' for x in common[:top_common]]
                
        molsPerRow = 5
        common_top = Draw.MolsToGridImage(mols,
                                          molsPerRow=molsPerRow,
                                          subImgSize=(150,150),
                                          legends=repet)
    
        common_top.save(f'{path}top_{top_common}_{name_data}.png')
        
        
def do_processing(split, data_path, augmentation, min_len, max_len, save_dir, verbose=True):
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
    - save_dir (string): directory to save the processed dataset.
    """
    
    # load the data with right SMILES limits, 
    # both in a list and in rdkit mol format 
    data_ori, data_rdkit = load_data(data_path, min_len, max_len, verbose=verbose)
    
    # we save the data without augmentation if it was
    # not already saved. We will need it to check the novelty
    # of the generated SMILES
    if os.path.isfile(f'{save_dir}pruned.txt'):
        hp_write_in_file(f'{save_dir}pruned.txt', data_ori)
    
    if verbose: print('Start data analysis')
    do_data_analysis(data_rdkit, FP_DESCRIPTORS['names'], save_dir)
    
    # draw top scaffolds
    if verbose: print('Start drawing scaffolds')
    top_common = 20
    draw_scaffolds(top_common, save_dir)
    
    if verbose: print('Start data processing')
    # define index for the tr-val split
    # and shuffle them
    all_idx = np.arange(len(data_ori))
    idx_split = int(split*len(all_idx))
    np.random.shuffle(all_idx)
    
    # we need to be careful about the case where
    # idx_split = 0 when there is only one 
    # SMILES in the data, e.g. for fine-tuning
    if idx_split==0:
        # in this case, we use the unique smile both  
        # for the training and validation
        idx_tr_canon = [0]
        idx_val_canon = [0]
    else:
        idx_tr_canon = all_idx[:idx_split]
        idx_val_canon = all_idx[idx_split:]
        
    assert len(idx_tr_canon)!=0
    assert len(idx_val_canon)!=0
    
    if verbose:
        print(f'Size of the training set after split: {len(idx_tr_canon)}')
        print(f'Size of the validation set after split: {len(idx_val_canon)}')
    
    d = dict(enumerate(data_ori))
    data_tr = [d.get(item) for item in idx_tr_canon]
    data_val = [d.get(item) for item in idx_val_canon]
    hp_write_in_file(f'{save_dir}data_tr.txt', data_tr)
    hp_write_in_file(f'{save_dir}data_val.txt', data_val)
    
    if augmentation>0:
        if verbose:
            print(f'Data augmentation {augmentation}-fold start')
        

        # Augment separately the training and validation splits
        # It's important to do those steps separetely in order
        # to avoid to have the same molecule represented in 
        # both splits
        tr_aug = augment_dataset(data_tr, augmentation, min_len, max_len, verbose=False) 
        val_aug = augment_dataset(data_val, augmentation, min_len, max_len, verbose=False) 
        
        # Merge with the original data and shuffle
        full_training_set = list(set(data_tr + tr_aug))
        shuffle(full_training_set)
        full_validation_set = list(set(data_val + val_aug))
        shuffle(full_validation_set)
        full_datalist = full_training_set + full_validation_set
                
        if verbose:
            print(f'Size of the training set after agumentation: {len(full_training_set)}')
            print(f'Size of the validation set after agumentation: {len(full_validation_set)}')
                    
        # Create the partitions for the data generators 
        # with the full augmented dataset
        idx_tr = np.arange(len(full_training_set))
        idx_val = np.arange(len(full_training_set), len(full_training_set) + len(full_validation_set))
    
        # Save
        hp_write_in_file(f'{save_dir}.txt', full_datalist)
        hp_save_obj(list(idx_tr), save_dir + 'idx_tr')
        hp_save_obj(list(idx_val), save_dir + 'idx_val')
    else:
        # Save
        hp_write_in_file(f'{save_dir}.txt', data_ori)
        hp_save_obj(list(idx_tr_canon), f'{save_dir}idx_tr')
        hp_save_obj(list(idx_val_canon), f'{save_dir}idx_val')

        
if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    # args = vars(parser.parse_args())
    
    # verbose = args['verbose']
    verbose = True
    # config = configparser.ConfigParser()
    # config.read('parameters.ini')
    
    # get back the experiment parameters
    # min_len = int(config['PROCESSING']['min_len'])
    min_len = 1
    # max_len = int(config['PROCESSING']['max_len'])
    max_len = 140
    # split = float(config['PROCESSING']['split'])
    split = 0.8
    # mode = config['EXPERIMENTS']['mode']
    mode = 'fine_tuning'
    
    # check if experiment mode exists
    if mode not in ['training', 'fine_tuning']:
        raise ValueError('The mode you picked does not exist. Available: training and fine_tuning')
    if verbose: 
        print('\nSTART PROCESSING')
        print(f'Experiment mode: {mode}')

    print('Current data being processed...')
    full_data_path = f'./dataset_sampled.smi'
    
    # define saving path
    # experiment parameters depending on the mode
    aug = 10

    do_processing(split, full_data_path, aug, min_len, max_len, './rnn.txt', verbose=verbose)
                
    end = time.time()
    print(f'PROCESSING DONE in {end - start:.04} seconds') 
    ####################################
    
    