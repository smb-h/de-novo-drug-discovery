import pickle
import time

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def hp_chem_get_rdkit_desc_functions(desc_names):
    """
    Allows to define RDKit Descriptors by regex wildcards.
    :return: Descriptor as functions for calculations and names.
    """
    functions = []
    descriptors = []

    for descriptor, function in Descriptors._descList:
        if desc_names.match(descriptor) != None:
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
        print(
            f"{len(functions)} descriptors for {len(mols)} mols were calculated in {time.time()-start:.03} seconds."
        )

    return descriptors


def hp_chem_fingerprint_calc(mols, verbose=True):
    """
    Calculate Morgan fingerprints (circular fingerprint) for set of molecules
    :param mols: input dataset in rdkit mol format.
    :return: Morgan fingerprint for each mol.
    """
    start = time.time()
    morganfp = [
        Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False) for mol in mols
    ]
    if verbose:
        print(
            f"Fingerprint calculation for {len(mols)} mols took {time.time()-start:.03} seconds."
        )
    return morganfp


def hp_chem_extract_murcko_scaffolds(mols, verbose=True):
    """Extract Bemis-Murcko scaffolds from a smile string.

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
            scaf.append(["error"])
            generic_scaf.append(["error"])
    if verbose:
        print("Extracted", len(scaf), "scaffolds in", time.time() - start, "seconds.")
    return scaf, generic_scaf


def hp_save_obj(obj, name):
    """save obj with pickle"""
    name = name.replace(".pkl", "")
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def hp_write_in_file(path_to_file, data):
    with open(path_to_file, "w+") as f:
        for item in data:
            f.write("%s\n" % item)


def hp_load_obj(name):
    """load a pickle object"""
    name = name.replace(".pkl", "")
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)
