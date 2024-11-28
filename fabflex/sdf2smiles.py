from rdkit import Chem
from rdkit.Chem import AllChem


def sdf_to_smiles(sdf_file_path):
    sdf_supplier = Chem.SDMolSupplier(sdf_file_path)
    smiles_list = []

    for mol in sdf_supplier:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)

    return smiles_list


sdf_file_path = '/home/workspace/test_ligand/6rot.sdf'
smiles = sdf_to_smiles(sdf_file_path)
for s in smiles:
    print(s)
