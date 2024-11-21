from rdkit import Chem
from rdkit.Chem import AllChem


def sdf_to_smiles(sdf_file_path):
    # 读取SDF文件
    sdf_supplier = Chem.SDMolSupplier(sdf_file_path)
    smiles_list = []

    # 遍历所有分子
    for mol in sdf_supplier:
        if mol is not None:  # 确保分子有效
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)

    return smiles_list


# 使用函数
sdf_file_path = '/home/workspace/test_ligand/6rot.sdf'
smiles = sdf_to_smiles(sdf_file_path)
for s in smiles:
    print(s)
