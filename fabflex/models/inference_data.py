
import pandas as pd
# from typing import Callable
from torch_geometric.data import Dataset
import torch
import lmdb
import os
import pickle
from torch_geometric.data import HeteroData
import numpy as np
import scipy
from scipy.spatial.distance import cdist
print(os.getcwd())

class InferenceDataset(Dataset):
    def __init__(self, root, flag=1, args=None, includeDisMap=True,
                 transform=None, pre_transform=None, pre_filter=None):
        '''
            data <- d3_with_clash_info.csv
            protein_dict <- protein_1d_3d.lmdb
            af2_protein_dict <- af2_protein_1d_3d.lmdb
            compound_dict <- ligand_LAS_edge_index.lmdb
            compound_rdkit_coords <- ligand_rdkit_coords.pt
            protein_esm2_feat <- esm2_t33_650M_UR50D.lmdb
        '''
        super().__init__(root, transform, pre_transform, pre_filter)
        self.args = args
        self.flag = flag

        # data = pd.read_csv(self.processed_paths[0], sep=",")
        data = torch.load(self.processed_paths[0])
        self.data = data.query("group == 'train' or group == 'valid' or group == 'test'").reset_index(drop=True)
        self.protein_dict = lmdb.open(self.processed_paths[1], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.af2_protein_dict = lmdb.open(self.processed_paths[2], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

        self.compound_dict = lmdb.open(self.processed_paths[3], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.compound_rdkit_coords = torch.load(self.processed_paths[4])
        self.protein_esm2_feat = lmdb.open(self.processed_paths[5], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

        self.includeDisMap = includeDisMap
        self.add_noise_to_com = args.addNoise
        self.pocket_idx_no_noise = args.pocket_idx_no_noise

    def len(self):
        return len(self.data)

    @property
    def processed_file_names(self):
        return ['data_new.pt',
                'protein_1d_3d.lmdb', 'af2_protein_1d_3d.lmdb',
                'ligand_LAS_edge_index.lmdb', 'ligand_rdkit_coords.pt',
                'esm2_t33_650M_UR50D.lmdb']

    def get(self, idx):
        line = self.data.iloc[idx]
        group = line['group']
        pdb = line['pdb'] # pdb id
        isomorphisms = line['isomorphics']

        with self.protein_dict.begin() as txn:
            protein_node_xyz, protein_seq = pickle.loads(txn.get(pdb.encode()))
        with self.af2_protein_dict.begin() as txn:
            af2_protein_node_xyz, af2_protein_seq = pickle.loads(txn.get(pdb.encode()))
        with self.protein_esm2_feat.begin() as txn:
            protein_esm2_feat = pickle.loads(txn.get(pdb.encode()))

        compound_name = line['compoundName']   # 实际 compoundName == pdb 就是唯一索引 protein, ligand, complex
        with self.compound_dict.begin() as txn:
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution, LAS_edge_index = \
                pickle.loads(txn.get(compound_name.encode()))
        rdkit_coords = self.compound_rdkit_coords[compound_name]

        data = construct_data(self.args, self.flag, group,
                              protein_node_xyz, protein_seq, af2_protein_node_xyz, af2_protein_seq, protein_esm2_feat,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              LAS_edge_index, rdkit_coords, add_noise_to_com=self.add_noise_to_com,
                              includeDisMap=self.includeDisMap, pocket_idx_no_noise=self.pocket_idx_no_noise,
                              isomorphisms=isomorphisms)

        data.pdb = pdb
        data.group = group

        return data


def compute_dis_between_two_vector(a, b):
    return (((a - b)**2).sum())**0.5


def get_keepNode(com, n_protein_node, protein_node_xyz, pocket_radius, add_noise_to_com):
    keepNode = np.zeros(n_protein_node, dtype=bool)
    if add_noise_to_com:
        com = com + add_noise_to_com * (2 * np.random.rand(*com.shape) - 1)
    for i, node in enumerate(protein_node_xyz):
        dis = compute_dis_between_two_vector(node, com)
        keepNode[i] = dis < pocket_radius
    return keepNode


def construct_data(args, flag, group, protein_node_xyz, protein_seq, af2_protein_node_xyz, af2_protein_seq,
                   protein_esm2_feat,
                   coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                   LAS_edge_index, rdkit_coords,
                   add_noise_to_com=None, includeDisMap=True, pocket_idx_no_noise=True, isomorphisms=None):
    n_protein = protein_node_xyz.shape[0]
    coords_bias = af2_protein_node_xyz.mean(dim=0)
    coords = coords - coords_bias.numpy()
    protein_node_xyz = protein_node_xyz - coords_bias
    af2_protein_node_xyz = af2_protein_node_xyz - coords_bias

    # centroid instead of com.
    com = coords.mean(axis=0)

    # construct heterogeneous graph data.
    data = HeteroData()
    data.isomorphisms = isomorphisms
    data.seq_whole = protein_seq
    data.coords = torch.tensor(coords, dtype=torch.float)
    data.num_atoms = data.coords.shape[0]
    data.coord_offset = coords_bias.unsqueeze(0)
    data.protein_node_xyz = protein_node_xyz
    data.af2_protein_node_xyz = af2_protein_node_xyz

    # max sphere radius
    coords_tensor = torch.tensor(coords)
    center = coords_tensor.mean(dim=0)
    distances = torch.norm(coords_tensor - center, dim=1)
    data.ligand_radius = distances.max()
    data.coords_center = torch.tensor(com, dtype=torch.float).unsqueeze(0)

    # pocket radius
    if args.pocket_radius_buffer <= 2.0:
        pocket_radius = args.pocket_radius_buffer * data.ligand_radius
    else:
        pocket_radius = args.pocket_radius_buffer + data.ligand_radius
    if pocket_radius < args.min_pocket_radius:
        pocket_radius = args.min_pocket_radius

    if args.force_fix_radius:
        pocket_radius = args.pocket_radius

    keepNode = get_keepNode(com, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=add_noise_to_com)
    keepNode_no_noise = get_keepNode(com, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=None)
    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True
    data.pocket_idx = torch.tensor(keepNode_no_noise, dtype=torch.int)
    data.pocket_node_xyz = protein_node_xyz[keepNode]
    data.af2_pocket_node_xyz = af2_protein_node_xyz[keepNode]
    # protein feature map
    if torch.is_tensor(protein_esm2_feat):
        data['protein_whole'].node_feats = protein_esm2_feat
    else:
        raise ValueError("protein_esm2_feat should be a tensor")

    # distance map
    dis_map = scipy.spatial.distance.cdist(protein_node_xyz[keepNode].cpu().numpy(), coords)
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map > args.dis_map_thres] = args.dis_map_thres
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()

    # pocket feature map
    if torch.is_tensor(protein_esm2_feat):
        data['pocket'].node_feats = protein_esm2_feat[keepNode]
    else:
        raise ValueError("protein_esm2_feat should be a tensor")
    data['pocket'].keepNode = torch.tensor(keepNode, dtype=torch.bool)
    data['pocket'].keepNode_noNoise = torch.tensor(keepNode_no_noise, dtype=torch.bool)
    data['compound'].node_feats = compound_node_features.float()
    data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index

    n_pocket = protein_node_xyz[keepNode].shape[0]
    n_protein_whole = protein_node_xyz.shape[0]
    n_compound = compound_node_features.shape[0]
    rdkit_coords = torch.tensor(rdkit_coords)
    data['compound'].rdkit_coords = rdkit_coords

    data['complex'].node_coords_LAS = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3),
            torch.zeros_like(protein_node_xyz[keepNode])
        ), dim=0
    ).float()
    segment = torch.zeros(n_compound + n_pocket + 2)
    segment[n_compound + 1:] = 1  # compound: 0, protein: 1
    data['complex'].segment = segment
    lig_mask = torch.zeros(n_compound + n_pocket + 2)
    lig_mask[:n_compound + 2] = 1  # glb_p can be updated
    data['complex'].lig_mask = lig_mask.bool()
    pro_mask = torch.zeros(n_compound + n_pocket + 2)
    pro_mask[0] = 1
    pro_mask[n_compound + 1:] = 1  # glb_c, glb_p can be updated
    data['complex'].pro_mask = pro_mask.bool()
    is_global = torch.zeros(n_compound + n_pocket + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex"].is_global = is_global.bool()

    data['complex', 'c2c', 'complex'].edge_index = input_atom_edge_list[:, :2].long().t().contiguous() + 1
    data['complex', 'LAS', 'complex'].edge_index = LAS_edge_index + 1

    data['complex_whole_protein'].node_coords_LAS = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3),
            torch.zeros_like(protein_node_xyz)
        ), dim=0
    ).float()

    segment = torch.zeros(n_compound + n_protein_whole + 2)
    segment[n_compound + 1:] = 1  # compound: 0, protein: 1
    data['complex_whole_protein'].segment = segment
    lig_mask = torch.zeros(n_compound + n_protein_whole + 2)
    lig_mask[:n_compound + 2] = 1  # glb_p can be updated?
    data['complex_whole_protein'].lig_mask = lig_mask.bool()
    pro_mask = torch.zeros(n_compound + n_protein_whole + 2)
    pro_mask[0] = 1
    pro_mask[n_compound + 1:] = 1
    data['complex_whole_protein'].pro_mask = pro_mask.bool()
    is_global = torch.zeros(n_compound + n_protein_whole + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex_whole_protein"].is_global = is_global.bool()

    data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:,
                                                                               :2].long().t().contiguous() + 1
    data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

    data['compound_atom_edge_list'].x = (input_atom_edge_list[:, :2].long().contiguous() + 1).clone()
    data['LAS_edge_list'].x = data['complex', 'LAS', 'complex'].edge_index.clone().t()

    input_pocket_node_xyz = af2_protein_node_xyz[keepNode]
    data.input_pocket_node_xyz = input_pocket_node_xyz.clone()

    # complex information
    coords_init = rdkit_coords - rdkit_coords.mean(dim=0).reshape(1, 3) + input_pocket_node_xyz.mean(dim=0).reshape(1,
                                                                                                                    3)
    data['complex'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            coords_init,
            torch.zeros(1, 3),
            input_pocket_node_xyz
        ), dim=0
    ).float()

    data['complex_whole_protein'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            coords_init - coords_init.mean(dim=0).reshape(1, 3),
            # for pocket prediction module, the ligand is centered at the protein center/origin
            torch.zeros(1, 3),
            af2_protein_node_xyz
        ), dim=0
    ).float()

    data['compound'].node_coords = coords_init
    data.input_protein_node_xyz = af2_protein_node_xyz

    return data


def get_data(args, logger, flag=1):
    logger.log_message(f"Loading dataset")
    logger.log_message(f"compound feature based on torchdrug")
    logger.log_message(f"protein feature based on esm2")
    dataset = InferenceDataset(root=args.root_path, flag=flag, args=args, includeDisMap=args.includeDisMap)

    test_tmp = dataset.data.query("group == 'test' and fraction_of_this_chain > 0.8").reset_index(drop=True)
    dataset.data = pd.concat([test_tmp], axis=0).reset_index(drop=True)
    test_index = dataset.data.query("group == 'test'").index.values
    test = dataset[test_index]
    n_dataset = len(test)
    logger.log_message(f"Number of test dataset: {n_dataset}")

    return test


if __name__ == "__main__":
    root_path = "./binddataset"
    data = InferenceDataset(root=root_path)





