import pandas as pd
# from typing import Callable
from torch_geometric.data import Dataset
import torch
import lmdb
import os
import pickle
print(os.getcwd())

from utils.utils import construct_data, construct_data_new


class BindDataset(Dataset):
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
        # self.af2_protein_dict = lmdb.open(self.processed_paths[1], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.af2_protein_dict = lmdb.open(self.processed_paths[2], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

        self.compound_dict = lmdb.open(self.processed_paths[3], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.compound_rdkit_coords = torch.load(self.processed_paths[4])
        self.protein_esm2_feat = lmdb.open(self.processed_paths[5], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

        self.includeDisMap = includeDisMap
        self.add_noise_to_com = args.addNoise
        self.pocket_idx_no_noise = args.pocket_idx_no_noise

        if args.use_p2rank:
            p2rank_predict_path = "/home/workspace/p2rank_2.3/af2_results"
            files_in_directory = os.listdir(p2rank_predict_path)
            p2rank_predictions_files = [file for file in files_in_directory if file.endswith("_predictions.csv")]
            print(f"P2Rank files: {len(p2rank_predictions_files)}")

            p2rank_center = {}
            count_fail = 0
            for file_name in p2rank_predictions_files:
                pdb = file_name[4:8]
                try:
                    df = pd.read_csv(os.path.join(p2rank_predict_path, file_name))
                    center_x = df.at[0, '   center_x']
                    center_y = df.at[0, '   center_y']
                    center_z = df.at[0, '   center_z']
                    p2rank_center[pdb] = [center_x, center_y, center_z]
                except:
                    print(f"failed p2rank file: {file_name}")
                    count_fail += 1
            print(f"processed pdb: {len(p2rank_center)}")
            print(f'After P2Rank: Failed {count_fail}')
            self.p2rank_center = p2rank_center

        if args.cut_train_set:
            protein_length_dict = {}
            compound_length_dict = {}
            for idx in range(len(self.data)):
                line = self.data.iloc[idx]
                pdb = line['pdb'] # pdb id
                with self.protein_dict.begin() as txn:
                    _, protein_seq = pickle.loads(txn.get(pdb.encode()))
                protein_length_dict[idx] = len(protein_seq)
                with self.compound_dict.begin() as txn:
                    coords, _, _, _, _, _ = pickle.loads(txn.get(pdb.encode()))
                compound_length_dict[idx] = coords.shape[0]

            data_dict = self.data.to_dict(orient='dict')
            data_dict.update({'protein_length': protein_length_dict})
            data_dict.update({'compound_length': compound_length_dict})
            self.data = pd.DataFrame(data_dict)

    def len(self):
        return len(self.data)
    
    @property
    def processed_file_names(self):
        # return ['d3_with_clash_info.csv',
        #         'protein_1d_3d.lmdb', 'af2_protein_1d_3d.lmdb',
        #         'ligand_LAS_edge_index.lmdb', 'ligand_rdkit_coords.pt',
        #         'esm2_t33_650M_UR50D.lmdb']
        return ['data_new.pt',
                'protein_1d_3d.lmdb', 'af2_protein_1d_3d.lmdb',
                'ligand_LAS_edge_index.lmdb', 'ligand_rdkit_coords.pt',
                'esm2_t33_650M_UR50D.lmdb']

    def get(self, idx):
        line = self.data.iloc[idx]
        group = line['group']
        pdb = line['pdb'] # pdb id
        isomorphisms = line['isomorphics']
        if self.args.use_p2rank:
            p2rank_center = self.p2rank_center[pdb]
        else:
            p2rank_center = None

        with self.protein_dict.begin() as txn:
            protein_node_xyz, protein_seq = pickle.loads(txn.get(pdb.encode()))
        # with self.old_protein_dict.begin() as txn:
        #     old_protein_node_xyz, old_protein_seq = pickle.loads(txn.get(pdb.encode()))
        with self.af2_protein_dict.begin() as txn:
            af2_protein_node_xyz, af2_protein_seq = pickle.loads(txn.get(pdb.encode()))
        with self.protein_esm2_feat.begin() as txn:
            protein_esm2_feat = pickle.loads(txn.get(pdb.encode()))

        compound_name = line['compoundName']   # 实际 compoundName == pdb 就是唯一索引 protein, ligand, complex
        # compound embedding from torchdrug
        with self.compound_dict.begin() as txn:
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution, LAS_edge_index = \
                pickle.loads(txn.get(compound_name.encode()))
        rdkit_coords = self.compound_rdkit_coords[compound_name]

        data = construct_data(self.args, self.flag, group,
                              protein_node_xyz, protein_seq, af2_protein_node_xyz, af2_protein_seq, protein_esm2_feat,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              LAS_edge_index, rdkit_coords, add_noise_to_com=self.add_noise_to_com,
                              includeDisMap=self.includeDisMap, pocket_idx_no_noise=self.pocket_idx_no_noise,
                              isomorphisms=isomorphisms, p2rank_center=p2rank_center)

        data.pdb = pdb
        data.group = group

        return data

    # def get(self, idx):
    #     line = self.data.iloc[idx]
    #     group = line['group']
    #     pdb = line['pdb']  # pdb id
    #     isomorphisms = line['isomorphics']
    #
    #     if self.flag == 1:
    #         with self.protein_dict.begin() as txn:
    #             protein_node_xyz, protein_seq = pickle.loads(txn.get(pdb.encode()))
    #     else:   # self.flag == 1 or self.flag == 2
    #         with self.af2_protein_dict.begin() as txn:
    #             protein_node_xyz, protein_seq = pickle.loads(txn.get(pdb.encode()))
    #
    #     with self.protein_esm2_feat.begin() as txn:
    #         protein_esm2_feat = pickle.loads(txn.get(pdb.encode()))
    #
    #     compound_name = line['compoundName']  # 实际 compoundName == pdb 就是唯一索引 protein, ligand, complex
    #     # compound embedding from torchdrug
    #     with self.compound_dict.begin() as txn:
    #         coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution, LAS_edge_index = \
    #             pickle.loads(txn.get(compound_name.encode()))
    #     rdkit_coords = self.compound_rdkit_coords[compound_name]
    #
    #     data = construct_data_new(self.args, self.flag, group,
    #                               protein_node_xyz, protein_seq, protein_esm2_feat,
    #                               coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
    #                               LAS_edge_index, rdkit_coords, add_noise_to_com=self.add_noise_to_com,
    #                               includeDisMap=self.includeDisMap, pocket_idx_no_noise=self.pocket_idx_no_noise,
    #                               isomorphisms=isomorphisms)
    #
    #     data.pdb = pdb
    #     data.group = group
    #
    #     return data



def get_data(args, logger, flag=1):
    logger.log_message(f"Loading dataset")
    logger.log_message(f"compound feature based on torchdrug")
    logger.log_message(f"protein feature based on esm2")
    dataset = BindDataset(root=args.root_path, flag=flag, args=args, includeDisMap=args.includeDisMap,)

    if args.cut_train_set:
        train_tmp = dataset.data.query("protein_length < 1500 and compound_length < 150 and group == 'train'").reset_index(drop=True)
    else:
        train_tmp = dataset.data.query("group == 'train'").reset_index(drop=True)
    valid_tmp = dataset.data.query("group == 'valid'").reset_index(drop=True)
    test_tmp = dataset.data.query("group == 'test' and fraction_of_this_chain > 0.8").reset_index(drop=True)
    dataset.data = pd.concat([train_tmp, valid_tmp, test_tmp], axis=0).reset_index(drop=True)
    n_dataset = len(dataset)
    logger.log_message(f"Number of dataset: {n_dataset}")

    train_index = dataset.data.query("group == 'train'").index.values
    train = dataset[train_index]
    valid_index = dataset.data.query("group == 'valid'").index.values
    valid = dataset[valid_index]
    if args.infer_unseen:
        with open(args.unseen_file, 'r') as f:
            content = f.read()
        unseen_pdbs = content.strip('\n').split(' ')
        # 筛选出在 unseen_pdbs 中的 test 数据
        test_index = dataset.data[(dataset.data['group'] == 'test') & (dataset.data['pdb'].isin(unseen_pdbs))].index.values
    else:
        if args.use_p2rank:
            test_index = dataset.data[(dataset.data['group'] == 'test') &
                                      (~dataset.data['pdb'].isin(['6d07', '6d08']))].index.values
        else:
            test_index = dataset.data.query("group == 'test'").index.values
    test = dataset[test_index]
    return train, valid, test


if __name__ == "__main__":
    root_path = "./binddataset"
    data = BindDataset(root=root_path)


