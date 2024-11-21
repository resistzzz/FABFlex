import os.path

import torch
from tqdm import tqdm
from torch_scatter import scatter_mean
import numpy as np
import copy
import time
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.metrics import pocket_metrics, pocket_metrics_each_sample
from utils.loss_utils import compute_permutation_loss

from Bio.PDB import PDBParser, PDBIO
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

@torch.no_grad()
def inference_one(accelerator, args, epoch, data_loader, model, device, prefix="test", stage=1, flag=2):
    batch_count = 0
    count = 0
    skip_count = 0
    keepNode_less_5_count = 0
    gt_lig_coord_list, lig_coord_pred_list = [], []
    gt_poc_coord_list, poc_coord_pred_list = [], []

    pdb_list = []
    infer_time_list = []
    rmsd_list = []
    rmsd_2A_list, rmsd_5A_list = [], []
    pocket_rmsd_list = []
    pocket_rmsd_05A_list, pocket_rmsd_1A_list = [], []
    af2_pocket_rmsd_list = []
    af2_pocket_rmsd_05A_list, af2_pocket_rmsd_1A_list = [], []

    pocket_cls_list, pocket_cls_pred_list, pocket_cls_pred_round_list = [], [], []
    pocket_center_list, pocket_center_pred_list = [], []

    out_protein_coords = []
    ref_protein_coords = []
    init_protein_coords = []
    offset_coords = []

    out_pocket_coords = []
    ref_pocket_coords = []
    pred_keepNode_list, gt_keepNode_list = [], []

    out_ligand_coords = []
    ref_ligand_coords = []
    init_ligand_coords = []

    protein_len_list = []
    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    for data in data_iter:

        pdb_list = pdb_list + data.pdb
        data = data.to(device)

        t0 = time.time()
        (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
         pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
         pred_pocket_center, pocket_radius_pred, gt_pocket_batch) = model.inference(data, stage=stage)

        t1 = time.time()
        delta_time = t1 - t0
        infer_time_list.append(delta_time)

        gt_ligand_coord = data.coords
        gt_pocket_coord = data.protein_node_xyz[data['pocket'].keepNode]

        # evaluate metrics
        sd = ((compound_coords_pred.detach() - gt_ligand_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(src=sd, index=compound_batch, dim=0).sqrt().detach()

        protein_batch = data['protein_whole'].batch
        for i in range(compound_batch.max() + 1):
            init_protein_coords.append(data.input_protein_node_xyz[protein_batch == i].clone())
            init_ligand_coords.append(data['compound'].node_coords[compound_batch == i])

        data.input_protein_node_xyz[data['pocket'].keepNode] = pocket_coords_pred
        for i in range(compound_batch.max() + 1):
            out_protein_coords.append(data.input_protein_node_xyz[protein_batch == i])
            ref_protein_coords.append(data.protein_node_xyz[protein_batch == i])
            offset_coords.append(data.coord_offset[i])

            out_pocket_coords.append(pocket_coords_pred[pocket_batch == i])
            ref_pocket_coords.append(gt_pocket_coord[pocket_batch == i])
            pred_keepNode_list.append(data['pocket'].keepNode[protein_batch == i].detach().cpu().numpy().tolist())
            gt_keepNode_list.append(data['pocket'].keepNode_noNoise[protein_batch == i].detach().cpu().numpy().tolist())

            out_ligand_coords.append(compound_coords_pred[compound_batch == i])
            ref_ligand_coords.append(data.coords[compound_batch == i])

        if args.use_p2rank:
            tmp_pocket_batch = []
            for i in range(compound_batch.max() + 1):
                tmp_num = data['pocket'].keepNode_noNoise[protein_batch == i].sum().item()
                tmp_pocket_batch.extend([i] * tmp_num)
            gt_pocket_batch = torch.tensor(tmp_pocket_batch, device=compound_batch.device)
            gt_pocket_batch = gt_pocket_batch.long()

        pocket_sd = ((data.input_protein_node_xyz[data['pocket'].keepNode_noNoise] - data.protein_node_xyz[data['pocket'].keepNode_noNoise]) ** 2).sum(dim=-1)
        pocket_rmsd = scatter_mean(src=pocket_sd, index=gt_pocket_batch, dim=0).sqrt().detach()

        # af2_pocket_sd = ((data.protein_node_xyz[data['pocket'].keepNode_noNoise] - data.af2_pocket_node_xyz) ** 2).sum(dim=-1)
        af2_pocket_sd = ((data.protein_node_xyz[data['pocket'].keepNode_noNoise] - data.af2_protein_node_xyz[data['pocket'].keepNode_noNoise]) ** 2).sum(dim=-1)
        af2_pocket_rmsd = scatter_mean(src=af2_pocket_sd, index=gt_pocket_batch, dim=0).sqrt().detach()

        # record current loss value
        batch_count += 1

        # record prediction result
        gt_lig_coord_list.append(gt_ligand_coord.detach())
        lig_coord_pred_list.append(compound_coords_pred.detach())
        gt_poc_coord_list.append(gt_pocket_coord.detach())
        poc_coord_pred_list.append(pocket_coords_pred.detach())
        pocket_center_list.append(data.coords_center.detach())
        pocket_center_pred_list.append(pred_pocket_center.detach())

        # record evaluation metrics
        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())

        pocket_rmsd_list.append(pocket_rmsd.detach())
        pocket_rmsd_05A_list.append((pocket_rmsd.detach() < 0.5).float())
        pocket_rmsd_1A_list.append((pocket_rmsd.detach() < 1).float())

        af2_pocket_rmsd_list.append(af2_pocket_rmsd.detach())
        af2_pocket_rmsd_05A_list.append((af2_pocket_rmsd.detach() < 0.5).float())
        af2_pocket_rmsd_1A_list.append((af2_pocket_rmsd.detach() < 1).float())

        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        protein_len_list.append(batch_len)
        for i, j in enumerate(batch_len):
            count += 1
            pocket_cls_list.append(pocket_cls.detach()[i][:j])
            pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid())
            pocket_cls_pred_round_list.append(pocket_cls_pred.detach()[i][:j].sigmoid().round().int())
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() == 0:
                skip_count += 1

    rmsd = torch.cat(rmsd_list)
    rmsd_2A = torch.cat(rmsd_2A_list)
    rmsd_5A = torch.cat(rmsd_5A_list)
    rmsd_25 = torch.quantile(rmsd, 0.25)
    rmsd_50 = torch.quantile(rmsd, 0.50)
    rmsd_75 = torch.quantile(rmsd, 0.75)

    pocket_rmsd = torch.cat(pocket_rmsd_list)
    pocket_rmsd_05A = torch.cat(pocket_rmsd_05A_list)
    pocket_rmsd_1A = torch.cat(pocket_rmsd_1A_list)
    pocket_rmsd_25 = torch.quantile(pocket_rmsd, 0.25)
    pocket_rmsd_50 = torch.quantile(pocket_rmsd, 0.50)
    pocket_rmsd_75 = torch.quantile(pocket_rmsd, 0.75)

    af2_pocket_rmsd = torch.cat(af2_pocket_rmsd_list)
    af2_pocket_rmsd_05A = torch.cat(af2_pocket_rmsd_05A_list)
    af2_pocket_rmsd_1A = torch.cat(af2_pocket_rmsd_1A_list)
    af2_pocket_rmsd_25 = torch.quantile(af2_pocket_rmsd, 0.25)
    af2_pocket_rmsd_50 = torch.quantile(af2_pocket_rmsd, 0.50)
    af2_pocket_rmsd_75 = torch.quantile(af2_pocket_rmsd, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_center = torch.cat(pocket_center_list)
    pocket_center_pred = torch.cat(pocket_center_pred_list)
    pocket_coords_pred = torch.cat(poc_coord_pred_list)

    protein_len = torch.cat(protein_len_list)
    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / pocket_cls_pred_round.shape[0]
    cls_acc = []
    for i in range(len(pocket_cls_list)):
        acc = (pocket_cls_list[i] == pocket_cls_pred_round_list[i]).sum().item() / len(pocket_cls_list[i])
        cls_acc.append(acc)
    pocket_cls_accuracy_1 = np.mean(cls_acc)

    metrics = {
        f"{prefix}/epoch": epoch, f"{prefix}/samples": count, f"{prefix}/skip_samples": skip_count,
        f"{prefix}/avg_infer_time": sum(infer_time_list)/len(infer_time_list)
    }

    eval_dict = {
        f"{prefix}/rmsd": rmsd.mean().item(), f"{prefix}/rmsd < 2A": rmsd_2A.mean().item(),
        f"{prefix}/rmsd < 5A": rmsd_5A.mean().item(),
        f"{prefix}/rmsd 25%": rmsd_25.item(), f"{prefix}/rmsd 50%": rmsd_50.item(),
        f"{prefix}/rmsd 75%": rmsd_75.item(),
        f"{prefix}/pocket_rmsd": pocket_rmsd.mean().item(), f"{prefix}/pocket_rmsd < 1A": pocket_rmsd_1A.mean().item(),
        f"{prefix}/pocket_rmsd < 0.5A": pocket_rmsd_05A.mean().item(),
        f"{prefix}/pocket_rmsd 25%": pocket_rmsd_25.item(), f"{prefix}/pocket_rmsd 50%": pocket_rmsd_50.item(),
        f"{prefix}/pocket_rmsd 75%": pocket_rmsd_75.item(),
        f"{prefix}/af2_pocket_rmsd": af2_pocket_rmsd.mean().item(), f"{prefix}/af2_pocket_rmsd < 1A": af2_pocket_rmsd_1A.mean().item(),
        f"{prefix}/af2_pocket_rmsd < 0.5A": af2_pocket_rmsd_05A.mean().item(),
        f"{prefix}/af2_pocket_rmsd 25%": af2_pocket_rmsd_25.item(), f"{prefix}/af2_pocket_rmsd 50%": af2_pocket_rmsd_50.item(),
        f"{prefix}/af2_pocket_rmsd 75%": af2_pocket_rmsd_75.item()
    }
    metrics.update(eval_dict)
    metrics.update({f"{prefix}/pocket_cls_accuracy": pocket_cls_accuracy_1})
    # pocket_eval_dict = pocket_metrics(pocket_center_pred, pocket_center, prefix=prefix)
    pocket_eval_dict = pocket_metrics_each_sample(pocket_center_pred, pocket_center, prefix=prefix)
    metrics.update(pocket_eval_dict)

    # if args.save_rmsd:
    #     with (open(args.rmsd_file, "w") as f):
    #         f.write("pdb\tLigand_RMSD\tPocket_RMSD\tAf2_Pocket_RMSD\tMAE\tPocket_RMSE\tPocket_dis\tPocket_cls_acc\n")
    #         for i in range(len(pdb_list)):
    #             pdb = pdb_list[i]
    #             rmsd_value = rmsd[i].item()
    #             pocket_rmsd_value = pocket_rmsd[i].item()
    #             af2_pocket_rmsd_value = af2_pocket_rmsd[i].item()
    #             mae_value = metrics[f'{prefix}/pocket_mae'].item()
    #             pocket_center_dis = metrics[f'{prefix}/pocket_center_dist'][i]
    #             cls_acc_value = cls_acc[i]
    #             rmse_value = metrics[f'{prefix}/pocket_rmse'].item()
    #             out = str(pdb) + "\t" + str(rmsd_value) + "\t" + str(pocket_rmsd_value) + "\t" + str(af2_pocket_rmsd_value) + "\t" + \
    #                   str(mae_value) + "\t" + str(rmse_value) + '\t' + str(pocket_center_dis) + "\t" + str(cls_acc_value) + "\n"
    #             f.write(out)
    #         count_less = 0
    #         for i in range(len(pdb_list)):
    #             pocket_rmsd_value = pocket_rmsd[i].item()
    #             af2_pocket_rmsd_value = af2_pocket_rmsd[i].item()
    #             if pocket_rmsd_value < af2_pocket_rmsd_value:
    #                 count_less += 1
    #         less_ratio = count_less / len(pdb_list)
    #         print(f'less ratio: {less_ratio}')
    #         f.write(f"Ligand RMSD: {rmsd.mean().item()}, Pocket RMSD: {pocket_rmsd.mean().item()}, Af2 Pocket RMSD: {af2_pocket_rmsd.mean().item()}, "
    #                 f"MAE: {metrics[f'{prefix}/pocket_mae']}, RMSE: {metrics[f'{prefix}/pocket_rmse']},  Pocket_center_avg_dis: {metrics[f'{prefix}/pocket_center_avg_dist']}, "
    #                 f"DCC<3: {metrics[f'{prefix}/DCC_3']}, DCC<4: {metrics[f'{prefix}/DCC_4']}, DCC<5: {metrics[f'{prefix}/DCC_5']}, "
    #                 f"pocket_cls_accuracy: {pocket_cls_accuracy_1}, less_ratio: {less_ratio}")

    # return (metrics, pdb_list, out_protein_coords, ref_protein_coords, init_protein_coords, offset_coords,
    #         out_ligand_coords, ref_ligand_coords, init_ligand_coords,
    #         out_pocket_coords, ref_pocket_coords, pred_keepNode_list, gt_keepNode_list)
    return (metrics, pdb_list, rmsd, pocket_rmsd, af2_pocket_rmsd, cls_acc, pred_keepNode_list, gt_keepNode_list)


def write_pdb_file(pdb_list, ori_pdb_files, pred_coords, ref_coords, init_coords, offset_coords, out_path):
    parser = PDBParser()
    # io = PDBIO()
    for pdb, pred_coo, ref_coo, init_coo, off_coo, pdb_file in zip(pdb_list, pred_coords, ref_coords, init_coords, offset_coords, ori_pdb_files):
        structure = parser.get_structure(pdb, pdb_file)
        io_pred = PDBIO()
        io_ref = PDBIO()
        io_init = PDBIO()
        if not os.path.exists(f"{out_path}/{pdb}"):
            os.mkdir(f"{out_path}/{pdb}")

        for coord_set, io_obj, suffix in zip([pred_coo, ref_coo, init_coo], [io_pred, io_ref, io_init], ["docked", "ref", "init"]):
            ca_index = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        residue_id = residue.get_id()[1]
                        # 仅处理Cα原子
                        ca_atom = residue['CA'] if 'CA' in residue else None
                        if ca_atom:
                            # if ca_index < len(pred_coo):
                            # 更新Cα原子的坐标为预测坐标
                            new_coord = coord_set[ca_index]
                            ca_atom.set_coord(new_coord)
                            ca_index += 1

                        # 对所有其他原子减去偏移量
                        if suffix == 'docked':
                            for atom in residue:
                                if atom.name != 'CA':  # 避免再次更新Cα原子
                                    atom_coord = atom.get_coord()
                                    adjusted_coord = atom_coord - off_coo
                                    atom.set_coord(adjusted_coord)

            io_obj.set_structure(structure)
            out_pdb_file = f"{out_path}/{pdb}/{pdb}_protein_{suffix}.pdb"
            io_obj.save(out_pdb_file)
            print(f"{suffix.capitalize()} PDB file saved as: {out_pdb_file}")


def write_single_pdb_file(pdb_list, ori_pdb_files, pred_coords, offset_coords, out_path):
    parser = PDBParser()
    # io = PDBIO()
    for pdb, pred_coo, off_coo, pdb_file in zip(pdb_list, pred_coords, offset_coords, ori_pdb_files):
        structure = parser.get_structure(pdb, pdb_file)
        io_pred = PDBIO()
        if not os.path.exists(f"{out_path}/{pdb}"):
            os.mkdir(f"{out_path}/{pdb}")

        ca_index = 0
        pred_coo = pred_coo + off_coo
        for model in structure:
            for chain in model:
                for residue in chain:
                    residue_id = residue.get_id()[1]
                    # 仅处理Cα原子
                    ca_atom = residue['CA'] if 'CA' in residue else None
                    if ca_atom:
                        # 更新Cα原子的坐标为预测坐标
                        old_coord = ca_atom.get_coord()
                        new_coord = pred_coo[ca_index]
                        # 计算位移
                        displacement = new_coord - old_coord
                        for atom in residue:
                            atom.set_coord(atom.get_coord() + displacement)
                        ca_index += 1

            io_pred.set_structure(structure)
            out_pdb_file = f"{out_path}/{pdb}/{pdb}_protein_docked.pdb"
            io_pred.save(out_pdb_file)
            print(f"Docked PDB file saved as: {out_pdb_file}")


def write_sdf_file(pdb_list, ori_sdf_files, pred_coords, ref_coords, init_coords, offset_coords, out_path):
    for pdb, sdf_file, pred_coo, ref_coo, init_coo, off_coo in zip(pdb_list, ori_sdf_files, pred_coords, ref_coords, init_coords, offset_coords):
        suppl = Chem.SDMolSupplier(sdf_file)
        mols = [mol for mol in suppl if mol is not None]

        mol = mols[0]

        conf = mol.GetConformer()
        original_coo = [list(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
        # original_coo = np.array(original_coo) - off_coo
        original_coo = np.array(original_coo)
        ref_coo = ref_coo + off_coo
        pred_coo = pred_coo + off_coo
        if not np.allclose(ref_coo, original_coo, atol=1e-3):
            print(f"error: {pdb}_sdf")

        if not os.path.exists(f"{out_path}/{pdb}"):
            os.mkdir(f"{out_path}/{pdb}")

        for coords, suffix in zip([pred_coo, ref_coo, init_coo], ['docked', 'ref', 'init']):
            for atom_idx, (x, y, z) in enumerate(coords):
                if atom_idx < mol.GetNumAtoms():
                    conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
            output_sdf_path = f"{out_path}/{pdb}/{pdb}_ligand_{suffix}.sdf"
            writer = Chem.SDWriter(output_sdf_path)
            writer.write(mol)
            writer.close()
            # print(f"{suffix.capitalize()} SDF file saved as: {output_sdf_path}")


def write_pocket_info(pdb_list, pred_keepNode_list, gt_keepNode_list, out_path):
    for pdb, pred_keep, gt_keep in zip(pdb_list, pred_keepNode_list, gt_keepNode_list):
        out_txt_path = f"{out_path}/{pdb}/{pdb}_keepNode.txt"
        if not os.path.exists(f"{out_path}/{pdb}"):
            os.mkdir(f"{out_path}/{pdb}")
        with open(out_txt_path, 'w') as f:
            for i in range(len(pred_keep)):
                pk = str(int(pred_keep[i]))
                gk = str(int(gt_keep[i]))
                f.write(pk + "\t" + gk + '\n')
        print(f"Write KeepNode file saved as: {out_txt_path}")



