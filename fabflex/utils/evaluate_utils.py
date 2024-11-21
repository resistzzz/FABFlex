
import torch
from tqdm import tqdm
from torch_scatter import scatter_mean
import numpy as np
import copy

from utils.metrics import pocket_metrics
from utils.loss_utils import compute_permutation_loss

@torch.no_grad()
def evaluate_one(accelerator, args, epoch, data_loader, model, device,
                 pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion, contact_criterion,
                 com_coord_criterion, pro_coord_criterion, prefix="test", stage=1, flag=2):
    batch_count = 0
    batch_loss = 0.0
    batch_com_coord_loss, batch_pro_coord_loss = 0.0, 0.0
    batch_contact_loss, batch_contact_by_feat_loss, batch_contact_by_coord_loss, batch_contact_distill_loss = 0.0, 0.0, 0.0, 0.0
    batch_pocket_cls_loss, batch_pocket_center_loss, batch_pocket_radius_loss = 0.0, 0.0, 0.0

    count = 0
    skip_count = 0
    keepNode_less_5_count = 0
    gt_lig_coord_list, lig_coord_pred_list = [], []
    gt_poc_coord_list, poc_coord_pred_list = [], []

    pdb_list = []
    rmsd_list = []
    rmsd_2A_list, rmsd_5A_list = [], []
    centroid_dis_list = []
    centroid_dis_2A_list, centroid_dis_5A_list = [], []
    pocket_rmsd_list = []
    pocket_rmsd_2A_list, pocket_rmsd_5A_list = [], []

    pocket_cls_list, pocket_cls_pred_list, pocket_cls_pred_round_list = [], [], []
    pocket_center_list, pocket_center_pred_list = [], []

    protein_len_list = []
    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    for data in data_iter:
        pdb_list = pdb_list + data.pdb
        data = data.to(device)

        (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
         y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
         pred_pocket_center, dis_map, keepNode_less_5, pocket_radius_pred, _, gt_pocket_batch) = model(data, stage=stage, train=False, flag=flag)

        gt_ligand_coord = data.coords
        if stage == 1:
            gt_pocket_coord = data.pocket_node_xyz
        else:
            gt_pocket_coord = data.pocket_node_xyz
        if args.permutation_invariant:
            ligand_coord_loss = args.coord_loss_weight * compute_permutation_loss(compound_coords_pred, gt_ligand_coord, data, com_coord_criterion).mean()
        else:
            ligand_coord_loss = args.coord_loss_weight * com_coord_criterion(compound_coords_pred, gt_ligand_coord)

        pocket_coord_loss = args.pro_loss_weight * pro_coord_criterion(pocket_coords_pred, gt_pocket_coord)
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (
                    protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
        pocket_center_loss = args.pocket_distance_loss_weight * pocket_center_criterion(pred_pocket_center,
                                                                                        data.coords_center)
        contact_by_feat_loss = args.pair_distance_loss_weight * contact_criterion(y_pred, dis_map)
        contact_by_coord_loss = args.pair_distance_loss_weight * contact_criterion(y_pred_by_coords, dis_map)
        contact_distill_loss = args.pair_distance_distill_loss_weight * contact_criterion(y_pred, y_pred_by_coords)
        if not args.force_fix_radius:
            pocket_radius_pred_loss = args.pocket_radius_loss_weight * pocket_radius_criterion(
                pocket_radius_pred.squeeze(1), data.ligand_radius.to(pocket_radius_pred.dtype))
        else:
            pocket_radius_pred_loss = torch.zeros_like(contact_distill_loss, device=contact_distill_loss.device)

        if flag == 1:
            loss = (ligand_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                    contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)
        elif flag == 2:
            if args.wodm and args.wopr:
                loss = pocket_coord_loss + pocket_cls_loss + pocket_center_loss + contact_by_coord_loss
            elif (not args.wodm) and args.wopr:
                loss = (pocket_coord_loss + pocket_cls_loss + pocket_center_loss +
                        contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)
            elif args.wodm and (not args.wopr):
                loss = (pocket_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                        contact_by_coord_loss)
            else:
                loss = (pocket_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                        contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)
        else:
            loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                    contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)

        # evaluate metrics
        sd = ((compound_coords_pred.detach() - gt_ligand_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(src=sd, index=compound_batch, dim=0).sqrt().detach()

        centroid_pred = scatter_mean(src=compound_coords_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=gt_ligand_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)
        
        if stage == 1:
            pocket_sd = ((pocket_coords_pred.detach() - gt_pocket_coord) ** 2).sum(dim=-1)
            pocket_rmsd = scatter_mean(src=pocket_sd, index=pocket_batch, dim=0).sqrt().detach()
        else:
            data.af2_protein_node_xyz[data['pocket'].keepNode] = pocket_coords_pred
            pocket_sd = ((data.af2_protein_node_xyz[data['pocket'].keepNode_noNoise] - data.protein_node_xyz[data['pocket'].keepNode_noNoise]) ** 2).sum(dim=-1)
            pocket_rmsd = scatter_mean(src=pocket_sd, index=gt_pocket_batch, dim=0).sqrt().detach()

        # record current loss value
        batch_count += 1
        keepNode_less_5_count += keepNode_less_5
        batch_loss += loss.item()
        batch_com_coord_loss += ligand_coord_loss.item()
        batch_pro_coord_loss += pocket_coord_loss.item()
        batch_pocket_cls_loss += pocket_cls_loss.item()
        batch_pocket_center_loss += pocket_center_loss.item()
        batch_pocket_radius_loss += pocket_radius_pred_loss.item()
        batch_contact_loss += (contact_by_feat_loss.item() + contact_by_coord_loss.item() + contact_distill_loss.item())
        batch_contact_by_feat_loss += contact_by_feat_loss.item()
        batch_contact_by_coord_loss += contact_by_coord_loss.item()
        batch_contact_distill_loss += contact_distill_loss.item()

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
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())
        pocket_rmsd_list.append(pocket_rmsd.detach())
        pocket_rmsd_2A_list.append((pocket_rmsd.detach() < 2).float())
        pocket_rmsd_5A_list.append((pocket_rmsd.detach() < 5).float())

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
    centroid_dis = torch.cat(centroid_dis_list)
    centroid_dis_2A = torch.cat(centroid_dis_2A_list)
    centroid_dis_5A = torch.cat(centroid_dis_5A_list)
    centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
    centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
    centroid_dis_75 = torch.quantile(centroid_dis, 0.75)
    pocket_rmsd = torch.cat(pocket_rmsd_list)
    pocket_rmsd_2A = torch.cat(pocket_rmsd_2A_list)
    pocket_rmsd_5A = torch.cat(pocket_rmsd_5A_list)
    pocket_rmsd_25 = torch.quantile(pocket_rmsd, 0.25)
    pocket_rmsd_50 = torch.quantile(pocket_rmsd, 0.50)
    pocket_rmsd_75 = torch.quantile(pocket_rmsd, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_center = torch.cat(pocket_center_list)
    pocket_center_pred = torch.cat(pocket_center_pred_list)
    pocket_coords_pred = torch.cat(poc_coord_pred_list)

    protein_len = torch.cat(protein_len_list)
    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / pocket_cls_pred_round.shape[0]

    metrics = {
        f"{prefix}/epoch": epoch, f"{prefix}/samples": count, f"{prefix}/skip_samples": skip_count
    }
    loss_dict = {
        f"{prefix}/loss": batch_loss / batch_count,
        f"{prefix}/lig_coo_loss": batch_com_coord_loss / batch_count,
        f"{prefix}/poc_coo_loss": batch_pro_coord_loss / batch_count,
        f"{prefix}/pocket_cls_loss": batch_pocket_cls_loss / batch_count,
        f"{prefix}/pocket_center_loss": batch_pocket_center_loss / batch_count,
        f"{prefix}/pocket_radius_loss": batch_pocket_radius_loss / batch_count,
        f"{prefix}/contact_loss": batch_contact_loss / batch_count,
        f"{prefix}/contact_loss_by_feat": batch_contact_by_feat_loss / batch_count,
        f"{prefix}/contact_loss_by_coord": batch_contact_by_coord_loss / batch_count,
        f"{prefix}/contact_distill_loss": batch_contact_distill_loss / batch_count
    }
    metrics.update(loss_dict)
    eval_dict = {
        f"{prefix}/rmsd": rmsd.mean().item(), f"{prefix}/rmsd < 2A": rmsd_2A.mean().item(),
        f"{prefix}/rmsd < 5A": rmsd_5A.mean().item(),
        f"{prefix}/rmsd 25%": rmsd_25.item(), f"{prefix}/rmsd 50%": rmsd_50.item(),
        f"{prefix}/rmsd 75%": rmsd_75.item(),
        f"{prefix}/centroid_dis": centroid_dis.mean().item(),
        f"{prefix}/centroid_dis < 2A": centroid_dis_2A.mean().item(),
        f"{prefix}/centroid_dis < 5A": centroid_dis_5A.mean().item(),
        f"{prefix}/centroid_dis 25%": centroid_dis_25.item(), f"{prefix}/centroid_dis 50%": centroid_dis_50.item(),
        f"{prefix}/centroid_dis 75%": centroid_dis_75.item(),
        f"{prefix}/pocket_rmsd": pocket_rmsd.mean().item(), f"{prefix}/pocket_rmsd < 2A": pocket_rmsd_2A.mean().item(),
        f"{prefix}/pocket_rmsd < 5A": pocket_rmsd_5A.mean().item(),
        f"{prefix}/pocket_rmsd 25%": pocket_rmsd_25.item(), f"{prefix}/pocket_rmsd 50%": pocket_rmsd_50.item(),
        f"{prefix}/pocket_rmsd 75%": pocket_rmsd_75.item()
    }
    metrics.update(eval_dict)
    metrics.update({f"{prefix}/pocket_cls_accuracy": pocket_cls_accuracy})
    pocket_eval_dict = pocket_metrics(pocket_center_pred, pocket_center, prefix=prefix)
    metrics.update(pocket_eval_dict)

    if args.save_rmsd:
        with open(args.rmsd_file, "w") as f:
            for i in range(len(pdb_list)):
                pdb = pdb_list[i]
                rmsd_value = rmsd[i].item()
                out = str(pdb) + "\t" + str(rmsd_value) + "\n"
                f.write(out)

    return metrics

@torch.no_grad()
def evaluate_iter(accelerator, args, epoch, data_loader, model, device,
                 pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion, contact_criterion,
                 com_coord_criterion, pro_coord_criterion, prefix="test", stage=1, flag=2):
    batch_count = 0
    batch_loss = 0.0
    batch_com_coord_loss, batch_pro_coord_loss = 0.0, 0.0
    batch_contact_loss, batch_contact_by_feat_loss, batch_contact_by_coord_loss, batch_contact_distill_loss = 0.0, 0.0, 0.0, 0.0
    batch_pocket_cls_loss, batch_pocket_center_loss, batch_pocket_radius_loss = 0.0, 0.0, 0.0

    count = 0
    skip_count = 0
    keepNode_less_5_count = 0
    gt_lig_coord_list, lig_coord_pred_list = [], []
    gt_poc_coord_list, poc_coord_pred_list = [], []

    rmsd_list = []
    rmsd_2A_list, rmsd_5A_list = [], []
    centroid_dis_list = []
    centroid_dis_2A_list, centroid_dis_5A_list = [], []
    pocket_rmsd_list = []
    pocket_rmsd_2A_list, pocket_rmsd_5A_list = [], []

    pocket_cls_list, pocket_cls_pred_list, pocket_cls_pred_round_list = [], [], []
    pocket_center_list, pocket_center_pred_list = [], []

    protein_len_list = []
    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    total_iter = args.total_iterative
    for ori_data in data_iter:
        data = copy.deepcopy(ori_data)
        data = data.to(device)
        for iter_i in range(total_iter):
            if iter_i == 0:
                first_iter = True
            else:
                first_iter = False
            (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
             y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls,
             protein_out_mask_whole, protein_coords_batched_whole, pred_pocket_center,
             dis_map, keepNode_less_5, pocket_radius_pred) = model(data, stage=stage, train=False, flag=flag, first_iter=first_iter)
            # iterative
            if iter_i < total_iter - 1:
                data.input_pocket_node_xyz = pocket_coords_pred
                data['compound'].node_coords = compound_coords_pred
                data.input_protein_node_xyz[data['pocket'].keepNode] = pocket_coords_pred

                compound_flag = torch.logical_and(data["complex"].segment == 0, ~data["complex"].is_global)
                data['complex'].node_coords[compound_flag] = compound_coords_pred
                pocket_flag = torch.logical_and(data["complex"].segment == 1, ~data["complex"].is_global)
                data['complex'].node_coords[pocket_flag] = pocket_coords_pred

                compound_flag = torch.logical_and(data["complex_whole_protein"].segment == 0, ~data["complex_whole_protein"].is_global)
                data['complex_whole_protein'].node_coords[compound_flag] = compound_coords_pred
                protein_flag = torch.logical_and(data["complex_whole_protein"].segment == 1, ~data["complex_whole_protein"].is_global)
                indices_of_true = torch.where(protein_flag)[0]
                selected_indices = indices_of_true[data["pocket"].keepNode.bool()]
                data['complex_whole_protein'].node_coords[selected_indices] = pocket_coords_pred

        gt_ligand_coord = data.coords
        if stage == 1:
            gt_pocket_coord = data.pocket_node_xyz
        else:
            gt_pocket_coord = data.pocket_node_xyz
        if args.permutation_invariant:
            ligand_coord_loss = args.coord_loss_weight * compute_permutation_loss(compound_coords_pred,
                                                                                  gt_ligand_coord, data,
                                                                                  com_coord_criterion).mean()
        else:
            ligand_coord_loss = args.coord_loss_weight * com_coord_criterion(compound_coords_pred, gt_ligand_coord)

        pocket_coord_loss = args.pro_loss_weight * pro_coord_criterion(pocket_coords_pred, gt_pocket_coord)
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred,pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
        pocket_center_loss = args.pocket_distance_loss_weight * pocket_center_criterion(pred_pocket_center, data.coords_center)
        contact_by_feat_loss = args.pair_distance_loss_weight * contact_criterion(y_pred, dis_map)
        contact_by_coord_loss = args.pair_distance_loss_weight * contact_criterion(y_pred_by_coords, dis_map)
        contact_distill_loss = args.pair_distance_distill_loss_weight * contact_criterion(y_pred, y_pred_by_coords)
        if not args.force_fix_radius:
            pocket_radius_pred_loss = args.pocket_radius_loss_weight * pocket_radius_criterion(pocket_radius_pred.squeeze(1), data.ligand_radius.to(pocket_radius_pred.dtype))
        else:
            pocket_radius_pred_loss = torch.zeros_like(contact_distill_loss, device=contact_distill_loss.device)

        if flag == 1:
            loss = (ligand_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                    contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)
        elif flag == 2:
            loss = (pocket_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                    contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)
        else:
            loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss + pocket_center_loss +
                    pocket_radius_pred_loss + contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)

        # evaluate metrics
        sd = ((compound_coords_pred.detach() - gt_ligand_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(src=sd, index=compound_batch, dim=0).sqrt().detach()

        centroid_pred = scatter_mean(src=compound_coords_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=gt_ligand_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)

        pocket_sd = ((pocket_coords_pred.detach() - gt_pocket_coord) ** 2).sum(dim=-1)
        pocket_rmsd = scatter_mean(src=pocket_sd, index=pocket_batch, dim=0).sqrt().detach()

        # record current loss value
        batch_count += 1
        keepNode_less_5_count += keepNode_less_5
        batch_loss += loss.item()
        batch_com_coord_loss += ligand_coord_loss.item()
        batch_pro_coord_loss += pocket_coord_loss.item()
        batch_pocket_cls_loss += pocket_cls_loss.item()
        batch_pocket_center_loss += pocket_center_loss.item()
        batch_pocket_radius_loss += pocket_radius_pred_loss.item()
        batch_contact_loss += (
                    contact_by_feat_loss.item() + contact_by_coord_loss.item() + contact_distill_loss.item())
        batch_contact_by_feat_loss += contact_by_feat_loss.item()
        batch_contact_by_coord_loss += contact_by_coord_loss.item()
        batch_contact_distill_loss += contact_distill_loss.item()

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
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())
        pocket_rmsd_list.append(pocket_rmsd.detach())
        pocket_rmsd_2A_list.append((pocket_rmsd.detach() < 2).float())
        pocket_rmsd_5A_list.append((pocket_rmsd.detach() < 5).float())

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
    centroid_dis = torch.cat(centroid_dis_list)
    centroid_dis_2A = torch.cat(centroid_dis_2A_list)
    centroid_dis_5A = torch.cat(centroid_dis_5A_list)
    centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
    centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
    centroid_dis_75 = torch.quantile(centroid_dis, 0.75)
    pocket_rmsd = torch.cat(pocket_rmsd_list)
    pocket_rmsd_2A = torch.cat(pocket_rmsd_2A_list)
    pocket_rmsd_5A = torch.cat(pocket_rmsd_5A_list)
    pocket_rmsd_25 = torch.quantile(pocket_rmsd, 0.25)
    pocket_rmsd_50 = torch.quantile(pocket_rmsd, 0.50)
    pocket_rmsd_75 = torch.quantile(pocket_rmsd, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_center = torch.cat(pocket_center_list)
    pocket_center_pred = torch.cat(pocket_center_pred_list)
    pocket_coords_pred = torch.cat(poc_coord_pred_list)

    protein_len = torch.cat(protein_len_list)
    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / pocket_cls_pred_round.shape[0]

    metrics = {
        f"{prefix}/epoch": epoch, f"{prefix}/samples": count, f"{prefix}/skip_samples": skip_count
    }
    loss_dict = {
        f"{prefix}/loss": batch_loss / batch_count,
        f"{prefix}/lig_coo_loss": batch_com_coord_loss / batch_count,
        f"{prefix}/poc_coo_loss": batch_pro_coord_loss / batch_count,
        f"{prefix}/pocket_cls_loss": batch_pocket_cls_loss / batch_count,
        f"{prefix}/pocket_center_loss": batch_pocket_center_loss / batch_count,
        f"{prefix}/pocket_radius_loss": batch_pocket_radius_loss / batch_count,
        f"{prefix}/contact_loss": batch_contact_loss / batch_count,
        f"{prefix}/contact_loss_by_feat": batch_contact_by_feat_loss / batch_count,
        f"{prefix}/contact_loss_by_coord": batch_contact_by_coord_loss / batch_count,
        f"{prefix}/contact_distill_loss": batch_contact_distill_loss / batch_count
    }
    metrics.update(loss_dict)
    eval_dict = {
        f"{prefix}/rmsd": rmsd.mean().item(), f"{prefix}/rmsd < 2A": rmsd_2A.mean().item(),
        f"{prefix}/rmsd < 5A": rmsd_5A.mean().item(),
        f"{prefix}/rmsd 25%": rmsd_25.item(), f"{prefix}/rmsd 50%": rmsd_50.item(),
        f"{prefix}/rmsd 75%": rmsd_75.item(),
        f"{prefix}/centroid_dis": centroid_dis.mean().item(),
        f"{prefix}/centroid_dis < 2A": centroid_dis_2A.mean().item(),
        f"{prefix}/centroid_dis < 5A": centroid_dis_5A.mean().item(),
        f"{prefix}/centroid_dis 25%": centroid_dis_25.item(), f"{prefix}/centroid_dis 50%": centroid_dis_50.item(),
        f"{prefix}/centroid_dis 75%": centroid_dis_75.item(),
        f"{prefix}/pocket_rmsd": pocket_rmsd.mean().item(),
        f"{prefix}/pocket_rmsd < 2A": pocket_rmsd_2A.mean().item(),
        f"{prefix}/pocket_rmsd < 5A": pocket_rmsd_5A.mean().item(),
        f"{prefix}/pocket_rmsd 25%": pocket_rmsd_25.item(), f"{prefix}/pocket_rmsd 50%": pocket_rmsd_50.item(),
        f"{prefix}/pocket_rmsd 75%": pocket_rmsd_75.item()
    }
    metrics.update(eval_dict)
    metrics.update({f"{prefix}/pocket_cls_accuracy": pocket_cls_accuracy})
    pocket_eval_dict = pocket_metrics(pocket_center_pred, pocket_center, prefix=prefix)
    metrics.update(pocket_eval_dict)

    return metrics


def obtain_best_result_epoch(epoch, best_result, best_epoch, metrics_test, prefix):
    rmsd = metrics_test[f"{prefix}/rmsd"]
    rmsd_2A = metrics_test[f"{prefix}/rmsd < 2A"]
    rmsd_5A = metrics_test[f"{prefix}/rmsd < 5A"]
    rmsd_25 = metrics_test[f"{prefix}/rmsd 25%"]
    rmsd_50 = metrics_test[f"{prefix}/rmsd 50%"]
    rmsd_75 = metrics_test[f"{prefix}/rmsd 75%"]
    centroid_dis = metrics_test[f"{prefix}/centroid_dis"]
    centroid_dis_2A = metrics_test[f"{prefix}/centroid_dis < 2A"]
    centroid_dis_5A = metrics_test[f"{prefix}/centroid_dis < 5A"]
    centroid_dis_25 = metrics_test[f"{prefix}/centroid_dis 25%"]
    centroid_dis_50 = metrics_test[f"{prefix}/centroid_dis 50%"]
    centroid_dis_75 = metrics_test[f"{prefix}/centroid_dis 75%"]
    pocket_rmsd = metrics_test[f"{prefix}/pocket_rmsd"]
    pocket_rmsd_2A = metrics_test[f"{prefix}/pocket_rmsd < 2A"]
    pocket_rmsd_5A = metrics_test[f"{prefix}/pocket_rmsd < 5A"]
    pocket_rmsd_25 = metrics_test[f"{prefix}/pocket_rmsd 25%"]
    pocket_rmsd_50 = metrics_test[f"{prefix}/pocket_rmsd 50%"]
    pocket_rmsd_75 = metrics_test[f"{prefix}/pocket_rmsd 75%"]

    # record best result
    if rmsd < best_result['rmsd']:
        best_result['rmsd'] = rmsd
        best_epoch['rmsd'] = epoch
    if rmsd_2A > best_result['rmsd < 2A']:
        best_result['rmsd < 2A'] = rmsd_2A
        best_epoch['rmsd < 2A'] = epoch
    if rmsd_5A > best_result['rmsd < 5A']:
        best_result['rmsd < 5A'] = rmsd_5A
        best_epoch['rmsd < 5A'] = epoch
    if rmsd_25 < best_result["rmsd 25%"]:
        best_result["rmsd 25%"] = rmsd_25
        best_epoch['rmsd 25%'] = epoch
    if rmsd_50 < best_result["rmsd 50%"]:
        best_result["rmsd 50%"] = rmsd_50
        best_epoch['rmsd 50%'] = epoch
    if rmsd_75 < best_result["rmsd 75%"]:
        best_result["rmsd 75%"] = rmsd_75
        best_epoch['rmsd 75%'] = epoch

    if centroid_dis < best_result["centroid_dis"]:
        best_result['centroid_dis'] = centroid_dis
        best_epoch['centroid_dis'] = epoch
    if centroid_dis_2A > best_result['centroid_dis < 2A']:
        best_result['centroid_dis < 2A'] = centroid_dis_2A
        best_epoch['centroid_dis < 2A'] = epoch
    if centroid_dis_5A > best_result['centroid_dis < 5A']:
        best_result['centroid_dis < 5A'] = centroid_dis_5A
        best_epoch['centroid_dis < 5A'] = epoch
    if centroid_dis_25 < best_result['centroid_dis 25%']:
        best_result['centroid_dis 25%'] = centroid_dis_25
        best_epoch['centroid_dis 25%'] = epoch
    if centroid_dis_50 < best_result['centroid_dis 50%']:
        best_result['centroid_dis 50%'] = centroid_dis_50
        best_epoch['centroid_dis 50%'] = epoch
    if centroid_dis_75 < best_result['centroid_dis 75%']:
        best_result['centroid_dis 75%'] = centroid_dis_75
        best_epoch['centroid_dis 75%'] = epoch

    if pocket_rmsd < best_result['pocket_rmsd']:
        best_result['pocket_rmsd'] = pocket_rmsd
        best_epoch['pocket_rmsd'] = epoch
    if pocket_rmsd_2A > best_result['pocket_rmsd < 2A']:
        best_result['pocket_rmsd < 2A'] = pocket_rmsd_2A
        best_epoch['pocket_rmsd < 2A'] = epoch
    if pocket_rmsd_5A > best_result['pocket_rmsd < 5A']:
        best_result['pocket_rmsd < 5A'] = pocket_rmsd_5A
        best_epoch['pocket_rmsd < 5A'] = epoch
    if pocket_rmsd_25 < best_result["pocket_rmsd 25%"]:
        best_result["pocket_rmsd 25%"] = pocket_rmsd_25
        best_epoch['pocket_rmsd 25%'] = epoch
    if pocket_rmsd_50 < best_result["pocket_rmsd 50%"]:
        best_result["pocket_rmsd 50%"] = pocket_rmsd_50
        best_epoch['pocket_rmsd 50%'] = epoch
    if pocket_rmsd_75 < best_result["pocket_rmsd 75%"]:
        best_result["pocket_rmsd 75%"] = pocket_rmsd_75
        best_epoch['pocket_rmsd 75%'] = epoch
    return best_result, best_epoch


def define_best_dict(require_pretrain=False):
    if not require_pretrain:
        best_result_gt = {
            "rmsd": 1000, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 1000, "rmsd 50%": 1000, "rmsd 75%": 1000,
            "centroid_dis": 1000, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 1000, "centroid_dis 50%": 1000, "centroid_dis 75%": 1000,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        best_epoch_gt = {
            "rmsd": 0, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 0, "rmsd 50%": 0, "rmsd 75%": 0,
            "centroid_dis": 0, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 0, "centroid_dis 50%": 0, "centroid_dis 75%": 0,
            "pocket_rmsd": 0, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 0, "pocket_rmsd 50%": 0, "pocket_rmsd 75%": 0,
        }
        best_result_pp = {
            "rmsd": 1000, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 1000, "rmsd 50%": 1000, "rmsd 75%": 1000,
            "centroid_dis": 1000, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 1000, "centroid_dis 50%": 1000, "centroid_dis 75%": 1000,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        best_epoch_pp = {
            "rmsd": 0, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 0, "rmsd 50%": 0, "rmsd 75%": 0,
            "centroid_dis": 0, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 0, "centroid_dis 50%": 0, "centroid_dis 75%": 0,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        return best_result_gt, best_epoch_gt, best_result_pp, best_epoch_pp
    else:
        pre_best_result_gt = {
            "rmsd": 1000, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 1000, "rmsd 50%": 1000, "rmsd 75%": 1000,
            "centroid_dis": 1000, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 1000, "centroid_dis 50%": 1000, "centroid_dis 75%": 1000,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        pre_best_epoch_gt = {
            "rmsd": 0, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 0, "rmsd 50%": 0, "rmsd 75%": 0,
            "centroid_dis": 0, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 0, "centroid_dis 50%": 0, "centroid_dis 75%": 0,
            "pocket_rmsd": 0, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 0, "pocket_rmsd 50%": 0, "pocket_rmsd 75%": 0,
        }
        pre_best_result_pp = {
            "rmsd": 1000, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 1000, "rmsd 50%": 1000, "rmsd 75%": 1000,
            "centroid_dis": 1000, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 1000, "centroid_dis 50%": 1000, "centroid_dis 75%": 1000,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        pre_best_epoch_pp = {
            "rmsd": 0, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 0, "rmsd 50%": 0, "rmsd 75%": 0,
            "centroid_dis": 0, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 0, "centroid_dis 50%": 0, "centroid_dis 75%": 0,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        iter_best_result_gt = {
            "rmsd": 1000, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 1000, "rmsd 50%": 1000, "rmsd 75%": 1000,
            "centroid_dis": 1000, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 1000, "centroid_dis 50%": 1000, "centroid_dis 75%": 1000,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        iter_best_epoch_gt = {
            "rmsd": 0, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 0, "rmsd 50%": 0, "rmsd 75%": 0,
            "centroid_dis": 0, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 0, "centroid_dis 50%": 0, "centroid_dis 75%": 0,
            "pocket_rmsd": 0, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 0, "pocket_rmsd 50%": 0, "pocket_rmsd 75%": 0,
        }
        iter_best_result_pp = {
            "rmsd": 1000, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 1000, "rmsd 50%": 1000, "rmsd 75%": 1000,
            "centroid_dis": 1000, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 1000, "centroid_dis 50%": 1000, "centroid_dis 75%": 1000,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        iter_best_epoch_pp = {
            "rmsd": 0, "rmsd < 2A": 0, "rmsd < 5A": 0, "rmsd 25%": 0, "rmsd 50%": 0, "rmsd 75%": 0,
            "centroid_dis": 0, "centroid_dis < 2A": 0, "centroid_dis < 5A": 0,
            "centroid_dis 25%": 0, "centroid_dis 50%": 0, "centroid_dis 75%": 0,
            "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
            "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
        }
        return (pre_best_result_gt, pre_best_epoch_gt, pre_best_result_pp, pre_best_epoch_pp,
                iter_best_result_gt, iter_best_epoch_gt, iter_best_result_pp, iter_best_epoch_pp)
