import torch
from tqdm import tqdm
from torch_scatter import scatter_mean
import numpy as np
import copy
import time

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
    infer_time_list = []
    rmsd_list = []
    rmsd_2A_list, rmsd_5A_list = [], []
    centroid_dis_list = []
    centroid_dis_2A_list, centroid_dis_5A_list = [], []
    pocket_rmsd_list = []
    pocket_rmsd_2A_list, pocket_rmsd_5A_list = [], []
    af2_pocket_rmsd_list = []

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

        t0 = time.time()
        (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
         f_y_pred, g_y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
         pred_pocket_center, dis_map, keepNode_less_5, pocket_radius_pred, _, gt_pocket_batch, gt_keepNode) = model(data, stage=stage, train=False, flag=flag)
        t1 = time.time()
        delta_time = t1 - t0
        infer_time_list.append(delta_time)

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
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
        pocket_center_loss = args.pocket_distance_loss_weight * pocket_center_criterion(pred_pocket_center, data.coords_center)
        contact_by_feat_loss = args.pair_distance_loss_weight * (contact_criterion(f_y_pred, dis_map) + contact_criterion(g_y_pred, dis_map))
        contact_by_coord_loss = args.pair_distance_loss_weight * contact_criterion(y_pred_by_coords, dis_map)
        contact_distill_loss = args.pair_distance_distill_loss_weight * (contact_criterion(f_y_pred, y_pred_by_coords) + contact_criterion(g_y_pred, y_pred_by_coords))
        if not args.force_fix_radius:
            pocket_radius_pred_loss = args.pocket_radius_loss_weight * pocket_radius_criterion(pocket_radius_pred.squeeze(1), data.ligand_radius.to(pocket_radius_pred.dtype))
        else:
            pocket_radius_pred_loss = torch.zeros_like(contact_distill_loss, device=contact_distill_loss.device)

        if args.freeze_ligandDock_pocketPred:
            loss = pocket_coord_loss + contact_by_coord_loss
        else:
            if args.wodm and args.wopr:
                contact_by_feat_loss = torch.zeros_like(contact_by_feat_loss, device=accelerator.device)
                contact_distill_loss = torch.zeros_like(contact_distill_loss, device=accelerator.device)
                pocket_radius_pred_loss = torch.zeros_like(pocket_radius_pred_loss, device=accelerator.device)
                loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss +
                        pocket_center_loss + contact_by_coord_loss)
            elif (not args.wodm) and args.wopr:
                pocket_radius_pred_loss = torch.zeros_like(pocket_radius_pred_loss, device=accelerator.device)
                loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss +
                        pocket_center_loss + contact_by_coord_loss + contact_by_feat_loss + contact_distill_loss)
            elif args.wodm and (not args.wopr):
                pocket_radius_pred_loss = torch.zeros_like(pocket_radius_pred_loss, device=accelerator.device)
                loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss +
                        pocket_center_loss + contact_by_coord_loss + pocket_radius_pred_loss)
            else:
                loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss + pocket_center_loss +
                        pocket_radius_pred_loss +contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)

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
            data.input_protein_node_xyz[data['pocket'].keepNode] = pocket_coords_pred
            pocket_sd = ((data.input_protein_node_xyz[gt_keepNode] - data.protein_node_xyz[gt_keepNode]) ** 2).sum(dim=-1)
            pocket_rmsd = scatter_mean(src=pocket_sd, index=gt_pocket_batch, dim=0).sqrt().detach()
        af2_pocket_sd = ((data.af2_protein_node_xyz[gt_keepNode] - data.protein_node_xyz[gt_keepNode]) ** 2).sum(dim=-1)
        af2_pocket_rmsd = scatter_mean(src=af2_pocket_sd, index=gt_pocket_batch, dim=0).sqrt().detach()

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
        af2_pocket_rmsd_list.append(af2_pocket_rmsd.detach())

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
    af2_pocket_rmsd = torch.cat(af2_pocket_rmsd_list)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_center = torch.cat(pocket_center_list)
    pocket_center_pred = torch.cat(pocket_center_pred_list)
    pocket_coords_pred = torch.cat(poc_coord_pred_list)

    protein_len = torch.cat(protein_len_list)
    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / pocket_cls_pred_round.shape[0]

    metrics = {
        f"{prefix}/epoch": epoch, f"{prefix}/samples": count, f"{prefix}/skip_samples": skip_count,
#        f"{prefix}/avg_infer_time": sum(infer_time_list)/len(infer_time_list)
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
        f"{prefix}/pocket_rmsd 75%": pocket_rmsd_75.item(),
        f"{prefix}/af2_pocket_rmsd": af2_pocket_rmsd.mean().item(),
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