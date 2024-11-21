
import torch
from torch_scatter import scatter_mean
from tqdm import tqdm
import random
import copy

from utils.loss_utils import compute_permutation_loss
from utils.metrics import pocket_metrics


def train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer, scheduler,
                    pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                    contact_criterion, com_coord_criterion, pro_coord_criterion, device, flag=1, prefix="protein-train"):
    batch_count = 0
    batch_loss = 0.0
    batch_com_coord_loss, batch_pro_coord_loss = 0.0, 0.0
    batch_contact_loss, batch_contact_by_feat_loss, batch_contact_by_coord_loss, batch_contact_distill_loss = 0.0, 0.0, 0.0, 0.0
    batch_pocket_cls_loss, batch_pocket_center_loss, batch_pocket_radius_loss = 0.0, 0.0, 0.0

    count = 0
    skip_count = 0
    keepNode_less_5_count = 0
    # gt_lig_coord_list, lig_coord_pred_list = [], []
    # gt_poc_coord_list, poc_coord_pred_list = [], []

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
        data_iter = train_loader
    else:
        data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    for batch_id, data in enumerate(data_iter, start=1):
        optimizer.zero_grad()

        (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
         y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
         pred_pocket_center, dis_map, keepNode_less_5, pocket_radius_pred, _, gt_pocket_batch) = model(data, stage=1, train=True, flag=flag)

        if pocket_cls_pred.isnan().any() or pred_pocket_center.isnan().any():
            print(f"nan occurs in model of epoch {epoch}- batch {batch_id}")
            continue
        if y_pred.isnan().any() or y_pred_by_coords.isnan().any() or compound_coords_pred.isnan().any() or pocket_coords_pred.isnan().any():
            print(f"nan occurs in model of epoch {epoch}- batch {batch_id}")
            continue

        gt_ligand_coord = data.coords
        gt_pocket_coord = data.pocket_node_xyz
        # 计算f_ligand_mode + pocket_pred_model的loss
        if args.permutation_invariant:
            ligand_coord_loss = args.coord_loss_weight * compute_permutation_loss(compound_coords_pred, gt_ligand_coord, data, com_coord_criterion).mean()
        else:
            ligand_coord_loss = args.coord_loss_weight * com_coord_criterion(compound_coords_pred, gt_ligand_coord)
        pocket_coord_loss = args.pro_loss_weight * pro_coord_criterion(pocket_coords_pred, gt_pocket_coord)
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
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
        accelerator.backward(loss)
        if args.clip_grad:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # evaluate metrics
        sd = ((compound_coords_pred.detach() - gt_ligand_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(src=sd, index=compound_batch, dim=0).sqrt().detach()

        centroid_pred = scatter_mean(src=compound_coords_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=gt_ligand_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)

        pocket_sd = ((pocket_coords_pred.detach() - gt_pocket_coord) ** 2).sum(dim=-1)
        pocket_rmsd = scatter_mean(src=pocket_sd, index=pocket_batch, dim=0).sqrt().detach()
        # data.input_protein_node_xyz[data['pocket'].keepNode] = pocket_coords_pred
        # pocket_sd = ((data.input_protein_node_xyz[data['pocket'].keepNode_noNoise] - data.protein_node_xyz[data['pocket'].keepNode_noNoise]) ** 2).sum(dim=-1)
        # pocket_rmsd = scatter_mean(src=pocket_sd, index=gt_pocket_batch, dim=0).sqrt().detach()

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

        # gt_lig_coord_list.append(gt_ligand_coord.detach())
        # lig_coord_pred_list.append(compound_coords_pred.detach())
        # gt_poc_coord_list.append(gt_pocket_coord.detach())
        # poc_coord_pred_list.append(pocket_coords_pred.detach())
        # pocket_center_list.append(data.coords_center.detach().cpu())
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

        if batch_id % args.log_interval == 0:
            state_dict = {}
            state_dict['step'] = batch_id
            state_dict['lr'] = optimizer.param_groups[0]['lr']
            state_dict['loss'] = loss.item()
            state_dict['lig_coo_loss'] = ligand_coord_loss.item()
            state_dict['poc_coo_loss'] = pocket_coord_loss.item()
            state_dict['pocket_radius_loss'] = pocket_radius_pred_loss.item()
            state_dict['pocket_cls_loss'] = pocket_cls_loss.item()
            state_dict['pocket_center_loss'] = pocket_center_loss.item()
            state_dict['contact_loss'] = contact_by_feat_loss.item() + contact_by_coord_loss.item() + contact_distill_loss.item()
            state_dict['keepNode_less_5'] = keepNode_less_5_count
            # logger写日志
            logger.log_stats(state_dict, epoch, args, prefix='train')

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
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_center = torch.cat(pocket_center_list)
    pocket_center_pred = torch.cat(pocket_center_pred_list)
    # pocket_coords_pred = torch.cat(poc_coord_pred_list)

    protein_len = torch.cat(protein_len_list)
    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / pocket_cls_pred_round.shape[0]

    metrics = {
        f"{prefix}/epoch": epoch, f"{prefix}/lr": optimizer.param_groups[0]['lr'],
        f"{prefix}/samples": count, f"{prefix}/skip_samples": skip_count,
        f"{prefix}/keepNode_less_5": keepNode_less_5_count
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

    return metrics




def iter_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer, scheduler,
                   pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                   contact_criterion, com_coord_criterion, pro_coord_criterion, device, flag=1, prefix="protein-train"):
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
        data_iter = train_loader
    else:
        data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    for batch_id, ori_data in enumerate(data_iter, start=1):
        data = copy.deepcopy(ori_data)
        optimizer.zero_grad()
        if args.random_total_iter:
            total_iter = random.randint(1, args.total_iterative)
        else:
            total_iter = args.total_iterative
        for iter_i in range(total_iter):
            if iter_i < total_iter - 1:
                with torch.no_grad():
                    (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
                     y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
                     pred_pocket_center, dis_map, keepNode_less_5, pocket_radius_pred) = model(data, stage=1, train=True, flag=flag)
                    # iterative
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
            else:
                (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch, y_pred, y_pred_by_coords,
                 pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole, pred_pocket_center,
                 dis_map, keepNode_less_5, pocket_radius_pred) = model(data, stage=1, train=True, flag=flag)

        if pocket_cls_pred.isnan().any() or pred_pocket_center.isnan().any():
            print(f"nan occurs in model of epoch {epoch}- batch {batch_id}")
            continue
        if y_pred.isnan().any() or y_pred_by_coords.isnan().any() or compound_coords_pred.isnan().any() or pocket_coords_pred.isnan().any():
            print(f"nan occurs in model of epoch {epoch}- batch {batch_id}")
            continue

        gt_ligand_coord = data.coords
        gt_pocket_coord = data.pocket_node_xyz
        # 计算f_ligand_mode + pocket_pred_model的loss
        if args.permutation_invariant:
            ligand_coord_loss = args.coord_loss_weight * compute_permutation_loss(compound_coords_pred, gt_ligand_coord, data, com_coord_criterion).mean()
        else:
            ligand_coord_loss = args.coord_loss_weight * com_coord_criterion(compound_coords_pred, gt_ligand_coord)
        pocket_coord_loss = args.pro_loss_weight * pro_coord_criterion(pocket_coords_pred, gt_pocket_coord)
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
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
            loss = (ligand_coord_loss + pocket_coord_loss + pocket_cls_loss + pocket_center_loss + pocket_radius_pred_loss +
                    contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss)
        accelerator.backward(loss)
        if args.clip_grad:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

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
        batch_contact_loss += (contact_by_feat_loss.item() + contact_by_coord_loss.item() + contact_distill_loss.item())
        batch_contact_by_feat_loss += contact_by_feat_loss.item()
        batch_contact_by_coord_loss += contact_by_coord_loss.item()
        batch_contact_distill_loss += contact_distill_loss.item()

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

        if batch_id % args.log_interval == 0:
            state_dict = {}
            state_dict['step'] = batch_id
            state_dict['lr'] = optimizer.param_groups[0]['lr']
            state_dict['loss'] = loss.item()
            state_dict['lig_coo_loss'] = ligand_coord_loss.item()
            state_dict['poc_coo_loss'] = pocket_coord_loss.item()
            state_dict['pocket_radius_loss'] = pocket_radius_pred_loss.item()
            state_dict['pocket_cls_loss'] = pocket_cls_loss.item()
            state_dict['pocket_center_loss'] = pocket_center_loss.item()
            state_dict['contact_loss'] = contact_by_feat_loss.item() + contact_by_coord_loss.item() + contact_distill_loss.item()
            state_dict['keepNode_less_5'] = keepNode_less_5_count
            # logger写日志
            logger.log_stats(state_dict, epoch, args, prefix='train')

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
        f"{prefix}/epoch": epoch, f"{prefix}/lr": optimizer.param_groups[0]['lr'],
        f"{prefix}/samples": count, f"{prefix}/skip_samples": skip_count,
        f"{prefix}/keepNode_less_5": keepNode_less_5_count
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

    return metrics










