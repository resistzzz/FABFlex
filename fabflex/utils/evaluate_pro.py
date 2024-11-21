
import torch
from tqdm import tqdm
from torch_scatter import scatter_mean
import numpy as np
import copy

from utils.metrics import pocket_metrics
from utils.loss_utils import compute_permutation_loss

@torch.no_grad()
def evaluate_one(accelerator, args, epoch, data_loader, model, device,
                 contact_criterion, pro_coord_criterion, prefix="test", stage=1, flag=2):
    batch_count = 0
    batch_loss = 0.0
    batch_pro_coord_loss = 0.0
    batch_contact_loss, batch_contact_by_feat_loss, batch_contact_by_coord_loss, batch_contact_distill_loss = 0.0, 0.0, 0.0, 0.0

    keepNode_less_5_count = 0

    pdb_list = []
    pocket_rmsd_list = []
    pocket_rmsd_05A_list, pocket_rmsd_1A_list = [], []
    af2_pocket_rmsd_list = []

    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    for data in data_iter:
        pdb_list = pdb_list + data.pdb
        data = data.to(device)

        (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
         y_pred, y_pred_by_coords, dis_map, keepNode_less_5) = model(data, stage=stage, train=False, flag=flag)

        gt_pocket_coord = data.pocket_node_xyz
        pocket_coord_loss = args.pro_loss_weight * pro_coord_criterion(pocket_coords_pred, gt_pocket_coord)
        contact_by_feat_loss = args.pair_distance_loss_weight * contact_criterion(y_pred, dis_map)
        contact_by_coord_loss = args.pair_distance_loss_weight * contact_criterion(y_pred_by_coords, dis_map)
        contact_distill_loss = args.pair_distance_distill_loss_weight * contact_criterion(y_pred, y_pred_by_coords)

        if args.wodm:
            loss = pocket_coord_loss
        else:
            loss = pocket_coord_loss + contact_by_feat_loss + contact_by_coord_loss + contact_distill_loss

        pocket_sd = ((pocket_coords_pred.detach() - gt_pocket_coord) ** 2).sum(dim=-1)
        pocket_rmsd = scatter_mean(src=pocket_sd, index=pocket_batch, dim=0).sqrt().detach()

        af2_pocket_sd = ((data.af2_protein_node_xyz[data['pocket'].keepNode] - data.protein_node_xyz[data['pocket'].keepNode]) ** 2).sum(dim=-1)
        af2_pocket_rmsd = scatter_mean(src=af2_pocket_sd, index=pocket_batch, dim=0).sqrt().detach()

        # record current loss value
        batch_count += 1
        keepNode_less_5_count += keepNode_less_5
        batch_loss += loss.item()
        batch_pro_coord_loss += pocket_coord_loss.item()
        batch_contact_loss += (contact_by_feat_loss.item() + contact_by_coord_loss.item() + contact_distill_loss.item())
        batch_contact_by_feat_loss += contact_by_feat_loss.item()
        batch_contact_by_coord_loss += contact_by_coord_loss.item()
        batch_contact_distill_loss += contact_distill_loss.item()

        # record evaluation metrics
        pocket_rmsd_list.append(pocket_rmsd.detach())
        pocket_rmsd_05A_list.append((pocket_rmsd.detach() < 0.5).float())
        pocket_rmsd_1A_list.append((pocket_rmsd.detach() < 1).float())
        af2_pocket_rmsd_list.append(af2_pocket_rmsd.detach())

    pocket_rmsd = torch.cat(pocket_rmsd_list)
    pocket_rmsd_05A = torch.cat(pocket_rmsd_05A_list)
    pocket_rmsd_1A = torch.cat(pocket_rmsd_1A_list)
    pocket_rmsd_25 = torch.quantile(pocket_rmsd, 0.25)
    pocket_rmsd_50 = torch.quantile(pocket_rmsd, 0.50)
    pocket_rmsd_75 = torch.quantile(pocket_rmsd, 0.75)
    af2_pocket_rmsd = torch.cat(af2_pocket_rmsd_list)

    metrics = {
        f"{prefix}/epoch": epoch,
    }
    loss_dict = {
        f"{prefix}/loss": batch_loss / batch_count,
        f"{prefix}/poc_coo_loss": batch_pro_coord_loss / batch_count,
        f"{prefix}/contact_loss": batch_contact_loss / batch_count,
        f"{prefix}/contact_loss_by_feat": batch_contact_by_feat_loss / batch_count,
        f"{prefix}/contact_loss_by_coord": batch_contact_by_coord_loss / batch_count,
        f"{prefix}/contact_distill_loss": batch_contact_distill_loss / batch_count
    }
    metrics.update(loss_dict)
    eval_dict = {
        f"{prefix}/pocket_rmsd": pocket_rmsd.mean().item(), f"{prefix}/pocket_rmsd < 0.5A": pocket_rmsd_05A.mean().item(),
        f"{prefix}/pocket_rmsd < 1A": pocket_rmsd_1A.mean().item(),
        f"{prefix}/pocket_rmsd 25%": pocket_rmsd_25.item(), f"{prefix}/pocket_rmsd 50%": pocket_rmsd_50.item(),
        f"{prefix}/pocket_rmsd 75%": pocket_rmsd_75.item(),
        f"{prefix}/af2_pocket_rmsd": af2_pocket_rmsd.mean().item()
    }
    metrics.update(eval_dict)

    return metrics


def obtain_best_result_epoch(epoch, best_result, best_epoch, metrics_test, prefix):
    pocket_rmsd = metrics_test[f"{prefix}/pocket_rmsd"]
    pocket_rmsd_2A = metrics_test[f"{prefix}/pocket_rmsd < 2A"]
    pocket_rmsd_5A = metrics_test[f"{prefix}/pocket_rmsd < 5A"]
    pocket_rmsd_25 = metrics_test[f"{prefix}/pocket_rmsd 25%"]
    pocket_rmsd_50 = metrics_test[f"{prefix}/pocket_rmsd 50%"]
    pocket_rmsd_75 = metrics_test[f"{prefix}/pocket_rmsd 75%"]

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


def define_best_dict():
    best_result = {
        "pocket_rmsd": 1000, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
        "pocket_rmsd 25%": 1000, "pocket_rmsd 50%": 1000, "pocket_rmsd 75%": 1000,
    }
    best_epoch = {
        "pocket_rmsd": 0, "pocket_rmsd < 2A": 0, "pocket_rmsd < 5A": 0,
        "pocket_rmsd 25%": 0, "pocket_rmsd 50%": 0, "pocket_rmsd 75%": 0,
    }
    return best_result, best_epoch
