import torchmetrics
import torch.nn.functional as F


def pocket_metrics(pocket_coord_pred, pocket_coord, prefix="train"):
    pearson_x = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    rmse_x = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 0], pocket_coord[:, 0], squared=False)
    mae_x = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    pearson_y = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    rmse_y = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 1], pocket_coord[:, 1], squared=False)
    mae_y = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    pearson_z = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    rmse_z = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 2], pocket_coord[:, 2], squared=False)
    mae_z = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    pearson = (pearson_x + pearson_y + pearson_z) / 3
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    mae = (mae_x + mae_y + mae_z) / 3
    pocket_pairwise_dist = F.pairwise_distance(pocket_coord_pred, pocket_coord, p=2)
    DCC = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
    return {f"{prefix}/pocket_pearson": pearson, f"{prefix}/pocket_rmse": rmse, f"{prefix}/pocket_mae": mae,
            f"{prefix}/pocket_center_avg_dist": pocket_pairwise_dist.mean().item(), f"{prefix}/pocket_center_DCC": DCC * 100}


def pocket_metrics_each_sample(pocket_coord_pred, pocket_coord, prefix="train"):
    pearson_x = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    rmse_x = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 0], pocket_coord[:, 0], squared=False)
    mae_x = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    pearson_y = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    rmse_y = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 1], pocket_coord[:, 1], squared=False)
    mae_y = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    pearson_z = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    rmse_z = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 2], pocket_coord[:, 2], squared=False)
    mae_z = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    pearson = (pearson_x + pearson_y + pearson_z) / 3
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    mae = (mae_x + mae_y + mae_z) / 3
    pocket_pairwise_dist = F.pairwise_distance(pocket_coord_pred, pocket_coord, p=2)
    DCC_3 = (pocket_pairwise_dist < 3).sum().item() / len(pocket_pairwise_dist)
    DCC_4 = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
    DCC_5 = (pocket_pairwise_dist < 5).sum().item() / len(pocket_pairwise_dist)
    return {f"{prefix}/pocket_pearson": pearson, f"{prefix}/pocket_rmse": rmse, f"{prefix}/pocket_mae": mae,
            f"{prefix}/pocket_center_dist": pocket_pairwise_dist.cpu().detach().numpy().tolist(),
            f"{prefix}/pocket_center_avg_dist": pocket_pairwise_dist.mean().item(),
            f"{prefix}/DCC_3": DCC_3 * 100, f"{prefix}/DCC_4": DCC_4 * 100, f"{prefix}/DCC_5": DCC_5 * 100,}

