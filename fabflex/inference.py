
import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_scatter import scatter_mean
from torch.utils.data import RandomSampler
import re
import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed

from data import get_data
from models.model_two import get_model
from utils.logging_utils import Logger
from utils.inference_utils import inference_one, write_pdb_file, write_sdf_file, write_pocket_info
from utils.evaluate_utils import obtain_best_result_epoch, define_best_dict
from utils.parsing import parse_args
from utils.training_two import train_one_epoch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args, parser = parse_args(description="Joint Learning")
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
set_seed(args.seed)

accelerator.print(args)

pre = f"{args.resultFolder}/{args.exp_name}"

if accelerator.is_main_process:
    os.system(f"mkdir -p {pre}/models")

accelerator.wait_for_everyone()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logger = Logger(accelerator=accelerator, log_path=f"{pre}/{timestamp}.log")

logger.log_message(f"{' '.join(sys.argv)}")

# ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')

last_epoch = -1
args.last_epoch = last_epoch

# 获得protein训练阶段的
args.infer_unseen = False
args.unseen_file = '/home/workspace/Dual4Dock/baselines/unseen_test_pdb.txt'
train, valid, test = get_data(args, logger=logger, flag=3)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

num_worker = 0
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_worker)

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# #################### Define Model ########################
args.infer_single_model = True
# if args.infer_single_model:
#     args.n_iter = 1
# else:
#     args.n_iter = 8
model = get_model(args, logger)

args.ckpt = './ckpt/epoch_235_Ti6/pytorch_model.bin'

if args.infer_single_model:
    logger.log_message("=================== Loading Model ===================")
    pretrained_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict, strict=True)
else:
    args.ckpt_ligand = '/ckpt/best_layer5_ckpt.bin'
    args.ckpt_protein = '/ckpt/protein_113_ckpt.bin'
    args.total_iterative = 1
    args.use_iterative = False
    args.use_stepbystey = False
    # args.force_fix_radius = False
    lig_pretrained_dict = torch.load(args.ckpt_ligand, map_location=torch.device('cpu'))
    pro_pretrained_dict = torch.load(args.ckpt_protein, map_location=torch.device('cpu'))
    logger.log_message("=================== Loading Pretrained each module =================== ")
    model.load_state_dict(lig_pretrained_dict, strict=False)
    complex_model_dict = {k.replace('complex_model.', ''): v for k, v in pro_pretrained_dict.items() if
                          k.startswith('complex_model.')}
    model.complex_model_pro.load_state_dict(complex_model_dict)

logger.log_message("=================== Loading Model Finished ===================")

if accelerator.is_main_process:
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message(f"======= Model trainable parameters: {model_trainable_params}")
# #########################################################

(pro_best_result_gt, pro_best_epoch_gt, pro_best_result_pp, pro_best_epoch_pp,
 joi_best_result_gt, joi_best_epoch_gt, joi_best_result_pp, joi_best_epoch_pp) = define_best_dict(require_pretrain=True)

logger.log_message("+++++++++++++++++++++++++++++++++++ Joint Tuning ++++++++++++++++++++++++++++++++++++++++++")

model = accelerator.prepare(model)
args.protein_epochs = 0
torch.cuda.empty_cache()
accelerator.wait_for_everyone()

epoch = 0

pattern = r"epoch_(\d+)"
save_epoch = re.search(pattern, args.ckpt)

args.noise_protein_rate = 0
args.noise_protein = 0
args.save_rmsd = False
args.save_file = f"result/{save_epoch.group()}/using_p2rank.txt"
args.save_file = f"result/{save_epoch.group()}/metric_testgt.txt"


args.out_path = f"result/{save_epoch.group()}"
if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)
model.eval()
if accelerator.is_main_process:
    logger.log_message(f"Begin Test")

    prefix = "test-pp"
    metrics_test_pp, pdb_list, rmsd_pp, pocket_rmsd_pp, af2_pocket_rmsd, cls_acc_pp, pred_keepNode_list, gt_keepNode_list = inference_one(accelerator, args, epoch, test_loader,
                                                                          accelerator.unwrap_model(model), device,
                                                                          prefix=prefix, stage=2, flag=3)
    pocket_cls_accuracy_1 = np.mean(cls_acc_pp)

    prefix = "test-gt"
    metrics_test_gt, _, rmsd_gt, pocket_rmsd_gt, _, cls_acc_gt, _, _ = inference_one(accelerator, args, epoch, test_loader,
                                                                   accelerator.unwrap_model(model), device,
                                                                   prefix=prefix, stage=1, flag=3)

    if args.save_keepNode:
        if not os.path.exists(args.keep_path):
            os.mkdir(args.keep_path)

        for pdb, pred_keep, gt_keep in zip(pdb_list, pred_keepNode_list, gt_keepNode_list):
            keepNode_file = f"{args.keep_path}/{pdb}_keep.txt"
            with open(keepNode_file, 'w') as f:
                pred_idx_list = []
                gt_idx_list = []
                for j in range(len(pred_keep)):
                    if pred_keep[j]:
                        pred_idx_list.append(str(j))
                    if gt_keep[j]:
                        gt_idx_list.append(str(j))
                pred_residue = " ".join(pred_idx_list)
                gt_residue = " ".join(gt_idx_list)
                f.write(pred_residue + "\n")
                f.write(gt_residue + "\n")

    if args.save_rmsd:
        with (open(args.rmsd_file, "w") as f):
            print(f"save_rmsd_file: {args.save_file}")
            f.write("pdb\tligand_rmsd_pp\tpocket_rmsd_pp\taf2_pocket_rmsd\tligand_rmsd_gt\tpocket_rmsd_gt\tMAE\tPocket_RMSE\tPocket_dis\tPocket_cls_acc\n")
            for i in range(len(pdb_list)):
                pdb = pdb_list[i]
                rmsd_value = rmsd_pp[i].item()
                pocket_rmsd_value = pocket_rmsd_pp[i].item()
                af2_pocket_rmsd_value = af2_pocket_rmsd[i].item()
                rmsd_value_gt = rmsd_gt[i].item()
                pocket_rmsd_value_gt = pocket_rmsd_gt[i].item()

                mae_value = metrics_test_pp[f'test-pp/pocket_mae'].item()
                pocket_center_dis = metrics_test_pp[f'test-pp/pocket_center_dist'][i]
                cls_acc_value = cls_acc_pp[i]
                rmse_value = metrics_test_pp[f'test-pp/pocket_rmse'].item()
                out = str(pdb) + "\t" + str(rmsd_value) + "\t" + str(pocket_rmsd_value) + "\t" + str(af2_pocket_rmsd_value) + "\t" + \
                      str(rmsd_value_gt) + "\t" + str(pocket_rmsd_value_gt) + "\t" + \
                      str(mae_value) + "\t" + str(rmse_value) + '\t' + str(pocket_center_dis) + "\t" + str(cls_acc_value) + "\n"
                f.write(out)
            count_less_pp = 0
            count_less_gt = 0
            for i in range(len(pdb_list)):
                pocket_rmsd_value = pocket_rmsd_pp[i].item()
                pocket_rmsd_value_gt = pocket_rmsd_gt[i].item()
                af2_pocket_rmsd_value = af2_pocket_rmsd[i].item()
                if pocket_rmsd_value < af2_pocket_rmsd_value:
                    count_less_pp += 1
                if pocket_rmsd_value_gt < af2_pocket_rmsd_value:
                    count_less_gt += 1

            less_ratio_pp = count_less_pp / len(pdb_list)
            less_ratio_gt = count_less_gt / len(pdb_list)
            print(f'less ratio_pp: {less_ratio_pp}')
            print(f'less ratio_gt: {less_ratio_gt}')
            rmsd_2A = (rmsd_pp < 2).sum() / rmsd_pp.shape[0]
            rmsd_5A = (rmsd_pp < 5).sum() / rmsd_pp.shape[0]
            rmsd_25 = torch.quantile(rmsd_pp, 0.25)
            rmsd_50 = torch.quantile(rmsd_pp, 0.50)
            rmsd_75 = torch.quantile(rmsd_pp, 0.75)

            pocket_rmsd_05A = (pocket_rmsd_pp < 0.5).sum() / pocket_rmsd_pp.shape[0]
            pocket_rmsd_1A = (pocket_rmsd_pp < 1).sum() / pocket_rmsd_pp.shape[0]
            pocket_rmsd_25 = torch.quantile(pocket_rmsd_pp, 0.25)
            pocket_rmsd_50 = torch.quantile(pocket_rmsd_pp, 0.50)
            pocket_rmsd_75 = torch.quantile(pocket_rmsd_pp, 0.75)
            f.write(f"Ligand RMSD-pp: {rmsd_pp.mean().item()}, RMSD<2A: {rmsd_2A.item()}, RMSD<5A: {rmsd_5A.item()}, rmsd 25%: {rmsd_25}, rmsd 50%: {rmsd_50}, rmsd 75%: {rmsd_75}, "
                    f"Pocket RMSD-pp: {pocket_rmsd_pp.mean().item()}, Pocket-RMSD<0.5A: {pocket_rmsd_05A.item()}, Pocket-RMSD<1A: {pocket_rmsd_1A.item()}, pocket-rmsd 25%: {pocket_rmsd_25}, pocket-rmsd 50%: {pocket_rmsd_50}, pocket-rmsd 75%: {pocket_rmsd_75}, "
                    f"Af2 Pocket RMSD: {af2_pocket_rmsd.mean().item()}, "
                    f"Ligand RMSD-gt: {rmsd_gt.mean().item()}, Pocket RMSD-gt: {pocket_rmsd_gt.mean().item()}, "
                    f"MAE: {metrics_test_pp[f'test-pp/pocket_mae']}, RMSE: {metrics_test_pp[f'test-pp/pocket_rmse']},  Pocket_center_avg_dis: {metrics_test_pp[f'test-pp/pocket_center_avg_dist']}, "
                    f"DCC<3: {metrics_test_pp[f'test-pp/DCC_3']}, DCC<4: {metrics_test_pp[f'test-pp/DCC_4']}, DCC<5: {metrics_test_pp[f'test-pp/DCC_5']}, "
                    f"pocket_cls_accuracy: {pocket_cls_accuracy_1}, less_ratio-pp: {less_ratio_pp}, less_ratio-gt: {less_ratio_gt}")

    del metrics_test_pp[f'test-pp/pocket_center_dist']
    logger.log_stats(metrics_test_pp, epoch, args, prefix='test-pp', pretrain=False)
    del metrics_test_gt[f'test-gt/pocket_center_dist']
    logger.log_stats(metrics_test_gt, epoch, args, prefix='test-gt', pretrain=False)

