
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

from models.inference_data import get_data
from models.model_two import get_model
from utils.logging_utils import Logger
from utils.inference_utils import write_sdf_file
from utils.parsing import parse_args
from utils.post_optim_utils import post_optimize_compound_coords
from utils.post_optim_utils import compute_RMSD

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args, parser = parse_args(description="Joint Learning")
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
set_seed(args.seed)

# post_optim parameters
args.post_optim = True
args.post_optim_mode = 0
args.post_optim_epoch = 1000

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


def post_optim_mol(args, pdb_list, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list,
                   com_coord_offset_per_sample_list, com_coord_per_sample_list, compound_batch, rmsd_list, before_rmsd_list,
                   gt_ligand_coord_list, LAS_tmp, rigid=False):
    post_optim_device='cpu'

    for i in range(compound_batch.max().item() + 1):
        try:
            i_mask = (compound_batch == i)
            com_coord_pred_i = com_coord_pred[i_mask]
            com_coord_i = data[i]['compound'].rdkit_coords - data[i]['compound'].rdkit_coords.mean(dim=0)
            com_coord_gt_i = data[i].coords

            com_coord_pred_center_i = com_coord_pred_i.mean(dim=0).reshape(1, 3)

            if args.post_optim:
                predict_coord, loss, rmsd, before_rmsd = post_optimize_compound_coords(
                    reference_compound_coords=com_coord_i.to(post_optim_device),
                    predict_compound_coords=com_coord_pred_i.to(post_optim_device),
                    gt_compound_coords=com_coord_gt_i.to(post_optim_device),
                    # LAS_edge_index=(data[i]['complex', 'LAS', 'complex'].edge_index - data[i]['complex', 'LAS', 'complex'].edge_index.min()).to(post_optim_device),
                    LAS_edge_index=LAS_tmp[i].to(post_optim_device),
                    mode=args.post_optim_mode,
                    total_epoch=args.post_optim_epoch,
                )
                predict_coord = predict_coord.to(accelerator.device)
                predict_coord = predict_coord - predict_coord.mean(dim=0).reshape(1, 3) + com_coord_pred_center_i
                com_coord_pred[i_mask] = predict_coord
                rmsd_list.append(rmsd)
                before_rmsd_list.append(before_rmsd)

            com_coord_pred_per_sample_list.append(com_coord_pred[i_mask])
            com_coord_per_sample_list.append(com_coord_i)
            com_coord_offset_per_sample_list.append(data[i].coord_offset)
            pdb_list.append(data.pdb[i])
            gt_ligand_coord_list.append(data[i].coords)
        except Exception as e:
            print(f'error pdb: {data.pdb[i]}')
            continue

    return


# test dataset
test = get_data(args, logger=logger, flag=3)
logger.log_message(f"Number data of testset: {len(test)}")

num_worker = 0
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_worker)

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# #################### Define Model ########################
model = get_model(args, logger)

args.ckpt = './ckpt/epoch_235_Ti6/pytorch_model.bin'

logger.log_message(f"=================== Loading Model: {args.ckpt} ===================")
pretrained_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
model.load_state_dict(pretrained_dict, strict=True)

logger.log_message("=================== Loading Model Finished ===================")

if accelerator.is_main_process:
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message(f"======= Model trainable parameters: {model_trainable_params}")
# #########################################################

logger.log_message("+++++++++++++++++++++++++++++++++++ Joint Tuning ++++++++++++++++++++++++++++++++++++++++++")

model = accelerator.prepare(model)
model = accelerator.unwrap_model(model)

args.protein_epochs = 0
torch.cuda.empty_cache()
accelerator.wait_for_everyone()

pattern = r"epoch_(\d+)"
save_epoch = re.search(pattern, args.ckpt)
save_epoch = save_epoch.group()
args.save_rmsd = True
write_path = f"./visulazation/write_ligand_epoch235_wo_optim"
args.rmsd_file = f"{write_path}/{save_epoch}/metrics_post_optim.txt"
args.out_path = f"{write_path}/{save_epoch}"
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

init_ligand_coord_list = []
pred_ligand_coord_list = []
ref_ligand_coord_list = []
offset_coord_list = []
ligand_rmsd_list, before_rmsd_list = [], []
pdb_list = []

model.eval()
if accelerator.is_main_process:
    logger.log_message(f"Begin Test")
    for data in tqdm(test_loader):
        data = data.to(device)
        pdb_list.extend(data.pdb)
        LAS_tmp = []
        for i in range(len(data)):
            LAS_tmp.append(data[i]['compound', 'LAS', 'compound'].edge_index.detach().clone())
        with torch.no_grad():
            (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
             pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
             pred_pocket_center, pocket_radius_pred, gt_pocket_batch) = model.inference(data, stage=2)
        for i in range(compound_batch.max().item() + 1):
            ref_ligand_coord_list.append(data.coords[compound_batch == i])
            pred_ligand_coord_list.append(compound_coords_pred[compound_batch == i])
            init_ligand_coord_list.append(data['compound'].rdkit_coords[compound_batch == i] - data['compound'].rdkit_coords[compound_batch == i].mean(dim=0))
            offset_coord_list.append(data.coord_offset[i].unsqueeze(0))

            rmsd = compute_RMSD(data.coords[compound_batch == i], compound_coords_pred[compound_batch == i])
            ligand_rmsd_list.append(rmsd.item())

    args.write_mol_to_file = False
    if args.write_mol_to_file:
        ori_sdf_files = []
        for pdb in pdb_list:
            ori_sdf_files.append(f"./binddataset/test_ligand/{pdb}.sdf")
        pred_ligand_coord_list = [x.cpu().numpy() for x in pred_ligand_coord_list]
        ref_ligand_coord_list = [x.cpu().numpy() for x in ref_ligand_coord_list]
        init_ligand_coord_list = [x.cpu().numpy() for x in init_ligand_coord_list]
        offset_coord_list = [x.cpu().numpy() for x in offset_coord_list]

        write_sdf_file(pdb_list, ori_sdf_files, pred_ligand_coord_list, ref_ligand_coord_list, init_ligand_coord_list, offset_coord_list, args.out_path)

    rmsd_array = np.array(ligand_rmsd_list)
    mean_rmsd = np.mean(rmsd_array)
    rmsd_2A_array = rmsd_array < 2
    rmsd_5A_array = rmsd_array < 5
    rmsd_25 = np.percentile(rmsd_array, 25)
    rmsd_50 = np.percentile(rmsd_array, 50)
    rmsd_75 = np.percentile(rmsd_array, 75)
    rmsd_2A = np.sum(rmsd_2A_array) / rmsd_2A_array.shape[0]
    rmsd_5A = np.sum(rmsd_5A_array) / rmsd_5A_array.shape[0]

    print(f"rmsd: {mean_rmsd}, rmsd < 2A: {rmsd_2A}, rmsd < 5A: {rmsd_5A}, rmsd 25%: {rmsd_25}, rmsd 50%: {rmsd_50}, rmsd 75%: {rmsd_75}")
