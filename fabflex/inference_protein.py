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
from utils.inference_utils import write_single_pdb_file
from utils.parsing import parse_args
from utils.post_optim_utils import post_optimize_compound_coords

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

# 获得test dataset
test = get_data(args, logger=logger, flag=3)
logger.log_message(f"Number data of testset: {len(test)}")

num_worker = 0
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False,
                         pin_memory=False, num_workers=num_worker)

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# #################### Define Model ########################
model = get_model(args, logger)

args.ckpt = './ckpt/FABFlex_model.bin'

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
args.out_path = f"./results/write_pocket"
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

pred_protein_coords_list = []
offset_coord_list = []
pocket_rmsd_list = []
pdb_list = []

model.eval()
if accelerator.is_main_process:
    logger.log_message(f"Begin Test")
    for data in test_loader:
        pdb_list.extend(data.pdb)
        data = data.to(device)
        with torch.no_grad():
            (compound_coords_pred, compound_batch, pocket_coords_pred, pocket_batch,
             pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole,
             pred_pocket_center, pocket_radius_pred, gt_pocket_batch) = model.inference(data, stage=2)

        data.input_protein_node_xyz[data['pocket'].keepNode] = pocket_coords_pred
        protein_batch = data['protein_whole'].batch
        for i in range(compound_batch.max().item() + 1):
            pred_protein_coords_list.append(data.input_protein_node_xyz[protein_batch == i])
            offset_coord_list.append(data.coord_offset[i])
    print(f'len pdb_list: {len(pdb_list)}')
    args.write_protein_to_pdb = True
    if args.write_protein_to_pdb:
        ori_pdb_files = []
        pdb_directory_path = './binddataset/test_af2pdb'
        for pdb in pdb_list:
            found = False
            for file_name in os.listdir(pdb_directory_path):
                if pdb in file_name:
                    full_path = os.path.join(pdb_directory_path, file_name)
                    ori_pdb_files.append(full_path)
        pred_protein_coords_list = [x.cpu().numpy() for x in pred_protein_coords_list]
        offset_coord_list = [x.cpu().numpy() for x in offset_coord_list]

        write_single_pdb_file(pdb_list, ori_pdb_files, pred_protein_coords_list, offset_coord_list, args.out_path)

