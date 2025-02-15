
import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_scatter import scatter_mean
from torch.utils.data import RandomSampler

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed

import wandb

from data import get_data
from models.model_pro import get_model
from utils.logging_utils import Logger
from utils.evaluate_pro import evaluate_one, obtain_best_result_epoch, define_best_dict
from utils.parsing import parse_args

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

args, parser = parse_args(description="Pretain-Iter")
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
set_seed(args.seed)

accelerator.print(args)

pre = f"{args.resultFolder}/{args.exp_name}"
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logger = Logger(accelerator=accelerator, log_path=f"{pre}/{timestamp}.log")

logger.log_message(f"{' '.join(sys.argv)}")

# ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')

last_epoch = -1
args.last_epoch = last_epoch
args.noise_protein = 0
args.noise_protein_rate = 0

# 获得protein训练阶段的
args.infer_unseen = False
train, valid, test = get_data(args, logger=logger, flag=3)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

num_worker = 10
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_worker)

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# #################### Define Model ########################
model = get_model(args, logger)
eval_mode = 'PPDM'
if eval_mode == 'PPDM':
    args.ckpt = './ckpt/protein_113_ckpt.bin'
    pretrained_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict, strict=True)
if eval_mode == 'FTPDM':
    args.ckpt = './ckpt/protein_113_ckpt.bin'
    pretrained_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict, strict=False)

    args.ckpt = './epoch_235_Ti6/pytorch_model.bin'
    pretrained_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))

    # 获取模型的当前状态字典
    model_dict = model.state_dict()

    # 创建一个新的字典来存储需要加载的参数
    filtered_pretrained_dict = {}

    # 遍历预训练模型中的所有参数
    for k, v in pretrained_dict.items():
        if k.startswith('complex_model_pro.'):
            # 替换 complex_model_pro 为 complex_model 并检查是否在当前模型中
            new_key = k.replace('complex_model_pro.', 'complex_model.')
            if new_key in model_dict:
                filtered_pretrained_dict[new_key] = v
        # elif k in model_dict:
        #     # 如果键名在当前模型中存在，直接添加
        #     filtered_pretrained_dict[k] = v

    # 更新当前模型的状态字典并加载预训练参数
    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict)


if accelerator.is_main_process:
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message(f"======= Model trainable parameters: {model_trainable_params}")
# #########################################################

# define loss function
contact_criterion = nn.MSELoss()
pro_coord_criterion = nn.SmoothL1Loss()

# #################################### Protein optimizer ############################################
model = accelerator.prepare(model)

logger.log_message(f"Protein epochs: {args.protein_epochs}")

best_result, best_epoch = define_best_dict()
torch.cuda.empty_cache()
# Pretrain stage

epoch = 0
model.eval()
if accelerator.is_main_process:
    logger.log_message(f"Begin Test")
    prefix = "test"
    metrics_test_gt = evaluate_one(accelerator, args, epoch, test_loader, accelerator.unwrap_model(model),
                                   device, contact_criterion, pro_coord_criterion,
                                   prefix=prefix, stage=1, flag=2)
    logger.log_stats(metrics_test_gt, epoch, args, prefix=prefix)




























