
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
from models.model_two import get_model
from utils.logging_utils import Logger
from utils.evaluate_two_utils import evaluate_one
from utils.evaluate_utils import obtain_best_result_epoch, define_best_dict
from utils.parsing import parse_args
from utils.training_two import train_one_epoch

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args, parser = parse_args(description="Joint Learning")
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
set_seed(args.seed)

accelerator.print(args)

pre = f"{args.resultFolder}/{args.exp_name}"
if accelerator.is_main_process and args.wandb:
    wandb.init(
        project=f"(FABind+Pretrained-ReviseCoo)",
        name=args.exp_name,
        group=args.resultFolder.split('/')[-1],
        id=args.exp_name,
        config=args
    )
accelerator.wait_for_everyone()

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
train, valid, test = get_data(args, logger=logger, flag=3)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

num_worker = 10
train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_worker)
valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_worker)
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_worker)

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# #################### Define Model ########################
model = get_model(args, logger)
if not args.not_load_ckpt:
    # 加载pretrained model
    logger.log_message("=================== Loading Pretrained FABind+ Model ===================")
    lig_pretrained_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
    pro_pretrained_dict = torch.load(args.ckpt_protein, map_location=torch.device('cpu'))
    if args.copy_two == 1:
        logger.log_message('=================== [lig-poc, pre-lig, pre-lig] ===================')
        model.load_state_dict(lig_pretrained_dict, strict=False)
        complex_model_dict = {k.replace('complex_model.', ''): v.detach() for k, v in lig_pretrained_dict.items() if k.startswith('complex_model.')}
        model.complex_model_pro.load_state_dict(complex_model_dict)
    elif args.copy_two == 2:
        logger.log_message('=================== [lig-poc, pre-lig, pre-pro] ===================')
        model.load_state_dict(lig_pretrained_dict, strict=False)
        complex_model_dict = {k.replace('complex_model.', ''): v for k, v in pro_pretrained_dict.items() if k.startswith('complex_model.')}
        model.complex_model_pro.load_state_dict(complex_model_dict)
    elif args.copy_two == 3:
        logger.log_message('=================== [lig-poc, pre-lig, random-pro] ===================')
        model.load_state_dict(lig_pretrained_dict, strict=False)
    elif args.copy_two == 4:
        logger.log_message('=================== [lig-poc, random-lig, pre-pro] ===================')
        lig_pocket_model_dict = {k.replace('pocket_pred_model.', ''): v for k, v in lig_pretrained_dict.items() if k.startswith('pocket_pred_model.')}
        model.pocket_pred_model.load_state_dict(lig_pocket_model_dict)
        pro_complex_model_dict = {k.replace('complex_model.', ''): v for k, v in pro_pretrained_dict.items() if k.startswith('complex_model.')}
        model.complex_model_pro.load_state_dict(pro_complex_model_dict)
    elif args.copy_two == 5:
        logger.log_message('=================== [pro-poc, random-lig, pre-pro] ===================')
        model.load_state_dict(pro_pretrained_dict, strict=False)
    elif args.copy_two == 6:
        logger.log_message('=================== [random-poc, pre-lig, pre-pro] ===================')
        lig_complex_model_dict = {k.replace('complex_model.', ''): v for k, v in lig_pretrained_dict.items() if k.startswith('complex_model.')}
        pro_complex_model_dict = {k.replace('complex_model.', ''): v for k, v in pro_pretrained_dict.items() if k.startswith('complex_model.')}
        model.complex_model.load_state_dict(lig_complex_model_dict)
        model.complex_model_pro.load_state_dict(pro_complex_model_dict)
    else:
        logger.log_message('=================== load whole model ===================')
        whole_pretrained_dict = torch.load(args.ckpt_whole, map_location=torch.device('cpu'))
        model.load_state_dict(whole_pretrained_dict, strict=True)

    logger.log_message("=================== Loading Pretrained FABind+ Model Finished ===================")
    if args.freeze_pocket_model:
        logger.log_message("=================== Freeze Pocket Model ===================")
        for param in model.pocket_pred_model.parameters():
            param.requires_grad = False
        for param in model.compound_linear_whole_protein.parameters():
            param.requires_grad = False
        for param in model.protein_linear_whole_protein.parameters():
            param.requires_grad = False
        for param in model.protein_to_pocket.parameters():
            param.requires_grad = False
        for param in model.pocket_radius_head.parameters():
            param.requires_grad = False
        logger.log_message("=================== Pocket Model is frozen ===================")

if args.load_whole_pretrained_model:
    logger.log_message('=================== load whole model ===================')
    whole_pretrained_dict = torch.load(args.ckpt_whole, map_location=torch.device('cpu'))
    model.load_state_dict(whole_pretrained_dict, strict=True)
    if args.freeze_ligandDock_pocketPred:
        logger.log_message("=================== Freeze Ligand Dockign model and Pocket prediction model ===================")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.complex_model_pro.parameters():
            param.requires_grad = True
        for param in model.distmap_mlp.parameters():
            param.requires_grad = True

if accelerator.is_main_process:
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message(f"======= Model trainable parameters: {model_trainable_params}")
    model_params = sum(p.numel() for p in model.parameters())
    logger.log_message(f"======= Model parameters: {model_params}")
# #########################################################

# define loss function
pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
pocket_center_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)
pocket_radius_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)
contact_criterion = nn.MSELoss()
if args.permutation_invariant:
    com_coord_criterion = nn.SmoothL1Loss(reduction='none')
else:
    com_coord_criterion = nn.SmoothL1Loss()
pro_coord_criterion = nn.SmoothL1Loss()

output_last_joint_epoch_dir = f"{pre}/models/epoch_last"
if not os.path.exists(output_last_joint_epoch_dir):
    os.system(f"mkdir -p {output_last_joint_epoch_dir}")
# if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "pytorch_model.bin")):
#     accelerator.load_state(output_last_epoch_dir)
#     last_epoch = round(scheduler.state_dict()['last_epoch'] / steps_per_epoch) - 1
#     logger.log_message(f'Load model from epoch: {last_epoch}')

(pro_best_result_gt, pro_best_epoch_gt, pro_best_result_pp, pro_best_epoch_pp,
 joi_best_result_gt, joi_best_epoch_gt, joi_best_result_pp, joi_best_epoch_pp) = define_best_dict(require_pretrain=True)

logger.log_message("+++++++++++++++++++++++++++++++++++ Joint Tuning ++++++++++++++++++++++++++++++++++++++++++")

# #################################### Iter optimizer ############################################
# optimizer
optimizer_joint = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler
total_joint_steps = args.joint_epochs * len(train_loader)
scheduler_warm_up = torch.optim.lr_scheduler.LinearLR(
    optimizer_joint,
    start_factor=0.5,
    end_factor=1,
    total_iters=args.warmup_epochs * len(train_loader),
    last_epoch=last_epoch,
)
scheduler_post = torch.optim.lr_scheduler.LinearLR(optimizer_joint, start_factor=1.0, end_factor=0.0,
                                                   total_iters=(args.joint_epochs - args.warmup_epochs) * len(train_loader), last_epoch=last_epoch)
scheduler_joint = torch.optim.lr_scheduler.SequentialLR(
    optimizer_joint,
    schedulers=[scheduler_warm_up, scheduler_post],
    milestones=[args.warmup_epochs * len(train_loader)],
)

model, optimizer_joint, scheduler_joint, train_loader = accelerator.prepare(
    model, optimizer_joint, scheduler_joint, train_loader
)
logger.log_message(f"Joint epochs: {total_joint_steps}")
args.protein_epochs = 0
torch.cuda.empty_cache()
accelerator.wait_for_everyone()
for epoch in range(last_epoch + 1, args.joint_epochs):
    model.train()
    prefix = "train"
    if args.stage == 1:
        metrics = train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer_joint, scheduler_joint,
                                  pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                                  contact_criterion, com_coord_criterion, pro_coord_criterion,
                                  accelerator.device, stage=1, flag=3, prefix=prefix)
    else:
        metrics = train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer_joint,
                                  scheduler_joint,
                                  pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                                  contact_criterion, com_coord_criterion, pro_coord_criterion,
                                  accelerator.device, stage=2, flag=3, prefix=prefix)

    if accelerator.is_main_process and args.wandb:
        wandb.log(metrics, step=epoch + args.protein_epochs)
    logger.log_stats(metrics, epoch, args, prefix=prefix, pretrain=False)
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    model.eval()
    if accelerator.is_main_process:
        if not args.disable_validate:
            logger.log_message(f"Begin Validation")
            prefix = "valid-gt"
            metrics_val_gt = evaluate_one(accelerator, args, epoch, valid_loader, accelerator.unwrap_model(model),
                                          device, pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                                          contact_criterion, com_coord_criterion, pro_coord_criterion,
                                          prefix=prefix, stage=1, flag=3)
            logger.log_stats(metrics_val_gt, epoch, args, prefix=prefix, pretrain=False)
            prefix = "valid-pp"
            metrics_val_pp = evaluate_one(accelerator, args, epoch, valid_loader, accelerator.unwrap_model(model),
                                          device, pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                                          contact_criterion, com_coord_criterion, pro_coord_criterion,
                                          prefix=prefix, stage=2, flag=3)
            logger.log_stats(metrics_val_pp, epoch, args, prefix=prefix, pretrain=False)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.log_message(f"Begin Test")
        prefix = "test-gt"
        metrics_test_gt = evaluate_one(accelerator, args, epoch, test_loader, accelerator.unwrap_model(model),
                                      device, pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                                      contact_criterion, com_coord_criterion, pro_coord_criterion,
                                      prefix=prefix, stage=1, flag=3)
        logger.log_stats(metrics_test_gt, epoch, args, prefix=prefix)
        joi_best_result_gt, joi_best_epoch_gt = obtain_best_result_epoch(epoch, joi_best_result_gt, joi_best_epoch_gt, metrics_test_gt, prefix)
        logger.log_stats(joi_best_result_gt, epoch, args, prefix="best_test_gt-joint", pretrain=False)
        logger.log_stats(joi_best_epoch_gt, epoch, args, prefix="best_test_gt-joint", pretrain=False)
        if args.wandb:
            wandb.log(metrics_test_gt, step=epoch + args.protein_epochs)
        prefix = "test-pp"
        metrics_test_pp = evaluate_one(accelerator, args, epoch, test_loader, accelerator.unwrap_model(model),
                                      device, pocket_cls_criterion, pocket_center_criterion, pocket_radius_criterion,
                                      contact_criterion, com_coord_criterion, pro_coord_criterion,
                                      prefix=prefix, stage=2, flag=3)
        logger.log_stats(metrics_test_pp, epoch, args, prefix=prefix, pretrain=False)
        joi_best_result_pp, joi_best_epoch_pp = obtain_best_result_epoch(epoch, joi_best_result_pp, joi_best_epoch_pp, metrics_test_pp, prefix)
        logger.log_stats(joi_best_result_pp, epoch, args, prefix="best_test_pp-joint")
        logger.log_stats(joi_best_epoch_pp, epoch, args, prefix="best_test_pp-joint")
        if args.wandb:
            wandb.log(metrics_test_pp, step=epoch + args.protein_epochs)

        # if (epoch + 1) % args.save_ckp_interval == 0:
        output_dir = f"{pre}/models/epoch_{epoch}"
        logger.log_message(f"save path: {output_dir}")
        # accelerator.save_state(output_dir=output_dir, safe_serialization=False)
        # accelerator.save_state(output_dir=output_last_joint_epoch_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        torch.save(accelerator.unwrap_model(model).state_dict(), f"{output_dir}/pytorch_model.bin")
        torch.save(accelerator.unwrap_model(model).state_dict(), f"{output_last_joint_epoch_dir}/pytorch_model.bin")

    accelerator.wait_for_everyone()


























