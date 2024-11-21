import argparse


def parse_args(description=""):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--root_path", type=str, default="./binddataset", help="root paht of the data.")
    parser.add_argument("--ckpt", type=str, default='binddataset/ckpt/fabind_plus_best_ckpt.bin')
    parser.add_argument("--ckpt_protein", type=str, default='binddataset/ckpt/protein_model.bin')
    parser.add_argument("--ckpt_whole", type=str, default='binddataset/ckpt/protein_model.bin')
    parser.add_argument("--not_load_ckpt", action='store_true', default=False)
    parser.add_argument("--load_whole_pretrained_model", action='store_true', default=False)
    parser.add_argument("--copy_two", type=int, default=2, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--use_iterative", action='store_true', default=False)
    parser.add_argument("--use_stepbystep", action='store_true', default=False)
    parser.add_argument("--wodm", action='store_true', default=False)
    parser.add_argument("--wopr", action='store_true', default=False)
    parser.add_argument("--freeze_pocket_model", action='store_true', default=False)
    parser.add_argument("--freeze_ligandDock_pocketPred", action='store_true', default=False)
    parser.add_argument("--shift_coord", action='store_true', default=False)
    parser.add_argument("--pocket_flag", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])

    # Parameters of get_data()
    parser.add_argument("--pocket_radius", type=int, default=20, help="radius to verify the pocket nodes: default 20 A")
    parser.add_argument("--pocket_radius_infer", type=int, default=20, help="radius to verify the pocket nodes: default 20 A")
    parser.add_argument("--includeDisMap", type=bool, default=True,
                        help="whether include the distance map between atoms for a ligand")
    parser.add_argument("--dis_map_thres", type=float, default=15.0, help="threshold of cutting off distance map")
    parser.add_argument("--addNoise", type=float, default=0, help="add noise to centroid for keepNode")
    parser.add_argument('--pocket_idx_no_noise', type=bool, default=True)

    # Parameters of orgnizing training
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='no', choices=['no', 'fp16'])
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--protein_epochs", type=int, default=8, help="training epoch")
    parser.add_argument("--joint_epochs", type=int, default=8, help="training epoch")
    parser.add_argument("--total_iterative", type=int, default=4, help="training iteration")
    parser.add_argument("--random_total_iter", type=bool, default=True, help="whether random number of recycle for training")
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument("--sample_n", type=int, default=0, help="number of samples in one epoch.")
    parser.add_argument('--clip_grad', action='store_true', default=False)
    parser.add_argument("--cut_train_set", action='store_true', default=False)
    parser.add_argument("--use_p2rank", action='store_true', default=False)

    # Parameters of coding model
    parser.add_argument("--model_mode", type=int, default=0,
                        help="[0, 1, 2] to determine which model is used for training")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pocket_pred_hidden_size", type=int, default=128)
    parser.add_argument("--coord_scale", type=float, default=5.0, help="coefficient to normalize the coords")
    parser.add_argument("--mean_layers", type=int, default=5, help="number of layers for mean.")
    parser.add_argument("--pocket_pred_layers", type=int, default=1, help="number of layers for pocket pred model.")
    parser.add_argument('--inter_cutoff', type=float, default=10.0)
    parser.add_argument('--intra_cutoff', type=float, default=8.0)
    parser.add_argument('--n_iter', type=int, default=1, help='number of recycle')
    parser.add_argument('--pocket_pred_n_iter', type=int, default=1, help='number of recycle for pocket layer')
    parser.add_argument("--random_n_iter", type=bool, default=True, help="whether random number of recycle for training")
    parser.add_argument('--norm_type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])
    parser.add_argument('--geometry_reg_step_size', type=float, default=0.001)
    parser.add_argument('--geometry_reg_step', type=int, default=1)
    parser.add_argument("--mha_heads", type=int, default=4)
    parser.add_argument("--rel_dis_pair_bias", type=str, default='no', choices=['no', 'add', 'mul'])
    parser.add_argument("--mlp_hidden_scale", type=int, default=1)
    parser.add_argument("--gs_tau", type=float, default=1, help="Tau for the temperature-based softmax.")
    parser.add_argument("--gs_hard", action='store_true', default=False, help="Hard mode for gumbel softmax.")

    parser.add_argument("--permutation_invariant", action='store_true', default=False)
    parser.add_argument("--coord_loss_weight", type=float, default=1.0)
    parser.add_argument("--pro_loss_weight", type=float, default=1.0)
    parser.add_argument("--pocket_cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--pocket_distance_loss_weight", type=float, default=0.05)
    parser.add_argument("--pocket_radius_loss_weight", type=float, default=0.05)
    parser.add_argument("--pair_distance_loss_weight", type=float, default=1.0)
    parser.add_argument("--pair_distance_distill_loss_weight", type=float, default=1.0)
    parser.add_argument("--pocket_coord_huber_delta", type=float, default=3.0)
    parser.add_argument("--com_loss_function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])
    parser.add_argument("--pro_loss_function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])

    parser.add_argument('--add_attn_pair_bias', type=bool, default=True)
    parser.add_argument('--rm_layernorm', action='store_true', default=False)
    parser.add_argument("--use_ln_mlp", action="store_true", default=False)
    parser.add_argument('--keep_trig_attn', type=bool, default=False)
    parser.add_argument('--force_fix_radius', action='store_true', default=False)
    parser.add_argument("--pocket_radius_buffer", type=float, default=5.0,
                        help='if buffer <= 2.0, use multiplication; else use addition')
    parser.add_argument("--min_pocket_radius", type=float, default=20.0)

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gradient_accumulate_step", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=15, help="warm-up epochs")
    parser.add_argument('--lr_scheduler', type=str, default="poly_decay",
                        choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])

    parser.add_argument("--resultFolder", type=str, default="./result", help="information you want to keep a record.")
    parser.add_argument("--exp_name", type=str, default="train_tmp", help="train_tmp / test_exp")
    parser.add_argument('--disable_tensorboard', action='store_true', default=False)
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument('--disable_tqdm', action='store_true', default=False)
    parser.add_argument("--tqdm_interval", type=float, default=0.1, help="tqdm bar update interval")
    parser.add_argument('--disable_validate', action='store_true', default=False)
    parser.add_argument("--save_ckp_interval", type=int, default=2, help="train_tmp / test_exp")

    parser.add_argument('--noise_protein_rate', type=int, default=0)
    parser.add_argument("--noise_protein", type=int, default=0)
    parser.add_argument('--save_keepNode', action='store_true', default=False)
    parser.add_argument('--keep_path', type=str, default='./visulazation/keep')

    parser.add_argument('--save_rmsd', action='store_true', default=False)
    parser.add_argument('--rmsd_file', type=str, default='a.txt')

    args = parser.parse_args()
    return args, parser


