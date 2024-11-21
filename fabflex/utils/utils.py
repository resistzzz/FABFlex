import scipy
from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from utils.metrics import pocket_metrics


def compute_dis_between_two_vector(a, b):
    return (((a - b)**2).sum())**0.5


def get_keepNode(com, n_protein_node, protein_node_xyz, pocket_radius, add_noise_to_com):
    keepNode = np.zeros(n_protein_node, dtype=bool)
    if add_noise_to_com:
        com = com + add_noise_to_com * (2 * np.random.rand(*com.shape) - 1)
    for i, node in enumerate(protein_node_xyz):
        dis = compute_dis_between_two_vector(node, com)
        keepNode[i] = dis < pocket_radius
    return keepNode


def get_keepNode_v1(coords, n_protein_node, protein_node_xyz, pocket_radius):
    keepNode = cdist(protein_node_xyz, coords).min(axis=-1) < pocket_radius
    assert keepNode.shape[0] == n_protein_node
    return keepNode


def compute_dis_between_two_vector_tensor(a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=-1))


def get_keepNode_tensor(protein_node_xyz, pocket_radius, chosen_pocket_com):
    # Compute the distances between all nodes and the chosen_pocket_com in a vectorized manner
    dis = compute_dis_between_two_vector_tensor(protein_node_xyz, chosen_pocket_com.unsqueeze(0))
    # Create the keepNode tensor using a boolean mask
    keepNode = dis < pocket_radius

    return keepNode


def construct_data(args, flag, group, protein_node_xyz, protein_seq, af2_protein_node_xyz, af2_protein_seq, protein_esm2_feat,
                   coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, LAS_edge_index, rdkit_coords,
                   add_noise_to_com=None, includeDisMap=True, pocket_idx_no_noise=True, isomorphisms=None, p2rank_center=None):
    n_protein = protein_node_xyz.shape[0]

    if args.noise_protein != 0:
        print('hello')
        if args.noise_protein_type == 'uniform':
            # Generate noise with uniform distribution between -args.noise_protein and args.noise_protein
            noise = (torch.rand(af2_protein_node_xyz.shape,
                                device=af2_protein_node_xyz.device) * 2 - 1) * args.noise_protein
            af2_protein_node_xyz += noise
        else:
            noise = torch.randn(af2_protein_node_xyz.shape, device=af2_protein_node_xyz.device) * args.noise_protein
            af2_protein_node_xyz += noise

    coords_bias = af2_protein_node_xyz.mean(dim=0)
    coords = coords - coords_bias.numpy()
    protein_node_xyz = protein_node_xyz - coords_bias
    af2_protein_node_xyz = af2_protein_node_xyz - coords_bias
    # centroid instead of com.
    com = coords.mean(axis=0)

    # construct heterogeneous graph data.
    data = HeteroData()
    data.isomorphisms = isomorphisms
    data.seq_whole = protein_seq
    data.coords = torch.tensor(coords, dtype=torch.float)
    data.num_atoms = data.coords.shape[0]
    data.coord_offset = coords_bias.unsqueeze(0)
    data.protein_node_xyz = protein_node_xyz
    data.af2_protein_node_xyz = af2_protein_node_xyz

    # max sphere radius
    coords_tensor = torch.tensor(coords)
    center = coords_tensor.mean(dim=0)
    distances = torch.norm(coords_tensor - center, dim=1)
    data.ligand_radius = distances.max()
    data.coords_center = torch.tensor(com, dtype=torch.float).unsqueeze(0)

    # pocket radius
    if args.pocket_radius_buffer <= 2.0:
        pocket_radius = args.pocket_radius_buffer * data.ligand_radius
    else:
        pocket_radius = args.pocket_radius_buffer + data.ligand_radius
    if pocket_radius < args.min_pocket_radius:
        pocket_radius = args.min_pocket_radius

    if args.force_fix_radius:
        pocket_radius = args.pocket_radius

    # keepNode
    keepNode = get_keepNode(com, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=add_noise_to_com)
    keepNode_no_noise = get_keepNode(com, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=None)
    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True

    if p2rank_center is not None:
        poc_center = np.array(p2rank_center) - coords_bias.numpy()
        keepNode = get_keepNode(poc_center, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=add_noise_to_com)

        # ##########################这段代码单纯用来测试一下随机给10%的残基作为pocket的性能
        # # 创建初始的 keepNode 数组
        # keepNode = np.zeros(n_protein, dtype=bool)
        # # 计算需要设置为 1 的数量（10%）
        # num_to_set = int(0.1 * n_protein)
        # # 随机选择 num_to_set 个索引
        # random_indices = np.random.choice(n_protein, num_to_set, replace=False)
        # # 将随机选中的索引处的值设置为 True (1)
        # keepNode[random_indices] = True

    # 添加噪声到pocket部分的坐标，按照概率来添加，
    if args.noise_protein_rate != 0:
        # 首先获取 keepNode 对应的坐标
        pocket_node_xyz = af2_protein_node_xyz[keepNode]

        # 确定哪些坐标需要添加噪声，根据 args.noise_protein_rate 生成随机布尔数组
        # 使用 torch.bernoulli 生成一个随机布尔tensor
        add_noise = torch.bernoulli(
            torch.full((pocket_node_xyz.size(0),), args.noise_protein_rate, device=pocket_node_xyz.device)) > 0

        # 生成噪声，假设噪声为均值 0，标准差为 5 的正态分布
        noise = torch.randn_like(pocket_node_xyz) * 1.0  # 标准正态分布乘以5变为标准差为5

        # 只在选定的坐标上添加噪声
        pocket_node_xyz[add_noise] += noise[add_noise]

        # 将添加了噪声的坐标重新放回 af2_protein_node_xyz
        af2_protein_node_xyz[keepNode] = pocket_node_xyz

    data.pocket_idx = torch.tensor(keepNode_no_noise, dtype=torch.int)
    data.pocket_node_xyz = protein_node_xyz[keepNode]
    data.af2_pocket_node_xyz = af2_protein_node_xyz[keepNode]
    # protein feature map
    if torch.is_tensor(protein_esm2_feat):
        data['protein_whole'].node_feats = protein_esm2_feat
    else:
        raise ValueError("protein_esm2_feat should be a tensor")

    # distance map
    dis_map = scipy.spatial.distance.cdist(protein_node_xyz[keepNode].cpu().numpy(), coords)
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map > args.dis_map_thres] = args.dis_map_thres
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()

    # pocket feature map
    if torch.is_tensor(protein_esm2_feat):
        data['pocket'].node_feats = protein_esm2_feat[keepNode]
    else:
        raise ValueError("protein_esm2_feat should be a tensor")
    data['pocket'].keepNode = torch.tensor(keepNode, dtype=torch.bool)
    data['pocket'].keepNode_noNoise = torch.tensor(keepNode_no_noise, dtype=torch.bool)
    data['compound'].node_feats = compound_node_features.float()
    data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index

    n_pocket = protein_node_xyz[keepNode].shape[0]
    n_protein_whole = protein_node_xyz.shape[0]
    n_compound = compound_node_features.shape[0]
    rdkit_coords = torch.tensor(rdkit_coords)
    data['compound'].rdkit_coords = rdkit_coords

    data['complex'].node_coords_LAS = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3),
            torch.zeros_like(protein_node_xyz[keepNode])
        ), dim=0
    ).float()
    segment = torch.zeros(n_compound + n_pocket + 2)
    segment[n_compound + 1:] = 1  # compound: 0, protein: 1
    data['complex'].segment = segment
    lig_mask = torch.zeros(n_compound + n_pocket + 2)
    lig_mask[:n_compound + 2] = 1  # glb_p can be updated
    data['complex'].lig_mask = lig_mask.bool()
    pro_mask = torch.zeros(n_compound + n_pocket + 2)
    pro_mask[0] = 1
    pro_mask[n_compound + 1:] = 1  # glb_c, glb_p can be updated
    data['complex'].pro_mask = pro_mask.bool()
    is_global = torch.zeros(n_compound + n_pocket + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex"].is_global = is_global.bool()

    data['complex', 'c2c', 'complex'].edge_index = input_atom_edge_list[:, :2].long().t().contiguous() + 1
    data['complex', 'LAS', 'complex'].edge_index = LAS_edge_index + 1

    data['complex_whole_protein'].node_coords_LAS = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3),
            torch.zeros_like(protein_node_xyz)
        ), dim=0
    ).float()

    segment = torch.zeros(n_compound + n_protein_whole + 2)
    segment[n_compound + 1:] = 1  # compound: 0, protein: 1
    data['complex_whole_protein'].segment = segment
    lig_mask = torch.zeros(n_compound + n_protein_whole + 2)
    lig_mask[:n_compound + 2] = 1  # glb_p can be updated?
    data['complex_whole_protein'].lig_mask = lig_mask.bool()
    pro_mask = torch.zeros(n_compound + n_protein_whole + 2)
    pro_mask[0] = 1
    pro_mask[n_compound + 1:] = 1
    data['complex_whole_protein'].pro_mask = pro_mask.bool()
    is_global = torch.zeros(n_compound + n_protein_whole + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex_whole_protein"].is_global = is_global.bool()

    data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:, :2].long().t().contiguous() + 1
    data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

    data['compound_atom_edge_list'].x = (input_atom_edge_list[:, :2].long().contiguous() + 1).clone()
    data['LAS_edge_list'].x = data['complex', 'LAS', 'complex'].edge_index.clone().t()

    if flag == 1:   # rdkit_ligand, holo_protein
        input_pocket_node_xyz = protein_node_xyz[keepNode]
        data.input_pocket_node_xyz = input_pocket_node_xyz

        # complex coords
        coords_init = rdkit_coords - rdkit_coords.mean(dim=0).reshape(1, 3) + input_pocket_node_xyz.mean(dim=0).reshape(1, 3)
        data['complex'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init,
                torch.zeros(1, 3),
                input_pocket_node_xyz
            ), dim=0
        ).float()

        # complex_whole coords
        data['complex_whole_protein'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0).reshape(1, 3),
                # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3),
                protein_node_xyz
            ), dim=0
        ).float()
        data['compound'].node_coords = coords_init
        data.input_protein_node_xyz = protein_node_xyz
    elif flag == 2:     # holo ligand, af2 protein
        input_pocket_node_xyz = af2_protein_node_xyz[keepNode]
        data.input_pocket_node_xyz = input_pocket_node_xyz

        # complex information
        coords_init = data.coords - data.coords.mean(dim=0).reshape(1, 3) + input_pocket_node_xyz.mean(dim=0).reshape(1, 3)
        data['complex'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init,
                torch.zeros(1, 3),
                input_pocket_node_xyz
            ), dim=0
        ).float()

        data['complex_whole_protein'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0).reshape(1, 3),
                # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3),
                af2_protein_node_xyz
            ), dim=0
        ).float()

        data['compound'].node_coords = coords_init
        data.input_protein_node_xyz = af2_protein_node_xyz
    else:       # rdkit ligand, af2 protein
        input_pocket_node_xyz = af2_protein_node_xyz[keepNode]
        data.input_pocket_node_xyz = input_pocket_node_xyz.clone()

        # complex information
        coords_init = rdkit_coords - rdkit_coords.mean(dim=0).reshape(1, 3) + input_pocket_node_xyz.mean(dim=0).reshape(1, 3)
        data['complex'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init,
                torch.zeros(1, 3),
                input_pocket_node_xyz
            ), dim=0
        ).float()

        data['complex_whole_protein'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0).reshape(1, 3),
                # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3),
                af2_protein_node_xyz
            ), dim=0
        ).float()

        data['compound'].node_coords = coords_init
        data.input_protein_node_xyz = af2_protein_node_xyz
    if args.shift_coord:
        data.pocket_residue_center = data.input_protein_node_xyz.mean(dim=0).unsqueeze(0)
        data.input_pocket_node_xyz = data.input_pocket_node_xyz - data.pocket_residue_center
    return data

def construct_data_new(args, flag, group, protein_node_xyz, protein_seq, protein_esm2_feat,
                   coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, LAS_edge_index, rdkit_coords,
                   add_noise_to_com=None, includeDisMap=True, pocket_idx_no_noise=True, isomorphisms=None):
    n_protein = protein_node_xyz.shape[0]
    coords_bias = protein_node_xyz.mean(dim=0)
    coords = coords - coords_bias.numpy()
    protein_node_xyz = protein_node_xyz - coords_bias
    # centroid instead of com.
    com = coords.mean(axis=0)

    # construct heterogeneous graph data.
    data = HeteroData()
    data.isomorphisms = isomorphisms
    data.seq_whole = protein_seq
    data.coords = torch.tensor(coords, dtype=torch.float)
    data.num_atoms = data.coords.shape[0]
    data.coord_offset = coords_bias.unsqueeze(0)
    data.protein_node_xyz = protein_node_xyz

    # max sphere radius
    coords_tensor = torch.tensor(coords)
    center = coords_tensor.mean(dim=0)
    distances = torch.norm(coords_tensor - center, dim=1)
    data.ligand_radius = distances.max()
    data.coords_center = torch.tensor(com, dtype=torch.float).unsqueeze(0)

    # pocket radius
    if args.pocket_radius_buffer <= 2.0:
        pocket_radius = args.pocket_radius_buffer * data.ligand_radius
    else:
        pocket_radius = args.pocket_radius_buffer + data.ligand_radius
    if pocket_radius < args.min_pocket_radius:
        pocket_radius = args.min_pocket_radius

    if args.force_fix_radius:
        pocket_radius = args.pocket_radius

    # keepNode
    keepNode = get_keepNode(com, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=add_noise_to_com)
    keepNode_no_noise = get_keepNode(com, n_protein, protein_node_xyz.numpy(), pocket_radius, add_noise_to_com=None)
    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True
    data.pocket_idx = torch.tensor(keepNode_no_noise, dtype=torch.int)
    data.pocket_node_xyz = protein_node_xyz[keepNode]
    # protein feature map
    if torch.is_tensor(protein_esm2_feat):
        data['protein_whole'].node_feats = protein_esm2_feat
    else:
        raise ValueError("protein_esm2_feat should be a tensor")

    # distance map
    dis_map = scipy.spatial.distance.cdist(protein_node_xyz[keepNode].cpu().numpy(), coords)
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map > args.dis_map_thres] = args.dis_map_thres
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()

    # pocket feature map
    if torch.is_tensor(protein_esm2_feat):
        data['pocket'].node_feats = protein_esm2_feat[keepNode]
    else:
        raise ValueError("protein_esm2_feat should be a tensor")
    data['pocket'].keepNode = torch.tensor(keepNode, dtype=torch.bool)
    data['compound'].node_feats = compound_node_features.float()
    data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index

    n_pocket = protein_node_xyz[keepNode].shape[0]
    n_protein_whole = protein_node_xyz.shape[0]
    n_compound = compound_node_features.shape[0]
    rdkit_coords = torch.tensor(rdkit_coords)
    data['compound'].rdkit_coords = rdkit_coords

    data['complex'].node_coords_LAS = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3),
            torch.zeros_like(protein_node_xyz[keepNode])
        ), dim=0
    ).float()
    segment = torch.zeros(n_compound + n_pocket + 2)
    segment[n_compound + 1:] = 1  # compound: 0, protein: 1
    data['complex'].segment = segment
    lig_mask = torch.zeros(n_compound + n_pocket + 2)
    lig_mask[:n_compound + 2] = 1  # glb_p can be updated
    data['complex'].lig_mask = lig_mask.bool()
    pro_mask = torch.zeros(n_compound + n_pocket + 2)
    pro_mask[0] = 1
    pro_mask[n_compound + 1:] = 1  # glb_c, glb_p can be updated
    data['complex'].pro_mask = pro_mask.bool()
    is_global = torch.zeros(n_compound + n_pocket + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex"].is_global = is_global.bool()

    data['complex', 'c2c', 'complex'].edge_index = input_atom_edge_list[:, :2].long().t().contiguous() + 1
    data['complex', 'LAS', 'complex'].edge_index = LAS_edge_index + 1

    data['complex_whole_protein'].node_coords_LAS = torch.cat(  # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3),
            torch.zeros_like(protein_node_xyz)
        ), dim=0
    ).float()

    segment = torch.zeros(n_compound + n_protein_whole + 2)
    segment[n_compound + 1:] = 1  # compound: 0, protein: 1
    data['complex_whole_protein'].segment = segment
    lig_mask = torch.zeros(n_compound + n_protein_whole + 2)
    lig_mask[:n_compound + 2] = 1  # glb_p can be updated?
    data['complex_whole_protein'].lig_mask = lig_mask.bool()
    pro_mask = torch.zeros(n_compound + n_protein_whole + 2)
    pro_mask[0] = 1
    pro_mask[n_compound + 1:] = 1
    data['complex_whole_protein'].pro_mask = pro_mask.bool()
    is_global = torch.zeros(n_compound + n_protein_whole + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex_whole_protein"].is_global = is_global.bool()

    data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:, :2].long().t().contiguous() + 1
    data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

    data['compound_atom_edge_list'].x = (input_atom_edge_list[:, :2].long().contiguous() + 1).clone()
    data['LAS_edge_list'].x = data['complex', 'LAS', 'complex'].edge_index.clone().t()

    if flag == 1 or flag == 3:   # rdkit_ligand, input_protein
        input_pocket_node_xyz = protein_node_xyz[keepNode]
        data.input_pocket_node_xyz = input_pocket_node_xyz

        # complex coords
        coords_init = rdkit_coords - rdkit_coords.mean(dim=0).reshape(1, 3) + input_pocket_node_xyz.mean(dim=0).reshape(1, 3)
        data['complex'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init,
                torch.zeros(1, 3),
                input_pocket_node_xyz
            ), dim=0
        ).float()

        # complex_whole coords
        data['complex_whole_protein'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0).reshape(1, 3),
                # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3),
                protein_node_xyz
            ), dim=0
        ).float()
        data['compound'].node_coords = coords_init

        data.input_protein_node_xyz = protein_node_xyz
        data.pocket_residue_center = input_pocket_node_xyz.mean(dim=0).unsqueeze(0)
        data.input_pocket_node_xyz = data.input_pocket_node_xyz - data.pocket_residue_center    # pocket自己为中心
        return data
    elif flag == 2:     # holo ligand, af2 protein
        input_pocket_node_xyz = protein_node_xyz[keepNode]
        data.input_pocket_node_xyz = input_pocket_node_xyz

        # complex information
        coords_init = data.coords - data.coords.mean(dim=0).reshape(1, 3) + input_pocket_node_xyz.mean(dim=0).reshape(1, 3)
        data['complex'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init,
                torch.zeros(1, 3),
                input_pocket_node_xyz
            ), dim=0
        ).float()

        data['complex_whole_protein'].node_coords = torch.cat(  # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0).reshape(1, 3),
                # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3),
                protein_node_xyz
            ), dim=0
        ).float()

        data['compound'].node_coords = coords_init
        data.input_protein_node_xyz = protein_node_xyz
        data.pocket_residue_center = input_pocket_node_xyz.mean(dim=0).unsqueeze(0)
        data.input_pocket_node_xyz = data.input_pocket_node_xyz - data.pocket_residue_center  # pocket自己为中心
        return data



def gumbel_softmax_no_random(logits: torch.Tensor, tau: float = 1, hard: bool = False,
                             eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    gumbels = logits / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret









