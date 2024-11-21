import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_batch
import random

from models.egnn import MCAttEGNN
from models.model_utils import InteractionModule


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res

class ComplexGraph(nn.Module):
    def __init__(self, args, inter_cutoff=10, intra_cutoff=8, normalize_coord=None, unnormalize_coord=None):
        super().__init__()
        self.args = args
        self.inter_cutoff = normalize_coord(inter_cutoff)
        self.intra_cutoff = normalize_coord(intra_cutoff)

    @torch.no_grad()
    def construct_edges(self, X, batch_id, segment_ids, is_global):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs], device="cpu", 记录batch中每个样本的蛋白质残基数目
        N, max_n = batch_id.shape[0], lengths.max().item()
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]

        gni = torch.arange(N, device=batch_id.device)   # 0 -> N
        gni2lni = gni - offsets[batch_id]  # [N]，按不同batch数

        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[gni, lengths[batch_id] - 1] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index

        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        # segment for compound is 0, for protein is 1
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = sequential_and(
            row_seg == col_seg,
            row_seg == 1,
            not_global_edges
        )
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]

        # ctx edges
        ctx_edges = _radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=self.intra_cutoff)

        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges = _radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=self.inter_cutoff)
        if inter_edges.shape[1] == 0:
            inter_edges = torch.tensor([[inter_all_row[0], inter_all_col[0]], [inter_all_col[0], inter_all_row[0]]], device=inter_all_row.device)
        reduced_inter_edge_batchid = batch_id[inter_edges[0][inter_edges[0] < inter_edges[1]]]  # [0, 0, 0, 1, 1]  make sure row belongs to compound and col belongs to protein
        reduced_inter_edge_offsets = offsets.gather(-1, reduced_inter_edge_batchid)     # [0, 0, 0, 306, 306]

        # edges between global and normal nodes
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(row_global, col_global)  # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        ctx_edges = torch.cat([ctx_edges, global_normal, global_global], dim=1)  # [2, E]

        return ctx_edges, inter_edges, (reduced_inter_edge_batchid, reduced_inter_edge_offsets)

    def forward(self, X, batch_id, segment_ids, is_global):
        return self.construct_edges(X, batch_id, segment_ids, is_global)


def _radial_edges(X, src_dst, cutoff):
    dist = X[:, 0][src_dst]  # [Ef, 2, 3], CA position
    dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1) # [Ef]
    src_dst = src_dst[dist <= cutoff]
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    return src_dst


class EfficientMCAttModel(nn.Module):
    def __init__(self, args, embed_size, hidden_size, n_channel, n_edge_feats=0,
                 n_layers=5, dropout=0.1, n_iter=5, dense=False,
                 inter_cutoff=10, intra_cutoff=8, normalize_coord=None, unnormalize_coord=None):
        super().__init__()
        self.n_iter = n_iter
        self.args = args
        self.random_n_iter = args.random_n_iter
        self.gnn = MCAttEGNN(args, embed_size, hidden_size, hidden_size,
                             n_channel, n_edge_feats, n_layers=n_layers,
                             residual=True, dropout=dropout, dense=dense,
                             normalize_coord=normalize_coord, unnormalize_coord=unnormalize_coord,
                             geometry_reg_step_size=args.geometry_reg_step_size)

        self.extract_edges = ComplexGraph(args, inter_cutoff=inter_cutoff, intra_cutoff=intra_cutoff,
                                          normalize_coord=normalize_coord, unnormalize_coord=unnormalize_coord)
        self.inter_layer = InteractionModule(hidden_size, hidden_size, hidden_size, rm_layernorm=args.rm_layernorm)

    def forward(self, X, H, batch_id, segment_id, lig_mask, pro_mask, is_global,
                compound_edge_index, LAS_edge_index, batched_complex_coord_LAS, LAS_mask=None, flag=None):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param batch: [n_all_node]
        '''
        p_p_dist_embed = None
        c_c_dist_embed = None
        c_batch = batch_id[segment_id == 0]
        p_batch = batch_id[segment_id == 1]
        c_embed = H[segment_id == 0]
        p_embed = H[segment_id == 1]
        p_embed_batched, p_mask = to_dense_batch(p_embed, p_batch)
        c_embed_batched, c_mask = to_dense_batch(c_embed, c_batch)
        pair_embed_batched, pair_mask = self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)
        pair_embed_batched = pair_embed_batched * pair_mask.to(torch.float).unsqueeze(-1)

        if self.training and self.random_n_iter:
            iter_i = random.randint(1, self.n_iter)
        else:
            iter_i = self.n_iter

        for r in range(iter_i):
            # refine
            if r < iter_i - 1:
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(X, batch_id, segment_id, is_global)
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                    _, Z, _ = self.gnn(H, X, ctx_edges, inter_edges, LAS_edge_index, batched_complex_coord_LAS,
                                       segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple,
                                       pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, LAS_mask=LAS_mask,
                                       p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed, mask=lig_mask)
                    if flag == 1:
                        X[lig_mask] = Z[lig_mask]
                    elif flag == 2:
                        X[pro_mask] = Z[pro_mask]
                    else:
                        X = Z
            else:
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(X, batch_id, segment_id, is_global)
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                H, Z, pair_embed_batched = self.gnn(H, X, ctx_edges, inter_edges, LAS_edge_index,
                                                    batched_complex_coord_LAS, segment_id=segment_id, batch_id=batch_id,
                                                    reduced_tuple=reduced_tuple, pair_embed_batched=pair_embed_batched,
                                                    pair_mask=pair_mask, LAS_mask=LAS_mask, p_p_dist_embed=p_p_dist_embed,
                                                    c_c_dist_embed=c_c_dist_embed, mask=lig_mask)
                if flag == 1:
                    X[lig_mask] = Z[lig_mask]
                elif flag == 2:
                    X[pro_mask] = Z[pro_mask]
                else:
                    X = Z
        return X, H, pair_embed_batched


# class BindLayer(nn.Module):
#     def __init__(self, args, embed_size, hidden_size, n_channel, n_edge_feats=0,
#                  n_layers=5, dropout=0.1, n_iter=5, inter_cutoff=10, intra_cutoff=8, normalize_coord=None, unnormalize_coord=None):
#         super().__init__()
#         self.args = args
#         self.n_iter = n_iter
#         self.random_n_iter = args.random_n_iter
#
#         self.extract_edges = ComplexGraph(args, inter_cutoff=inter_cutoff, intra_cutoff=intra_cutoff, normalize_coord=normalize_coord, unnormalize_coord=unnormalize_coord)
#
#         self.inter_layer = InteractionModule(hidden_size, hidden_size, hidden_size)
#
#         self.gnn = MCAttEGNN(args, embed_size, hidden_size, hidden_size,
#                              n_channel, in_edge_nf=n_edge_feats, n_layers=n_layers, residual=True, dropout=dropout,
#                              normalize_coord=normalize_coord, unnormalize_coord=unnormalize_coord,
#                              geometry_reg_step_size=args.geometry_reg_step_size)
#
#     def forward(self, X, H, batch_id, segment_id, lig_mask, pro_mask, is_global,
#                 compound_edge_index, LAS_ege_index, batched_complex_coord_LAS, LAS_mask=None, flag=None):
#         '''
#             X: coords, [n_allnodes, 3]
#             H: features, [n_allnodes, 128]
#         '''
#         p_p_dist_embed = None
#         c_c_dist_embed = None
#
#         c_batch = batch_id[segment_id == 0]
#         p_batch = batch_id[segment_id == 1]
#         c_embed = H[segment_id == 0]
#         p_embed = H[segment_id == 1]
#         c_embed_batched, c_mask = to_dense_batch(c_embed, c_batch)  # bs, Nc_max, d
#         p_embed_batched, p_mask = to_dense_batch(p_embed, p_batch)  # bs, Np_max, d
#         pair_embed_batched, pair_mask = self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)
#
#         if self.training and self.random_n_iter:
#             total_iter = random.randint(1, self.n_iter)
#         else:
#             total_iter = self.n_iter
#
#         for r in range(total_iter):
#             # 使用refine coords, 只有最后一次recycle计算梯度
#             if r < total_iter - 1:
#                 with torch.no_grad():
#                     ctx_edges, inter_edges, reduced_tuple = self.extract_edges(X, batch_id, segment_id, is_global)
#                     ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
#                     # H: (836, 128)     X: (836, 1, 3)
#                     _, Z, _ = self.gnn(H, X, ctx_edges, inter_edges, LAS_ege_index, batched_complex_coord_LAS,
#                                     segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple,
#                                     pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, LAS_mask=LAS_mask,
#                                     p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed, mask=lig_mask)
#                     # X[mask] = Z[mask]
#                     if flag == 1:
#                         X[lig_mask] = Z[lig_mask]
#                     elif flag == 2:
#                         X[pro_mask] = Z[pro_mask]
#                     else:
#                         X = Z
#             else:
#                 with torch.no_grad():
#                     ctx_edges, inter_edges, reduced_tuple = self.extract_edges(X, batch_id, segment_id, is_global)
#                     ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
#                 H, Z, pair_embed_batched = self.gnn(H, X, ctx_edges, inter_edges, LAS_ege_index, batched_complex_coord_LAS,
#                                 segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple,
#                                 pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, LAS_mask=LAS_mask,
#                                 p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed, mask=lig_mask)
#                 # X[mask] = Z[mask]
#                 if flag == 1:
#                     X[lig_mask] = Z[lig_mask]
#                 elif flag == 2:
#                     X[pro_mask] = Z[pro_mask]
#                 else:
#                     X = Z
#
#         return X, H, pair_embed_batched




