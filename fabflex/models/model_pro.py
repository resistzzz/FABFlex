
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
import random

from models.attn_model import EfficientMCAttModel
from models.model_utils import MLP
from utils.utils import gumbel_softmax_no_random, get_keepNode_tensor


class FABindProteinComplex(nn.Module):
    def __init__(self, args, embedding_channels=128, pocket_pred_embedding_channels=128):
        super().__init__()
        self.args = args
        self.coordinate_scale = args.coord_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale
        self.unnormalize_coord = lambda x: x * self.coordinate_scale

        # global nodes for protein / compound
        self.glb_c = nn.Parameter(torch.ones(1, embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, embedding_channels))
        protein_hidden = 1280  # hard-coded for ESM2 feature
        compound_hidden = 56  # hard-coded for hand-crafted feature

        self.protein_linear_whole_protein = nn.Linear(protein_hidden, embedding_channels)
        self.compound_linear_whole_protein = nn.Linear(compound_hidden, embedding_channels)

        n_channel = 1  # ligand node has only one coordinate dimension.

        self.complex_model = EfficientMCAttModel(
            args, embedding_channels, embedding_channels, n_channel, n_edge_feats=0, n_layers=args.mean_layers,
            n_iter=args.n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord,
            unnormalize_coord=self.unnormalize_coord,
        )
        self.distmap_mlp = MLP(args, embedding_channels=embedding_channels, n=self.args.mlp_hidden_scale, out_channels=1)

        torch.nn.init.xavier_uniform_(self.protein_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_linear_whole_protein.weight, gain=0.001)

    def forward(self, data, stage=1, train=False, flag=1):
        keepNode_less_5 = 0
        compound_batch = data['compound'].batch
        pocket_batch = data['pocket'].batch
        complex_batch = data['complex'].batch
        protein_batch_whole = data['protein_whole'].batch
        complex_batch_whole_protein = data['complex_whole_protein'].batch

        batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))
        batched_pocket_emb = self.protein_linear_whole_protein(data['protein_whole'].node_feats[data['pocket'].keepNode])
        batched_compound_emb = self.compound_linear_whole_protein(data['compound'].node_feats)
        for i in range(complex_batch.max() + 1):
            if i == 0:
                new_samples = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch == i],
                    self.glb_p, batched_pocket_emb[pocket_batch == i]
                ), dim=0)
            else:
                new_sample = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch == i],
                    self.glb_p, batched_pocket_emb[pocket_batch == i]
                ), dim=0)
                new_samples = torch.cat((new_samples, new_sample), dim=0)
        dis_map = data.dis_map

        complex_coords, complex_out, pair_embed_batched = self.complex_model(
            batched_complex_coord,
            new_samples,
            batch_id=complex_batch,
            segment_id=data['complex'].segment,
            lig_mask=data['complex'].lig_mask,
            pro_mask=data['complex'].pro_mask,
            is_global=data['complex'].is_global,
            compound_edge_index=data['complex', 'c2c', 'complex'].edge_index,
            LAS_edge_index=data['complex', 'LAS', 'complex'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None,
            flag=flag
        )
        compound_flag = torch.logical_and(data['complex'].segment == 0, ~data['complex'].is_global)
        protein_flag = torch.logical_and(data['complex'].segment == 1, ~data['complex'].is_global)
        pocket_out = complex_out[protein_flag]
        pocket_coords_out = complex_coords[protein_flag].squeeze(-2)
        compound_out = complex_out[compound_flag]
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)

        _, pocket_out_mask = to_dense_batch(pocket_out, pocket_batch)
        _, compound_out_mask = to_dense_batch(compound_out, compound_batch)
        compound_coords_out_batched, _ = to_dense_batch(compound_coords_out, compound_batch)
        pocket_coords_out_batched, _ = to_dense_batch(pocket_coords_out, pocket_batch)

        pocket_com_dis_map = torch.cdist(pocket_coords_out_batched, compound_coords_out_batched)

        z = pair_embed_batched[:, 1:, 1:, ...]
        z_mask = torch.einsum("bi,bj->bij", pocket_out_mask, compound_out_mask)

        b = self.distmap_mlp(z).squeeze(-1)
        y_pred = b[z_mask]
        y_pred = y_pred.sigmoid() * self.args.dis_map_thres  # normalize to 0 to 10.

        y_pred_by_coords = pocket_com_dis_map[z_mask]
        y_pred_by_coords = self.unnormalize_coord(y_pred_by_coords)
        y_pred_by_coords = torch.clamp(y_pred_by_coords, 0, self.args.dis_map_thres)

        compound_coords_out = self.unnormalize_coord(compound_coords_out)
        pocket_coords_out = self.unnormalize_coord(pocket_coords_out)

        return (compound_coords_out, compound_batch, pocket_coords_out, pocket_batch,
                y_pred, y_pred_by_coords, dis_map, keepNode_less_5)


def get_model(args, logger):
    logger.log_message("FABind plus")
    model = FABindProteinComplex(args, args.hidden_size, args.pocket_pred_hidden_size)
    return model





