import math

import torch
import torch.nn as nn
from einops.einops import repeat
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from .cascade_functions import *

INF = 1e9


class CascadeFinePreprocess(nn.Module):
    def __init__(self, config, config_fine, config_coarse, coarse_level):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']
        self.coarse_level = coarse_level

        d_model_c = config_coarse['d_model']
        d_model_f = config_fine['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data[f'hw0_{self.coarse_level}'][0]

        data.update({'W': W})
        if data[f'stage_{self.coarse_level}']['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data[f'stage_{self.coarse_level}']['b_ids'], data[f'stage_{self.coarse_level}']['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data[f'stage_{self.coarse_level}']['b_ids'], data[f'stage_{self.coarse_level}']['j_ids']]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[data[f'stage_{self.coarse_level}']['b_ids'], data[f'stage_{self.coarse_level}']['i_ids']],
                                                   feat_c1[data[f'stage_{self.coarse_level}']['b_ids'], data[f'stage_{self.coarse_level}']['j_ids']]], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W ** 2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold


class CascadeFineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, coarse_level='4c'):
        super().__init__()
        self.coarse_level = coarse_level

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['stage_4c']['mkpts0_c'],
                'mkpts1_f': data['stage_4c']['mkpts1_c'],
            })
            return

        # feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
        feat_f0_picked = feat_f0[:, WW // 2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C ** .5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized ** 2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data[f'stage_{self.coarse_level}']['mkpts0_c']
        scale1 = scale * data['scale1'][data[f'stage_{self.coarse_level}']['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data[f'stage_{self.coarse_level}']['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data[f'stage_{self.coarse_level}']['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W ** 2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        # feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
        feat_f0_picked = feat_f0[:, WW // 2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C ** .5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized ** 2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
