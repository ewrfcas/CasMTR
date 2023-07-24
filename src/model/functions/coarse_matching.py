import torch
import torch.nn as nn

from .cascade_functions import *

INF = 1e9


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks [B, H, W]
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]  # [B,]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])  # [B, 2]两张图求match，求交集所以取min
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config, coarse_config=None):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        if coarse_config is not None:
            self.next_topk = coarse_config.get('next_topk', None)
        else:
            self.next_topk = None

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        self.temperature = config['dsmax_temperature']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None, level='8c'):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_c0, feat_c1])
        assert self.match_type == 'dual_softmax'

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
        sim_matrix_softmax10 = F.softmax(sim_matrix, 1)
        sim_matrix_softmax01 = F.softmax(sim_matrix, 2)
        conf_matrix = sim_matrix_softmax10 * sim_matrix_softmax01

        next_conf_c01, next_idx_c01 = torch.max(sim_matrix_softmax01, dim=2)
        next_conf_c10, next_idx_c10 = torch.max(sim_matrix_softmax10, dim=1)
        next_conf_c01_s, next_idx_c01_s = None, None

        next_conf_c01_topk, next_idx_c01_topk = None, None
        next_conf_c10_topk, next_idx_c10_topk = None, None

        data[f'stage_{level}'] = {'conf_matrix': conf_matrix,
                                  'next_conf_c01_topk': next_conf_c01_topk, 'next_idx_c01_topk': next_idx_c01_topk,
                                  'next_conf_c10_topk': next_conf_c10_topk, 'next_idx_c10_topk': next_idx_c10_topk,
                                  'next_idx_c01': next_idx_c01, 'next_idx_c10': next_idx_c10,
                                  'next_conf_c01': next_conf_c01, 'next_conf_c10': next_conf_c10,
                                  'next_conf_c01_s': next_conf_c01_s, 'next_idx_c01_s': next_idx_c01_s}

        # predict coarse matches from conf_matrix
        match_result = self.get_coarse_match(conf_matrix, data, level)

        data[f'stage_{level}'].update(**match_result)

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data, level):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data[f'hw0_{level}'][0],
            'w0c': data[f'hw0_{level}'][1],
            'h1c': data[f'hw1_{level}'][0],
            'w1c': data[f'hw1_{level}'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        if f'mask_{level}0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data[f'mask_{level}0'], data[f'mask_{level}1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)

        # 2. mutual nearest 保证行列最大值是同一个idx [B, HW, HW]
        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)  # [B, HW]
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]  # j_ids: image0 target at which idx of image1
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data[f'hw0_{level}'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack([i_ids % data[f'hw0_{level}'][1], torch.div(i_ids, data[f'hw0_{level}'][1], rounding_mode='trunc')], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data[f'hw1_{level}'][1], torch.div(j_ids, data[f'hw1_{level}'][1], rounding_mode='trunc')], dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches
