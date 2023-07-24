import torch
import torch.nn as nn
from einops.einops import repeat

from .cascade_functions import *
from .post_processing import PostProcess

INF = 1e9


def torch_gather(params, indices):
    if params.shape[0] != indices.shape[0]:  # [B,]
        raise ValueError(
            f"Make sure that the first two dimensions of params and indices are identical, \
                but they are params: {params.shape[0]} vs. indices: {params.shape[0]}"
        )
    num_indices_to_gather = indices.shape[-2] * indices.shape[-1]  # L*K
    num_indices_to_pick_from = params.shape[1]  # L

    indices_shift = (
            torch.div(torch.arange(indices.shape[0] * num_indices_to_gather, device=indices.device),  # B*L*K
                      num_indices_to_gather, rounding_mode='trunc')  # L*K
            * num_indices_to_pick_from  # L
    )  # [B*L*K,] 即每L*K个+L，0...,0(L*K), L...,L, 2L...,2L,...

    flattened_indices = indices.reshape(-1) + indices_shift
    flattened_params = params.reshape(-1, params.shape[-1])  # [B,L,C] ---> [B*L,C]

    out_flattened = flattened_params.index_select(0, flattened_indices)  # [B*L,C] gather [B*L*K,]->[B*L*K,C]

    out = out_flattened.reshape(params.shape[0], indices.shape[1], indices.shape[-1], params.shape[2])  # [B*L*K,C]->[B,L,K,C]
    return out


class CascadeMatching(nn.Module):
    def __init__(self, config, cas_config, stage=None):
        super().__init__()
        self.config = config
        self.cas_config = cas_config
        # general config
        self.thr = config['thr']
        self.test_thr = config['test_thr']
        self.pre_thr = config['pre_thr']
        self.border_rm = config['border_rm']
        self.double_check = config['double_check']
        # -- # for trainig fine-level LoFTR
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        self.propagation = cas_config['propagation']
        self.dilated = cas_config['dilated']
        self.post_process = PostProcess(post_config=cas_config['post_config'])
        self.detector_mode = cas_config.get('detector_mode', None)
        self.grid_size = cas_config.get('grid_size', None)
        self.rt = cas_config['post_config'].get('rt', None)
        self.rd = cas_config['post_config'].get('rd', None)
        self.stage = stage
        self.next_topk = cas_config.get('next_topk', None)

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        assert self.match_type == 'softmax'
        self.temperature = config['dsmax_temperature']

    def forward(self, feat_c0, feat_c1, idx_c01, idx_c10, data, mask_c0=None, mask_c1=None,
                heatmap_c0=None, level='4c', pre_level='8c'):
        """
        Args:
            feat0 (torch.Tensor): [B, HW0, C]
            feat1 (torch.Tensor): [B, HW1, C]
            idx_c01 (torch.Tensor): [B,HW0,4ww] 1d coord
            idx_c10 (torch.Tensor): [B,HW1,4ww] 1d coord
            data (dict)
            mask_c0 (torch.Tensor): [B, HW0] (optional)
            mask_c1 (torch.Tensor): [B, HW1] (optional)
            heatmap_c0: [B,1,H0,W0]
            level: feature map scale
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
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_c0, feat_c1])

        if self.cas_config['post_config']['method'] == 'd2d':  # d2d detection
            S_as = torch.std(feat_c0, dim=-1, keepdim=True)
            S_as = rearrange(S_as, 'b (h w) c -> b c h w', h=data[f'hw0_{level}'][0])
            S_as = F.interpolate(S_as, scale_factor=0.25, mode='nearest')
            S_as = rearrange(S_as, 'b c h w -> b (h w) c')
            C = feat_c0.shape[-1]
            feat_c0_2d = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data[f'hw0_{level}'][0])
            kernel = -torch.ones((C, 1, 5, 5), dtype=feat_c0.dtype, device=feat_c0.device) / 25.0
            kernel[:, :, 2, 2] = 24.0
            S_rs = torch.norm(F.conv2d(feat_c0_2d, weight=kernel, stride=4, groups=C, padding=2), dim=1, keepdim=True)
            S_rs = (S_rs - S_rs.min()) / (S_rs.max() - S_rs.min())
            S_rs = rearrange(S_rs, 'b c h w -> b (h w) c')
            S_d2d = S_as * S_rs
            data['S_d2d'] = S_d2d
            data['d2d_w'] = data[f'hw0_{level}'][1] // 4

        # convert mask to window based mask
        # mask:[B,HW1]->[B,repeat(HW0),HW1], index:[B,HW0,4ww]
        if mask_c0 is not None and mask_c1 is not None:
            window_mask_c0 = torch.gather(repeat(mask_c1, 'b l1 -> b l0 l1', l0=idx_c01.shape[1]), dim=2, index=idx_c01)  # [B,HW0,4ww]
            window_mask_c0 = window_mask_c0 * mask_c0.unsqueeze(-1)
            window_mask_c1 = torch.gather(repeat(mask_c0, 'b l1 -> b l0 l1', l0=idx_c10.shape[1]), dim=2, index=idx_c10)  # [B,HW1,4ww]
            window_mask_c1 = window_mask_c1 * mask_c1.unsqueeze(-1)
        else:
            window_mask_c0, window_mask_c1 = None, None

        assert self.match_type == 'softmax'

        # CUDA similarity feat_c0:[B,HW0,C], feat_c1:[B,HW1,C], idx_c01:[B,HW0,4ww], return:sim_matrix01:[B,HW0,4ww]
        sim_matrix01 = ScoreComputation.apply(feat_c0, feat_c1, idx_c01)
        sim_matrix01 /= self.temperature
        # # PYTHON similarity
        # feat_c1_topk = torch_gather(feat_c1, idx_c01) # [B,HW0,4ww,C]
        # sim_matrix01 = (feat_c0.unsqueeze(2) * feat_c1_topk).sum(-1) / self.temperature
        if window_mask_c0 is not None:
            sim_matrix01.masked_fill_(~(window_mask_c0.bool()), -INF)
        conf_matrix01 = torch.softmax(sim_matrix01, dim=2)  # [B,HW0,4ww]

        next_conf_c01, next_idx_c01 = torch.max(conf_matrix01, dim=2)
        next_idx_c01 = torch.gather(idx_c01, dim=2, index=next_idx_c01.unsqueeze(dim=-1)).squeeze(dim=-1)  # [B,HW0]
        next_conf_c01_s, next_idx_c01_s = None, None

        if self.training and self.detector_mode is not None:  # if training use detector for conf
            if heatmap_c0 is None:  # use max conf logits instead
                heatmap_c0, _ = torch.max(sim_matrix01, dim=2)
                heatmap_c0 = heatmap_c0.reshape(heatmap_c0.shape[0], 1, data[f'hw0_{level}'][0], data[f'hw0_{level}'][1])
            detector_matrix01 = detect_keypoints(heatmap_c0, conf_matrix01, mode=self.detector_mode, grid_size=self.grid_size)
        else:
            detector_matrix01 = None

        sim_matrix10 = ScoreComputation.apply(feat_c1, feat_c0, idx_c10).detach()  # 1->0 are not optimized
        sim_matrix10 /= self.temperature
        # # PYTHON similarity
        # feat_c0_topk = torch_gather(feat_c0, idx_c10) # [B,HW0,4ww,C]
        # sim_matrix10 = (feat_c1.unsqueeze(2) * feat_c0_topk).sum(-1) / self.temperature
        if window_mask_c1 is not None:
            sim_matrix10.masked_fill_(~(window_mask_c1.bool()), -INF)
        conf_matrix10 = torch.softmax(sim_matrix10, dim=2)  # [B,HW1,4ww]
        next_conf_c10, next_idx_c10 = torch.max(conf_matrix10, dim=2)
        next_idx_c10 = torch.gather(idx_c10, dim=2, index=next_idx_c10.unsqueeze(dim=-1)).squeeze(dim=-1)  # [B,HW1]

        # if self.next_topk is None:
        next_conf_c01_topk, next_idx_c01_topk = None, None
        next_conf_c10_topk, next_idx_c10_topk = None, None

        data[f'stage_{level}'] = {'conf_matrix': conf_matrix01, 'detector_matrix01': detector_matrix01,
                                  'next_conf_c01_topk': next_conf_c01_topk, 'next_idx_c01_topk': next_idx_c01_topk,
                                  'next_conf_c10_topk': next_conf_c10_topk, 'next_idx_c10_topk': next_idx_c10_topk,
                                  'idx_c01': idx_c01, 'idx_c10': idx_c10,
                                  'next_idx_c01': next_idx_c01, 'next_idx_c10': next_idx_c10,
                                  'next_conf_c01': next_conf_c01, 'next_conf_c10': next_conf_c10,
                                  'next_conf_c01_s': next_conf_c01_s, 'next_idx_c01_s': next_idx_c01_s}

        # predict coarse matches from conf_matrix
        match_result = self.get_coarse_match(conf_matrix01, idx_c01, next_conf_c01, next_idx_c01, next_idx_c10, data, level, pre_level)

        data[f'stage_{level}'].update(**match_result)
        if 'm_bids' in match_result:
            data['m_bids'] = match_result['m_bids']

    def get_coarse_match(self, conf_matrix01, idx_c01, next_conf_c01, next_idx_c01, next_idx_c10, data, level, pre_level):
        """
        Args:
            next_conf_c01 (torch.Tensor): [B,HW0]
            next_conf_c10 (torch.Tensor): [B,HW1]
            next_idx_c01 (torch.Tensor): [B,HW0]
            next_idx_c10 (torch.Tensor): [B,HW1]
        """
        _device = next_conf_c01.device
        axes_lengths = {
            'h0c': data[f'hw0_{level}'][0],
            'w0c': data[f'hw0_{level}'][1],
            'h1c': data[f'hw1_{level}'][0],
            'w1c': data[f'hw1_{level}'][1],
        }
        # 1. confidence thresholding
        if self.training:
            if self.thr > 0:
                mask = next_conf_c01 > (1 / idx_c01.shape[-1])
            else:
                mask = next_conf_c01 > self.thr
        else:  # test specific post-processing
            mask = self.post_process.apply(data, axes_lengths, next_idx_c01, next_conf_c01, self.test_thr, level)

            if self.rt is not None:
                ts = data[f'stage_{level}']['next_conf_c01_s'] / (next_conf_c01 + 1e-7)
                mask[ts > self.rt] = False

            # previous stage conf
            if type(pre_level) != list:
                pre_level = [pre_level]
            for i, pre_level_ in enumerate(pre_level):
                pre_conf_c01 = data[f'stage_{pre_level_}']['next_conf_c01'].detach()
                pre_conf_c01 = rearrange(pre_conf_c01, 'b (h0p w0p) -> b 1 h0p w0p', h0p=data[f'hw0_{pre_level_}'][0], w0p=data[f'hw0_{pre_level_}'][1])
                pre_conf_c01 = F.interpolate(pre_conf_c01, (data[f'hw0_{level}'][0], data[f'hw0_{level}'][1]), mode='nearest')
                pre_conf_c01 = rearrange(pre_conf_c01, 'b 1 h0c w0c -> b (h0c w0c)', h0c=data[f'hw0_{level}'][0], w0c=data[f'hw0_{level}'][1])
                mask[pre_conf_c01 <= self.pre_thr[i]] = False

                if self.rt is not None:
                    pre_conf_c01_s = data[f'stage_{pre_level_}']['next_conf_c01_s'].detach()
                    pre_conf_c01_s = rearrange(pre_conf_c01_s, 'b (h0p w0p) -> b 1 h0p w0p', h0p=data[f'hw0_{pre_level_}'][0], w0p=data[f'hw0_{pre_level_}'][1])
                    pre_conf_c01_s = F.interpolate(pre_conf_c01_s, (data[f'hw0_{level}'][0], data[f'hw0_{level}'][1]), mode='nearest')
                    pre_conf_c01_s = rearrange(pre_conf_c01_s, 'b 1 h0c w0c -> b (h0c w0c)', h0c=data[f'hw0_{level}'][0], w0c=data[f'hw0_{level}'][1])

                    ts = pre_conf_c01_s / (pre_conf_c01 + 1e-7)
                    mask[ts > self.rt] = False

                if self.rd is not None and pre_level_ == '8c':
                    coarse_idx_c01 = data[f'stage_{pre_level_}']['next_idx_c01']
                    coarse_idx_c01_s = data[f'stage_{pre_level_}']['next_idx_c01_s']
                    h_, w_ = data[f'hw0_{pre_level_}'][0], data[f'hw0_{pre_level_}'][1]
                    coarse_c01_xy = torch.stack([coarse_idx_c01 % w_, torch.div(coarse_idx_c01, w_, rounding_mode='trunc')], dim=-1).float()
                    coarse_c01_xy[:, :, 0] = coarse_c01_xy[:, :, 0] / w_
                    coarse_c01_xy[:, :, 1] = coarse_c01_xy[:, :, 1] / h_
                    coarse_c01_xy_s = torch.stack([coarse_idx_c01_s % w_, torch.div(coarse_idx_c01_s, w_, rounding_mode='trunc')], dim=-1).float()
                    coarse_c01_xy_s[:, :, 0] = coarse_c01_xy_s[:, :, 0] / w_
                    coarse_c01_xy_s[:, :, 1] = coarse_c01_xy_s[:, :, 1] / h_
                    ds = torch.sqrt(torch.pow(coarse_c01_xy - coarse_c01_xy_s, 2).sum(dim=-1))
                    ds = rearrange(ds, 'b (h0p w0p) -> b 1 h0p w0p', h0p=h_, w0p=w_)
                    ds = F.interpolate(ds, (data[f'hw0_{level}'][0], data[f'hw0_{level}'][1]), mode='nearest')
                    ds = rearrange(ds, 'b 1 h0c w0c -> b (h0c w0c)', h0c=data[f'hw0_{level}'][0], w0c=data[f'hw0_{level}'][1])
                    mask[ds > self.rd] = False

        # border会存在很多误匹配，需要mask
        mask = rearrange(mask, 'b (h0c w0c) -> b h0c w0c', h0c=axes_lengths['h0c'], w0c=axes_lengths['w0c'])
        idx_c01_2d = torch.stack([torch.div(next_idx_c01, axes_lengths['w1c'], rounding_mode='trunc'),
                                  next_idx_c01 % axes_lengths['w1c']], dim=-1)
        idx_c01_2d = rearrange(idx_c01_2d, 'b (h0c w0c) c -> b h0c w0c c', h0c=axes_lengths['h0c'], w0c=axes_lengths['w0c'])
        if f'mask_{level}0' not in data:  # 这里只mask source border
            mask = mask_window_border(mask, idx_c01_2d, self.border_rm, False, axes_lengths['h1c'], axes_lengths['w1c'])
        else:
            mask = mask_window_border_with_padding(mask, idx_c01_2d, self.border_rm, False, data[f'mask_{level}0'], data[f'mask_{level}1'])
        mask = rearrange(mask, 'b h0c w0c -> b (h0c w0c)', h0c=axes_lengths['h0c'], w0c=axes_lengths['w0c'])

        if self.double_check:
            # double check保证0->1的idx和1->0的idx是相同的
            # example: gather(idx_c01:[3,1,2,0], index=idx_c10:[2(False),1(True),3(False),0(True)]) == [0,1,2,3]?
            arange_idx = torch.arange(0, next_idx_c01.shape[1], dtype=next_idx_c01.dtype,
                                      device=next_idx_c01.device)[None].repeat(next_idx_c01.shape[0], 1)
            # [B,HW1]->[B,HW0]
            double_check_mask = (torch.gather(next_idx_c10, dim=1, index=next_idx_c01) == arange_idx)
            mask = mask * double_check_mask  # [B,HW0]


        if mask.sum() == 0:
            mask[:, 0] = True  # 保底一个

        # 3. find all valid coarse matches
        b_ids, i_ids = torch.where(mask)  # [L]
        j_ids = next_idx_c01[mask]  # [L]
        mconf = next_conf_c01[mask]  # [L]
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # convert global gt label to window label
        if self.training:
            global_idx_masked = data[f'gt_stage_{level}']['gt_idx_c01'][mask]  # [L]
            gt_label_mask = data[f'gt_stage_{level}']['gt_mask_c01'][mask]  # gt label mask
            conf_matrix01_masked = conf_matrix01[mask]  # [L,4ww]
            idx_c01_masked = idx_c01[mask]
            window_gt_label = convert_global_to_window_coordinate(global_idx_masked, idx_c01_masked)  # [L,4ww]
            window_gt_label[~gt_label_mask] = 0
            # filter with new mask
            new_mask = (window_gt_label.sum(-1) == 1)
            valid_patch_num = new_mask.sum().item()
            if valid_patch_num == 0:
                new_mask[0] = True  # 保底一个
            if valid_patch_num > self.train_pad_num_gt_min:  # mask超出的部分
                valid_pos = torch.where(new_mask == True)[0]
                valid_pos = valid_pos[torch.randperm(valid_pos.shape[0])[:valid_patch_num - self.train_pad_num_gt_min]]
                new_mask[valid_pos] = False
            b_ids, i_ids, j_ids, mconf = b_ids[new_mask], i_ids[new_mask], j_ids[new_mask], mconf[new_mask]
            window_gt_label = window_gt_label[new_mask]
            conf_matrix01_masked = conf_matrix01_masked[new_mask]
            coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
            coarse_matches['window_gt_label'] = window_gt_label
            coarse_matches['window_conf_matrix'] = conf_matrix01_masked
            coarse_matches['valid_patch_num'] = valid_patch_num

            # for detection
            detector_matrix01 = data[f'stage_{level}']['detector_matrix01']
            if detector_matrix01 is not None:
                detector_conf_c01, _ = torch.max(detector_matrix01, dim=2)  # [B,HW0]
                detector_mask = detector_conf_c01 > (1 / idx_c01.shape[-1])
                detector_mask = detector_mask & mask

                detector_global_idx_masked = data[f'gt_stage_{level}']['gt_idx_c01'][detector_mask]  # [L]
                detector_gt_label_mask = data[f'gt_stage_{level}']['gt_mask_c01'][detector_mask]  # gt label mask
                detector_matrix01_masked = detector_matrix01[detector_mask]  # [L,4ww]
                detector_idx_c01_masked = idx_c01[detector_mask]
                detector_window_gt_label = convert_global_to_window_coordinate(detector_global_idx_masked, detector_idx_c01_masked)  # [L,4ww]
                detector_window_gt_label[~detector_gt_label_mask] = 0

                detector_new_mask = (detector_window_gt_label.sum(-1) == 1)
                detector_valid_patch_num = detector_new_mask.sum().item()
                if detector_valid_patch_num == 0:
                    detector_new_mask[0] = True  # 保底一个
                if detector_valid_patch_num > self.train_pad_num_gt_min:  # mask超出的部分
                    detector_valid_pos = torch.where(detector_new_mask == True)[0]
                    detector_valid_pos = detector_valid_pos[torch.randperm(detector_valid_pos.shape[0])[:valid_patch_num - self.train_pad_num_gt_min]]
                    detector_new_mask[detector_valid_pos] = False
                detector_window_gt_label = detector_window_gt_label[detector_new_mask]
                detector_matrix01_masked = detector_matrix01_masked[detector_new_mask]
                coarse_matches['detector_window_gt_label'] = detector_window_gt_label
                coarse_matches['detector_window_conf_matrix'] = detector_matrix01_masked
                coarse_matches['detector_valid_patch_num'] = detector_valid_patch_num

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data[f'hw0_{level}'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack([i_ids % data[f'hw0_{level}'][1], torch.div(i_ids, data[f'hw0_{level}'][1], rounding_mode='trunc')], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data[f'hw1_{level}'][1], torch.div(j_ids, data[f'hw1_{level}'][1], rounding_mode='trunc')], dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
            'mconf': mconf
        })

        return coarse_matches
