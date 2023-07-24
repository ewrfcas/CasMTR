import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .functions.cascade_matching import CascadeMatching
from .functions.coarse_matching import CoarseMatching
from .functions.fine_matching import CascadeFineMatching, CascadeFinePreprocess
from .functions.position_encoding import PositionEncodingSineNorm
from .modules.transformer import LocalFeatureTransformer, CascadeFeatureTransformer


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class UpBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.inner = nn.Sequential(nn.Conv2d(dim1, dim2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(dim2))
        self.up = nn.Sequential(
            nn.Conv2d(dim2, dim2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU()
        )

    def forward(self, feat_2x_c0, feat_2x_c1, feat_c0, feat_c1, size0, size1, bs):
        if size0 == size1:
            feat_2x_c = torch.cat([feat_2x_c0, feat_2x_c1], dim=0)
            feat_c = F.interpolate(torch.cat([feat_c0, feat_c1], dim=0), scale_factor=2, mode='bilinear', align_corners=True)
            feat_c = self.inner(feat_c)
            feat_2x_c = self.up(feat_2x_c + feat_c)
            (feat_2x_c0, feat_2x_c1) = feat_2x_c.split(bs)
        else:
            feat_2x_c0 = self.up(feat_2x_c0 + self.inner(F.interpolate(feat_c0, scale_factor=2, mode='bilinear', align_corners=True)))
            feat_2x_c1 = self.up(feat_2x_c1 + self.inner(F.interpolate(feat_c1, scale_factor=2, mode='bilinear', align_corners=True)))

        return feat_2x_c0, feat_2x_c1


def get_match_config(config, idx):
    new_config = {}
    for k in config:
        if type(config[k]) == list:
            new_config[k] = config[k][idx]
        else:
            new_config[k] = config[k]
    return new_config


def set_stage_mask(data, level):
    mask_c0 = mask_c1 = None  # mask is useful in training
    if 'mask0_origin' in data:
        mask_c0 = F.interpolate(data['mask0_origin'].unsqueeze(1).float(), size=data[f'hw0_{level}'], mode='nearest')[:, 0].bool()
        mask_c1 = F.interpolate(data['mask1_origin'].unsqueeze(1).float(), size=data[f'hw1_{level}'], mode='nearest')[:, 0].bool()
        data[f'mask_{level}0'] = mask_c0
        data[f'mask_{level}1'] = mask_c1
        mask_c0, mask_c1 = mask_c0.flatten(-2), mask_c1.flatten(-2)
    return mask_c0, mask_c1, data


def temp_outputs(data, level):
    data['m_bids'] = data[f'stage_{level}']['m_bids']
    data['mkpts0_f'] = data[f'stage_{level}']['mkpts0_c']
    data['mkpts1_f'] = data[f'stage_{level}']['mkpts1_c']


class CasMTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.stage = config['training_stage']
        block_dims = config['resnetfpn']['block_dims']
        train_size = config['train_size']  # dynamically adjust the PE size

        # Modules
        self.backbone = build_backbone(config)

        self.pos_encoding_8c = PositionEncodingSineNorm(block_dims[2], max_shape=(train_size // 8, train_size // 8))
        self.loftr_coarse_8c = LocalFeatureTransformer(config['coarse'], train_size // 8)
        self.coarse_matching_8c = CoarseMatching(config['match_coarse'], config['coarse'])

        self.pos_encoding_4c = PositionEncodingSineNorm(block_dims[1], max_shape=(train_size // 4, train_size // 4))
        self.up_block1 = UpBlock(block_dims[2], block_dims[1])
        self.loftr_coarse_4c = CascadeFeatureTransformer(config['coarse2'], train_size // 4)
        self.cascade_matching_4c = CascadeMatching(get_match_config(config['match_cascade'], 0), config['coarse2'], self.stage)

        self.pos_encoding_2c = PositionEncodingSineNorm(block_dims[0], max_shape=(train_size // 2, train_size // 2))
        self.up_block2 = UpBlock(block_dims[1], block_dims[0])
        self.loftr_coarse_2c = CascadeFeatureTransformer(config['coarse3'], train_size // 2)
        self.cascade_matching_2c = CascadeMatching(get_match_config(config['match_cascade'], 1), config['coarse3'], self.stage)

        self.fine_preprocess = CascadeFinePreprocess(config, config['fine'], config['coarse2'], coarse_level='2c')
        self.loftr_fine = LocalFeatureTransformer(config["fine"], train_size // 2)
        self.fine_matching = CascadeFineMatching(coarse_level='2c')

        self.autocast = torch.cuda.amp.autocast

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        # from torch.profiler import profile, record_function, ProfilerActivity
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_8c, feats_4c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_8c0, feat_8c1) = feats_8c.split(data['bs'])
            (feat_4c0, feat_4c1) = feats_4c.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_8c0, feat_4c0, feat_f0), (feat_8c1, feat_4c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_8c': feat_8c0.shape[2:], 'hw1_8c': feat_8c1.shape[2:],
            'hw0_4c': feat_4c0.shape[2:], 'hw1_4c': feat_4c1.shape[2:],
            'hw0_2c': feat_f0.shape[2:], 'hw1_2c': feat_f1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_8c0 = self.pos_encoding_8c(feat_8c0)
        feat_8c1 = self.pos_encoding_8c(feat_8c1)
        mask_8c0, mask_8c1, data = set_stage_mask(data, level='8c')
        feat_8c0, feat_8c1 = self.loftr_coarse_8c.forward(feat_8c0, feat_8c1, mask_8c0, mask_8c1)

        # 3. match coarse-level
        with self.autocast(enabled=False):
            self.coarse_matching_8c.forward(feat_8c0.to(torch.float32), feat_8c1.to(torch.float32), data,
                                            mask_c0=mask_8c0, mask_c1=mask_8c1, level='8c')

        if self.stage == 1:
            temp_outputs(data, level='8c')

        if self.stage >= 2:
            feat_8c0 = rearrange(feat_8c0, 'b (h w) c -> b c h w', h=data['hw0_8c'][0], w=data['hw0_8c'][1])
            feat_8c1 = rearrange(feat_8c1, 'b (h w) c -> b c h w', h=data['hw1_8c'][0], w=data['hw1_8c'][1])
            feat_4c0, feat_4c1 = self.up_block1.forward(feat_4c0, feat_4c1, feat_8c0, feat_8c1, data['hw0_4c'], data['hw1_4c'], data['bs'])

            feat_4c0 = self.pos_encoding_4c(feat_4c0)
            feat_4c1 = self.pos_encoding_4c(feat_4c1)
            mask_4c0, mask_4c1, data = set_stage_mask(data, level='4c')
            next_idx_c01 = data['stage_8c']['next_idx_c01_topk']
            next_idx_c10 = data['stage_8c']['next_idx_c10_topk']
            if next_idx_c01 is None or next_idx_c10 is None:
                next_idx_c01 = data['stage_8c']['next_idx_c01']
                next_idx_c10 = data['stage_8c']['next_idx_c10']
            feat_4c0, feat_4c1, idx_4c01, idx_4c10, heatmap_4c0 = self.loftr_coarse_4c.forward(feat_4c0, feat_4c1, next_idx_c01, next_idx_c10, data=data)

            with self.autocast(enabled=False):
                self.cascade_matching_4c.forward(feat_4c0.to(torch.float32), feat_4c1.to(torch.float32), idx_4c01, idx_4c10, data,
                                                 mask_c0=mask_4c0, mask_c1=mask_4c1, heatmap_c0=heatmap_4c0, level='4c', pre_level='8c')

            if self.stage == 2:
                temp_outputs(data, level='4c')

        if self.stage >= 3:
            feat_4c0 = rearrange(feat_4c0, 'b (h w) c -> b c h w', h=data['hw0_4c'][0], w=data['hw0_4c'][1])
            feat_4c1 = rearrange(feat_4c1, 'b (h w) c -> b c h w', h=data['hw1_4c'][0], w=data['hw1_4c'][1])
            feat_2c0, feat_2c1 = self.up_block2.forward(feat_f0, feat_f1, feat_4c0, feat_4c1, data['hw0_2c'], data['hw1_2c'], data['bs'])

            feat_2c0 = self.pos_encoding_2c(feat_2c0)
            feat_2c1 = self.pos_encoding_2c(feat_2c1)
            mask_2c0, mask_2c1, data = set_stage_mask(data, level='2c')
            next_idx_c01 = data['stage_4c']['next_idx_c01_topk']
            next_idx_c10 = data['stage_4c']['next_idx_c10_topk']
            if next_idx_c01 is None or next_idx_c10 is None:
                next_idx_c01 = data['stage_4c']['next_idx_c01']
                next_idx_c10 = data['stage_4c']['next_idx_c10']
            feat_2c0, feat_2c1, idx_2c01, idx_2c10, heatmap_2c0 = self.loftr_coarse_2c.forward(feat_2c0, feat_2c1, next_idx_c01, next_idx_c10, data=data)

            with self.autocast(enabled=False):
                self.cascade_matching_2c.forward(feat_2c0.to(torch.float32), feat_2c1.to(torch.float32), idx_2c01, idx_2c10, data,
                                                 mask_c0=mask_2c0, mask_c1=mask_2c1, heatmap_c0=heatmap_2c0, level='2c', pre_level=['8c', '4c'])

            # fine-level refinement
            feat_2c0 = rearrange(feat_2c0, 'b (h w) c -> b c h w', h=data['hw0_2c'][0], w=data['hw0_2c'][1])
            feat_2c1 = rearrange(feat_2c1, 'b (h w) c -> b c h w', h=data['hw1_2c'][0], w=data['hw1_2c'][1])
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess.forward(feat_2c0, feat_2c1, feat_c0=None, feat_c1=None, data=data)
            if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

            # match fine-level
            with self.autocast(enabled=False):
                self.fine_matching.forward(feat_f0_unfold.to(torch.float32), feat_f1_unfold.to(torch.float32), data)


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
