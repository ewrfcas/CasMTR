import copy

import torch
from einops.einops import repeat
from kornia.utils.grid import create_meshgrid

from src.model.functions.cascade_functions import torch_gather
from .POLAttention import POLATransBlock
from .cascade_attention import *
from .linear_attention import LinearAttention, FullAttention
from .propagations import get_propagations
from .quadtree_attention import QuadtreeAttention, CascadeQuadtreeAttention


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dwconv(x, H, W)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        with torch.cuda.amp.autocast(enabled=False):
            message = self.attention(query.to(torch.float32), key.to(torch.float32), value.to(torch.float32),
                                     q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class QuadtreeBlock(nn.Module):

    def __init__(self, dim, num_heads, topks, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, scale=1, attn_type='B'):

        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = QuadtreeAttention(dim, num_heads=num_heads, topks=topks, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      attn_drop=attn_drop, proj_drop=drop, scale=scale, attn_type=attn_type)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'init'):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W, H1=None, W1=None, rel_pos=None, topk_pos=None):

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(target), H, W, H1, W1, rel_pos=rel_pos, topk_pos=topk_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config, train_size):
        super(LocalFeatureTransformer, self).__init__()
        self.block_type = config['block_type']
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.relative_pe = config.get('relative_pe', False)
        self.train_size = train_size
        if self.relative_pe:  # for quadtree scale=3
            self.h_pos_bias = nn.ModuleList()
            self.w_pos_bias = nn.ModuleList()
            for i in range(3):
                s = 2 ** i
                self.h_pos_bias.append(nn.Linear(self.train_size // s, self.nhead, bias=False))
                self.h_pos_bias.append(nn.Linear(self.train_size // s, self.nhead, bias=False))

        if config['block_type'] == 'loftr':
            encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
            self._reset_parameters()
        elif config['block_type'] == 'quadtree':
            encoder_layer = QuadtreeBlock(config['d_model'], config['nhead'], attn_type=config['attn_type'], topks=config['topks'], scale=3)
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _cal_2d_pos_emb(self, x, scale_i):  # [B,C,H,W]
        _, _, max_h, max_w = x.shape
        s = 2 ** scale_i
        grid_2d = create_meshgrid(max_h // s, max_w // s, False, device=x.device, dtype=torch.long)  # [B,h,w,2] (xy)
        grid_2d = rearrange(grid_2d, "b h w c -> b (h w) c")  # [B,hw,2]
        position_coord_x = grid_2d[..., 0]
        position_coord_y = grid_2d[..., 1]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)  # [B,hw,hw]
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.train_size // s,
            max_distance=max_w // s,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.train_size // s,
            max_distance=max_h // s,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.train_size // s).type_as(x)  # [B,hw,hw,c]
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.train_size // s).type_as(x)
        rel_pos_x = self.w_pos_bias[scale_i](rel_pos_x).permute(0, 3, 1, 2)  # [B,nhead,hw,hw]
        rel_pos_y = self.h_pos_bias[scale_i](rel_pos_y).permute(0, 3, 1, 2)  # [B,nhead,hw,hw]
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        if len(feat0.shape) == 4:
            B, C, H0, W0 = feat0.shape
            _, _, H1, W1 = feat1.shape
            if self.relative_pe:
                rel_pos = []
                for i in range(3):
                    rel_pos.append(self._cal_2d_pos_emb(feat0, i))
            else:
                rel_pos = None
            feat0 = rearrange(feat0, 'b c h w -> b (h w) c')
            feat1 = rearrange(feat1, 'b c h w -> b (h w) c')
        else:
            H0, W0, H1, W1 = 0, 0, 0, 0
            rel_pos = None

        if self.block_type == 'loftr':
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                elif name == 'cross':
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)
                else:
                    raise KeyError
        else:
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, H0, W0, rel_pos=rel_pos)
                    feat1 = layer(feat1, feat1, H1, W1, rel_pos=rel_pos)
                elif name == 'cross':
                    if self.config['block_type'] == 'quadtree':
                        feat0, feat1 = layer(feat0, feat1, H0, W0, H1, W1, rel_pos=rel_pos), layer(feat1, feat0, H1, W1, H0, W0, rel_pos=rel_pos)
                    else:
                        feat0 = layer(feat0, feat1, H0, W0, H1, W1, rel_pos=rel_pos)
                        feat1 = layer(feat1, feat0, H1, W1, H0, W0, rel_pos=rel_pos)
                else:
                    raise KeyError

        return feat0, feat1


class CascadeQuadtreeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, scale=2, dilated=1):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = CascadeQuadtreeAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             attn_drop=attn_drop, proj_drop=drop, scale=scale, dilated=dilated)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'init'):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W, H1=None, W1=None, idx=None, rel_pos=None):
        y, upsampled_idx = self.attn(self.norm1(x), self.norm1(target), H, W, H1, W1, idx, rel_pos)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x, upsampled_idx


class CascadeFeatureTransformer(nn.Module):
    def __init__(self, config, train_size):
        super(CascadeFeatureTransformer, self).__init__()
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.window_size = config['window_size']
        self.attn_window_size = config.get('attn_window_size', None)
        if self.attn_window_size is None:
            self.attn_window_size = self.window_size
        self.sr_ratio = config['sr_ratio']
        self.propagation = config.get('propagation', 'window')
        self.dilated = config.get('dilated', 1)
        window, full_window = get_propagations(config)
        self.window = nn.Parameter(window, requires_grad=False)
        self.full_window = full_window
        self.relative_pe = config.get('relative_pe', False)
        self.train_size = train_size
        if self.relative_pe:  # for quadtree scale=3
            if self.sr_ratio == 2:
                self.LB = self.window_size * 2
            else:
                self.LB = self.window_size * 6
            self.h_pos_bias = nn.Embedding(self.LB * 2 + self.sr_ratio, self.nhead)
            self.w_pos_bias = nn.Embedding(self.LB * 2 + self.sr_ratio, self.nhead)
        self.layers = nn.ModuleList()
        for name in self.layer_names:
            if name == 'self':
                if config['self_attn_type'] == 'local_global':
                    self.layers.append(DoubleGroupBlock(self.d_model, self.nhead, mlp_ratio=4., sr_ratio=self.sr_ratio, ws=self.attn_window_size))
                elif config['self_attn_type'] == 'local':
                    self.layers.append(LocalBlock(self.d_model, self.nhead, mlp_ratio=4., ws=self.attn_window_size))
                elif config['self_attn_type'] == 'SK':
                    raise NotImplementedError(f"Not implemented {config['self_attn_type']} for cascade self attention")
                elif config['self_attn_type'] == 'LKA':
                    self.layers.append(LKABlock(self.d_model, mlp_ratio=4.))
                elif config['self_attn_type'] == 'topk':
                    self.layers.append(QuadtreeBlock(config['d_model'], config['nhead'], attn_type='Guided', topks=config['topks'], scale=len(config['topks'])))
                elif config['self_attn_type'] == 'POLA':
                    self.layers.append(POLATransBlock(config['d_model'], config['nhead'], window_size=self.attn_window_size, mlp_ratio=4.))
                elif config['self_attn_type'] == 'linear':
                    self.layers.append(LoFTREncoderLayer(config['d_model'], config['nhead'], attention='linear'))
                else:
                    raise NotImplementedError(f"Not implemented {config['self_attn_type']} for cascade self attention")
            elif name == 'cross':
                self.layers.append(CascadeQuadtreeBlock(self.d_model, self.nhead, scale=2, dilated=config.get('dilated', 1)))
            else:
                raise NotImplementedError

        detector = config.get('detector', None)
        if detector == 'learnable':
            self.detector = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(self.d_model), nn.SiLU(inplace=True),
                                          nn.Conv2d(self.d_model, 1, kernel_size=1))
        else:
            self.detector = None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_window_warp_idx(self, idx, B, H, W):
        # idx [B,HW]-->[B,HW,2] or [B,HW,topk]-->[B,HW,topk,2] convert 1d to 2d coordinate (y,x)
        idx_yx_2d = torch.stack([torch.div(idx, W, rounding_mode='trunc'), idx % W], dim=-1)
        if len(idx_yx_2d.shape) == 3:
            idx_yx = idx_yx_2d.reshape(B, -1, 1, 2) + self.window.reshape(1, 1, self.window.shape[0], 2)  # [B,HW,w*w,2]
            if self.full_window is not None:
                idx_yx_full = idx_yx_2d.reshape(B, -1, 1, 2) + self.full_window.reshape(1, 1, self.full_window.shape[0], 2).to(idx_yx.device)  # [B,HW,fw*fw,2]
            else:
                idx_yx_full = None

            # mask for over boundary
            under_bound = torch.min(idx_yx, dim=2, keepdim=True)[0]
            under_bound = under_bound * (under_bound < 0).long()
            over_bound = torch.max(idx_yx, dim=2, keepdim=True)[0]
            over_bound[..., 0] = (over_bound[..., 0] - (H - 1)) * (over_bound[..., 0] >= H).long()
            over_bound[..., 1] = (over_bound[..., 1] - (W - 1)) * (over_bound[..., 1] >= W).long()

            idx_yx = idx_yx - under_bound - over_bound  # [B,HW,ww,2]
            if idx_yx_full is not None:
                idx_yx_full = idx_yx_full - under_bound - over_bound  # [B,HW,fw*fw,2]
        else:  # topk use topk_idx directly
            idx_yx = idx_yx_2d
            idx_yx_full = None

        return idx_yx, idx_yx_full

    def upsample_idx(self, topk_pos, h0, h1, w1):
        bs = topk_pos.shape[0]
        k = topk_pos.shape[2]
        topk_pos = rearrange(topk_pos, "b l k rc -> rc b l k")
        topk_pos = topk_pos * 2  # feature scale是之前2倍，索引*2
        idx_gather = []
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (topk_pos[0] + x) * w1 + topk_pos[1] + y  # convert to index
                idx_gather.append(idx)
        idx = torch.stack(idx_gather, dim=3)  # [B, L//4, topk, 4]
        idx = torch.clamp(idx, min=0, max=(h1 * 2) * (w1 * 2) - 1)
        idx = idx.view(bs, -1, 1, k * 4).repeat(1, 1, 4, 1)  # [B, L//4, 4, topk*4]
        upsampled_idx = rearrange(idx, "b (H W) (t1 t2) k -> b (H t1 W t2) k", t1=2, H=h0)  # [B,L,topk*4]

        return upsampled_idx

    def get_cycle_topk(self, data):
        conf_matrix = data['stage_8c']['conf_matrix']  # [B,HW0,HW1]
        topk_idx_c01 = torch.topk(conf_matrix, dim=2, k=self.config['topks'][0], largest=True)[1]  # [B,HW0,topk]
        topk_idx_c10 = torch.topk(conf_matrix, dim=1, k=self.config['topks'][0], largest=True)[1].permute(0, 2, 1)  # [B,HW1,topk]
        cycle_idx_c0 = torch_gather(topk_idx_c10, topk_idx_c01[:, :, 0:1]).squeeze(2)  # [B,HW0,topk]
        cycle_idx_c1 = torch_gather(topk_idx_c01, topk_idx_c10[:, :, 0:1]).squeeze(2)  # [B,HW1,topk]

        cycle_idx_c0 = repeat(torch.stack([torch.div(cycle_idx_c0, data['hw0_8c'][1], rounding_mode='trunc'),
                                           cycle_idx_c0 % data['hw0_8c'][1]]), 't2 b l k -> t2 b l k nh', nh=self.nhead)
        cycle_idx_c1 = repeat(torch.stack([torch.div(cycle_idx_c1, data['hw1_8c'][1], rounding_mode='trunc'),
                                           cycle_idx_c1 % data['hw1_8c'][1]]), 't2 b l k -> t2 b l k nh', nh=self.nhead)

        return cycle_idx_c0, cycle_idx_c1

    def get_relative_pe(self, data, H, window_idx, device, i=0):
        h, w = data[f'hw{i}_8c'][0], data[f'hw{i}_8c'][1]
        w1 = data[f'hw{1 - i}_8c'][1]
        s = H // h
        W1 = w1 * s
        src_rel_pos = create_meshgrid(s, s, False, device=device, dtype=torch.long)  # [1,s,s,2]
        src_rel_pos = repeat(src_rel_pos, '1 s1 s2 c -> 1 hw s1 s2 c', hw=h * w)
        src_rel_pos = rearrange(src_rel_pos, '1 (h w) s1 s2 c -> 1 (h s1) (w s2) c', h=h, w=w).reshape(1, -1, 1, 2)  # [1,H,W,2]->[1,HW,1,2]

        tgt_idx = data['stage_8c']['next_idx_c01'] if i == 0 else data['stage_8c']['next_idx_c10']  # [B,hw]
        tgt_idx_2d = torch.stack([tgt_idx % w1, torch.div(tgt_idx, w1, rounding_mode='trunc')], dim=-1)  # [B,hw,2]
        tgt_idx_2d = repeat(tgt_idx_2d, 'b hw c -> b hw ss c', ss=s * s)
        tgt_idx_2d = rearrange(tgt_idx_2d, 'b (h w) (s1 s2) c -> b (h s1 w s2) c', s1=s, h=h) * s + (s // 2 - 1)  # [B,HW,2]

        window_idx = window_idx * 2  # [B,H/2*W/2,ww,2]
        idx_gather = []
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (window_idx[..., 0] + x) * W1 + window_idx[..., 1] + y  # convert to index
                idx_gather.append(idx)
        window_idx = torch.stack(idx_gather, dim=3)  # [B,H/2*W/2,ww,4]
        window_idx = rearrange(window_idx, 'b hw k t4 -> b hw (k t4)')  # [B,H/2*W/2,4ww]
        window_idx = repeat(window_idx, 'b hw k -> b t hw k', t=4)
        window_idx = rearrange(window_idx, 'b (t1 t2) (h w) k -> b (h t1 w t2) k', t1=2, h=H // 2)  # [B,HW,4ww]
        window_idx = torch.stack([window_idx % W1, torch.div(window_idx, W1, rounding_mode='trunc')], dim=-1)  # [B,HW,4ww,2]

        tgt_rel_pos = tgt_idx_2d.unsqueeze(2) - window_idx + self.LB  # [B,HW,4ww,2]

        rel_pos = src_rel_pos - tgt_rel_pos  # [B,HW,4ww,2]
        rel_pos = rel_pos + 2 * self.LB  # [0~s+2LB]
        rel_pos_x = self.w_pos_bias(rel_pos[..., 0]).permute(0, 3, 1, 2)  # [B,nhead,HW,4ww]
        rel_pos_y = self.h_pos_bias(rel_pos[..., 1]).permute(0, 3, 1, 2)  # [B,nhead,HW,4ww]
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y

        return rel_2d_pos

    def forward(self, feat0, feat1, idx_c01, idx_c10, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
            idx_c01: [N, L] or [N, L, topk]
        """

        B, C, H0, W0 = feat0.shape
        _, _, H1, W1 = feat1.shape
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c')
        feat1 = rearrange(feat1, 'b c h w -> b (h w) c')

        # get window based warping idxs (using previous H,W)
        idx_c01, idx_c01_full = self.get_window_warp_idx(idx_c01, B, H0 // 2, W0 // 2)  # [B,HW,ww,2]
        idx_c10, idx_c10_full = self.get_window_warp_idx(idx_c10, B, H1 // 2, W1 // 2)  # [B,HW,ww,2]
        idx_c01_x2 = None
        idx_c10_x2 = None

        # get relative PE
        if self.relative_pe:
            rel_pe_c01 = self.get_relative_pe(data, H0, idx_c01, feat0.device, i=0)
            rel_pe_c10 = self.get_relative_pe(data, H1, idx_c10, feat1.device, i=1)
        else:
            rel_pe_c01, rel_pe_c10 = None, None

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                if self.config['self_attn_type'] == 'topk':  # get cycle match topk from the coarest level
                    [cycle_idx_c0, cycle_idx_c1] = self.get_cycle_topk(data)
                    feat0, feat1 = layer(feat0, feat0, H0, W0, topk_pos=cycle_idx_c0), layer(feat1, feat1, H1, W1, topk_pos=cycle_idx_c1)
                elif self.config['self_attn_type'] == 'linear':
                    feat0, feat1 = layer(feat0, feat0), layer(feat1, feat1)
                else:
                    feat0, feat1 = layer(feat0, H0, W0), layer(feat1, H1, W1)
            elif name == 'cross':
                [feat0, idx_c01_x2], [feat1, idx_c10_x2] = layer(feat0, feat1, H0, W0, H1, W1, idx_c01, rel_pe_c01), layer(feat1, feat0, H1, W1, H0, W0, idx_c10, rel_pe_c10)

        idx_c01_full = idx_c01_x2 if idx_c01_full is None else self.upsample_idx(idx_c01_full, H0, H1, W1)
        idx_c10_full = idx_c10_x2 if idx_c10_full is None else self.upsample_idx(idx_c10_full, H1, H0, W0)

        if self.detector is not None:
            feat0_ = rearrange(feat0, 'b (h w) c -> b c h w', h=H0)
            heatmap0 = self.detector(feat0_)
        else:
            heatmap0 = None

        return feat0.contiguous(), feat1.contiguous(), idx_c01_full, idx_c10_full, heatmap0
