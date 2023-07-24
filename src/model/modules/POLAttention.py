# --------------------------------------------------------
# This script is modified from the following source by Shiyu Zhao
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# --------------------------------------------------------
# Backbones for GMFlowNet
# --------------------------------------------------------
class NeighborWindowAttention(nn.Module):
    """ Patch-based OverLapping multi-head self-Attention (POLA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window (or patch).
        num_heads (int): Number of attention heads.
        neig_win_num (int): Number of neighbor windows. Default: 1
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, neig_win_num=1,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., use_proj=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.use_proj = use_proj

        # define a parameter table of relative position bias
        self.n_win = 2 * neig_win_num + 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((self.n_win + 1) * window_size[0] - 1) * ((self.n_win + 1) * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww

        coords_h_neig = torch.arange(self.n_win * self.window_size[0])
        coords_w_neig = torch.arange(self.n_win * self.window_size[1])
        coords_neig = torch.stack(torch.meshgrid([coords_h_neig, coords_w_neig]))  # 2, Wh, Ww

        coords_flat = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords_neig_flat = torch.flatten(coords_neig, 1)  # 2, (n_win*Wh)*(n_win*Ww)
        relative_coords = coords_flat[:, :, None] - coords_neig_flat[:, None, :]  # 2, Wh*Ww, (n_win*Wh)*(n_win*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww,(n_win*Wh)*(n_win*Ww), 2
        relative_coords[:, :, 0] += self.n_win * self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.n_win * self.window_size[1] - 1
        relative_coords[:, :, 0] *= (self.n_win + 1) * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """ Forward function.
        Args:
            q: input queries with shape of (num_windows*B, N, C)
            k: input keys with shape of (num_windows*B, N, C)
            v: input values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N_q, C = q.shape
        N_kv = k.shape[1]
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        dim_per_head = C // self.num_heads
        q = self.Wq(q).reshape(B_, N_q, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        k = self.Wk(k).reshape(B_, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        v = self.Wv(v).reshape(B_, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.n_win * self.window_size[0] * self.n_win * self.window_size[1], -1)  # Wh*Ww,(n_win*Wh)*(n_win*Ww),nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, (n_win*Wh)*(n_win*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_kv) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_kv)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)
        if self.use_proj:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class MultiHeadAttention(nn.Module):
    """ MultiHeadAttention modified from SwinTransformer
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_proj=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.use_proj = use_proj

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """ Forward function.
        Args:
            q: input queries with shape of (B, Nq, C)
            k: input keys with shape of (B, Nk, C)
            v: input values with shape of (B, Nk, C)
            mask: (0/-inf) mask with shape of (Nq, Nk) or None
        """
        B, N_q, C = q.shape
        N_kv = k.shape[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        dim_per_head = C // self.num_heads
        q = self.Wq(q).reshape(B, N_q, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        k = self.Wk(k).reshape(B, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        v = self.Wv(v).reshape(B, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B, num_heads, Nq, Nk

        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        if self.use_proj:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class POLATransBlock(nn.Module):
    """ Transformer block with POLA.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window/patch size.
        neig_win_num (int): Number of overlapped windows
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, neig_win_num=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.neig_win_num = neig_win_num
        self.mlp_ratio = mlp_ratio

        self.n_win = 2 * neig_win_num + 1

        self.norm1 = norm_layer(dim)

        self.attn = NeighborWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, neig_win_num=neig_win_num,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # print('LocalTransBlock x.shape: ', x.shape)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # partition windows
        x_win = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_win = x_win.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # pad and unfold
        pad_size = self.neig_win_num * self.window_size
        key_val = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size))  # B, H'+2*1*win, W'+2*1*win, C
        key_val = F.unfold(key_val.permute(0, 3, 1, 2), self.n_win * self.window_size, stride=self.window_size)
        key_val = key_val.permute(0, 2, 1).reshape(-1, C, (self.n_win * self.window_size) ** 2).permute(0, 2, 1)  # (B*num_win, (3*3)*win_size*win_size, C)

        # Local attention feature
        attn_windows = self.attn(x_win, key_val, key_val, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MixAxialPOLABlock(nn.Module):
    """ Transformer block with mixture of POLA, vertical and horizontal axis self-attentions
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads=8, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.dim_per_head = dim // self.num_heads
        self.axis_head = 2  # self.num_heads // 4
        self.local_head = self.num_heads - 2 * self.axis_head

        self.local_chl = self.local_head * self.dim_per_head
        self.axis_chl = self.axis_head * self.dim_per_head

        # for POLA
        self.neig_win_num = 1
        self.n_win = 2 * self.neig_win_num + 1
        self.norm1 = norm_layer(dim)
        self.localAttn = NeighborWindowAttention(self.local_chl, window_size=to_2tuple(self.window_size),
                                                 num_heads=self.local_head, neig_win_num=self.neig_win_num,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # for axial attention
        self.vertiAttn = MultiHeadAttention(self.axis_chl, num_heads=self.axis_head,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_proj=False)

        self.horizAttn = MultiHeadAttention(self.axis_chl, num_heads=self.axis_head,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_proj=False)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # print('LocalTransBlock x.shape: ', x.shape)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_local, x_horiz, x_verti = torch.split(x, [self.local_chl, self.axis_chl, self.axis_chl], dim=-1)

        # Local patch update
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_local = F.pad(x_local, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x_local.shape

        # partition windows
        x_windows = window_partition(x_local, self.window_size)  # nW*B, window_size, window_size, C1
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.local_chl)  # nW*B, window_size*window_size, C1

        # pad and unfold
        pad_size = self.neig_win_num * self.window_size
        key_val = F.pad(x_local, (0, 0, pad_size, pad_size, pad_size, pad_size))  # B, H'+2*1*win, W'+2*1*win, C
        key_val = F.unfold(key_val.permute(0, 3, 1, 2), self.n_win * self.window_size, stride=self.window_size)
        key_val = key_val.permute(0, 2, 1).reshape(-1, self.local_chl, (self.n_win * self.window_size) ** 2).permute(0, 2, 1)  # (B*num_win, (3*3)*win_size*win_size, C)

        # Local attention feature
        attn_windows = self.localAttn(x_windows, key_val, key_val, mask=attn_mask)  # nW*B, window_size*window_size, C1

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.local_chl)
        x_local = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C1

        if pad_r > 0 or pad_b > 0:
            x_local = x_local[:, :H, :W, :].contiguous()

        # Horizontal update
        x_horiz = x_horiz.view(-1, W, self.axis_chl)  # (B*H), W, C2
        x_horiz = self.horizAttn(x_horiz, x_horiz, x_horiz)
        x_horiz = x_horiz.view(B, H, W, self.axis_chl)  # B, H, W, C2

        # Vertical update
        x_verti = x_verti.transpose(1, 2).reshape(-1, H, self.axis_chl)  # B, W, H, C3 -> (B*W), H, C3
        x_verti = self.vertiAttn(x_verti, x_verti, x_verti)
        x_verti = x_verti.view(B, W, H, self.axis_chl).transpose(1, 2)  # B, H, W, C3

        x = torch.cat([x_local, x_horiz, x_verti], dim=-1)  # B, H, W, C
        x = x.view(B, H * W, C)

        x = self.proj(x)
        x = self.proj_drop(x)  # B, (H*W), C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
