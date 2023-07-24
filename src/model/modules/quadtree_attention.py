import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from cuda_imp.QuadTreeAttention.QuadtreeAttention.modules.quadtree_attention import QTAttA, QTAttB, CascadeQTAttB, QTAttGuided


class QuadtreeAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            topks,
            value_branch=False,
            act=nn.GELU(),
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            scale=1,
            attn_type="B",
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.attn_type = attn_type
        if attn_type == "Guided":
            self.py_att = QTAttGuided(num_heads, dim // num_heads, scale=scale, topks=topks)
        elif attn_type == "A":
            self.py_att = QTAttA(num_heads, dim // num_heads, scale=scale, topks=topks)
        else:
            self.py_att = QTAttB(num_heads, dim // num_heads, scale=scale, topks=topks)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = scale

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            trunc_normal_(m.weight, std=0.02)
            m.init = True
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W, H1=None, W1=None, rel_pos=None, topk_pos=None):
        H1 = H if H1 is None else H1
        W1 = W if W1 is None else W1
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        target = target.permute(0, 2, 1).reshape(B, C, H1, W1).contiguous()
        keys = []
        values = []
        queries = []

        q = self.q_proj(x)
        k = self.k_proj(target)
        v = self.v_proj(target)
        for i in range(self.scale):
            keys.append(k.to(torch.float32))
            values.append(v.to(torch.float32))
            queries.append(q.to(torch.float32))

            if i != self.scale - 1:
                k = F.avg_pool2d(k, kernel_size=2, stride=2)
                q = F.avg_pool2d(q, kernel_size=2, stride=2)
                v = F.avg_pool2d(v, kernel_size=2, stride=2)

        if self.attn_type == 'Guided':
            msg = self.py_att(queries, keys, values, rel_pos=rel_pos, topk_pos=topk_pos).view(B, -1, C).contiguous()
        else:
            msg = self.py_att(queries, keys, values, rel_pos=rel_pos).view(B, -1, C).contiguous()

        x = self.proj(msg)
        x = self.proj_drop(x)

        return x


class CascadeQuadtreeAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            scale=2,
            dilated=1
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.cross_attn = CascadeQTAttB(num_heads, dim // num_heads, dilated=dilated)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = scale

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            trunc_normal_(m.weight, std=0.02)
            m.init = True
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W, H1=None, W1=None, idx=None, rel_pos=None):
        H1 = H if H1 is None else H1
        W1 = W if W1 is None else W1
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        target = target.permute(0, 2, 1).reshape(B, C, H1, W1).contiguous()

        q = self.q_proj(x)
        k = self.k_proj(target)
        v = self.v_proj(target)

        if rel_pos is not None:
            rel_pos = rel_pos.to(torch.float32)
        msg, upsampled_idx = self.cross_attn(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), idx, rel_pos)
        msg = msg.view(B, -1, C).contiguous()

        x = self.proj(msg)
        x = self.proj_drop(x)

        return x, upsampled_idx
