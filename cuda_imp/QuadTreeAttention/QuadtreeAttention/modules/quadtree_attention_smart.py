import torch
import torch.nn as nn
from einops.einops import rearrange

from ..functions.quadtree_attention import score_computation_op, value_aggregation_op


# smart implementation
def torch_gather_b2(params, indices):
    # this operation is equivalent to tf.gather when batch_dims=2

    if params.shape[:2] != indices.shape[:2]:  # [B, nh]
        raise ValueError(
            f"Make sure that the first two dimensions of params and indices are identical, \
                but they are params: {params.shape[:2]} vs. indices: {params.shape[:2]}"
        )
    num_indices_to_gather = indices.shape[-2] * indices.shape[-1]  # L*K
    num_indices_to_pick_from = params.shape[2]  # L

    indices_shift = (
            torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)  # B*nh*L*K
            // num_indices_to_gather  # L*K
            * num_indices_to_pick_from  # L
    )  # [B*nh*L*K,] 即每L*K个+L，0...,0(L*K), L...,L, 2L...,2L,...

    flattened_indices = indices.reshape(-1) + indices_shift
    flattened_params = params.reshape(-1, params.shape[-1])  # [B,nh,L,C] ---> [B*nh*L,C]

    out_flattened = flattened_params.index_select(0, flattened_indices)  # [B*nh*L,C] gather [B*nh*L*K,]->[B*nh*L*K,C]

    out = out_flattened.reshape(params.shape[:3] + (indices.shape[-1],) + params.shape[3:])  # [B*nh*L*K,C]->[B,nh,L,K,C]
    return out


class QTAttA(nn.Module):
    def __init__(
            self,
            nhead,
            dim,
            topks=[32, 32, 32, 32],
            scale=None,
            use_dropout=False,
            attention_dropout=0.1,
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.dim = dim

    def process_coarse_level(self, query, key, value, topk):
        bs, c, h, w = key.shape
        cur_dim = key.shape[1] // self.nhead

        key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]
        query = rearrange(query, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)

        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        softmax_temp = 1.0 / cur_dim ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)

        # mask out top K tokens
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)
        mask = torch.ones_like(A)
        mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())

        # message is only computed within the unmasked
        message = torch.einsum("nlsh,nshd->nlhd", A * mask, value)  # .reshape(bs, h, w, self.nhead, cur_dim)

        return A, message, topk_score, topk_idx

    def process_fine_level(self, query, key, value, topk_score, topk_pos, topk_prev, topk, final=False):
        bs, c, h, w = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]

        query = query.view(bs, c, h // 2, 2, w // 2, 2)
        query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(bs, -1, 4, self.nhead, cur_dim)

        # convert 2d coordinates to 1d index
        idx_gather = []
        topk_pos = topk_pos * 2
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (topk_pos[0] + x) * w + topk_pos[1] + y  # convert to index
                idx_gather.append(idx)

        idx = torch.stack(idx_gather, dim=3)  # [N, L, K, 4, H, D]

        # Compute score
        # query: [b, N, 4, H, D]
        # key: [b, 4N, H, D]
        # idx: [b, N, K, 4, H]
        # QK: [b, N, 4, 4K, H]
        QK = score_computation_op(query, key.contiguous(), idx.view(bs, -1, topk_prev * 4, self.nhead))
        QK = rearrange(QK, "n l w (k f) h -> n l w k f h", k=topk_prev, f=4)
        softmax_temp = 1.0 / cur_dim ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)  # [N, L//scale**i, K, 4, H]
        # Score redistribution
        topk_score = topk_score.unsqueeze(-2).unsqueeze(2)
        A = (A * topk_score).reshape(bs, -1, 4, topk_prev * 4, self.nhead)
        idx = idx.view(bs, -1, 1, topk_prev * 4, self.nhead).repeat(1, 1, 4, 1, 1)  # [N, L,4, K*4, H]
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)

        if not final:
            mask = torch.ones_like(A)
            mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())
            message = value_aggregation_op(A * mask, value.contiguous(), idx)
        else:
            message = value_aggregation_op(A, value.contiguous(), idx)

        if not final:
            topk_idx = torch.gather(idx, index=topk_idx, dim=-2)
            topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2)  # reshape back
            topk_score = rearrange(
                topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
            )  # reshape back

        return A, message, topk_score, topk_idx

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

        bs = queries[0].shape[0]
        messages = []
        topk = self.topks[0]

        for i, (query, key, value) in enumerate(zip(reversed(queries), reversed(keys), reversed(values))):
            bs, c, h, w = key.shape
            if i == 0:
                A, message, topk_score, topk_idx = self.process_coarse_level(
                    query, key, value, topk
                )  # Full attention for coarest level
            else:
                topk_prev = topk
                topk = self.topks[i]
                final = True if i == len(queries) - 1 else False
                A, message, topk_score, topk_idx = self.process_fine_level(
                    query, key, value, topk_score, topk_pos, topk_prev, topk, final
                )  # Quadtree attention

            messages.append(message)
            if topk_idx is not None:
                topk_pos = torch.stack([topk_idx // w, topk_idx % w])  # convert to coordinate

        final_message = 0
        for i, m in enumerate(messages):
            if i == 0:
                final_message = m
            else:
                final_message = final_message.unsqueeze(2) + m
                final_message = rearrange(
                    final_message, "b (H W) (t1 t2) h d -> b (H t1 W t2) h d", t1=2, t2=2, H=queries[-i].shape[2]
                )

        return final_message

# cuda quadtree 704*704, 130 steps, batch=1, 4GPU, FP16: 1.19s/step, 14.2G
# python quadtree 704*704, 130 steps, batch=1, 4GPU, FP16: 1.34s/step, 25.7G
class QTAttB(nn.Module):
    def __init__(self, nhead, dim, scale, topks=[32, 32, 32, 32], use_dropout=False, attention_dropout=0.1, lepe=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.dim = dim
        self.lepe = lepe
        if lepe:  # locally enhanced position encoding
            self.get_vs = nn.ModuleList(
                [
                    nn.Conv2d(dim * nhead, dim * nhead, kernel_size=3, stride=1, padding=1, groups=dim * nhead)
                    for _ in range(scale)
                ]
            )
        self.register_parameter("weight", nn.Parameter(torch.randn(scale)))

    def process_coarse_level(self, query, key, value, topk):
        bs, c, h, w = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim).contiguous()  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim).contiguous()  # [N, S, H, D]
        query = rearrange(query, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim).contiguous()
        QK = torch.einsum("nlhd,nshd->nlsh", query, key).contiguous()
        softmax_temp = 1.0 / cur_dim ** 0.5  # sqrt(D)

        A = torch.softmax(softmax_temp * QK, dim=-2)  # [B, HW, softmax(HW), nhead]
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)  # [B, HW, topk, nhead]

        message = torch.einsum("nlsh,nshd->nlhd", A, value).contiguous()  # .reshape(bs, h, w, self.nhead, cur_dim)

        return A, message, topk_score, topk_idx

    def process_fine_level(self, query, key, value, topk_score, topk_pos, topk_prev, topk, final=False):
        bs, c, h0, w0 = query.shape
        bs, c, h1, w1 = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim).contiguous()  # [B, Lk, nhead, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim).contiguous()  # [B, Lk, nhead, D]

        query = query.view(bs, c, h0 // 2, 2, w0 // 2, 2)
        query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(bs, -1, 4, self.nhead, cur_dim).contiguous()  # [B, L//4, 4, nhead, D]

        # convert 2D coordiantes to 1D index
        # topk_pos: [2(row,col), B(N), HW(L), topk(K), nhead]
        topk_pos = topk_pos * 2  # feature scale是之前2倍，索引*2
        idx_gather = []
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (topk_pos[0] + x) * w1 + topk_pos[1] + y  # convert to index
                idx_gather.append(idx)
        idx = torch.stack(idx_gather, dim=3)  # [B, L//4, topk, 4, nhead]

        # score computation
        # query: [B, L//4, 4, nhead, D]
        # key: [B, L, nhead, D]
        # idx: [B, L//4, topk, 4, nhead]
        # QK: [B, L//4, 4, topk*4, nhead]
        key = rearrange(key, "b l h d -> b h l d")  # [B,L,nhead,D]->[B,nhead,L,D]
        idx = idx.reshape(bs, -1, topk_prev * 4, self.nhead).unsqueeze(2).repeat(1, 1, 4, 1, 1)  # [B,L//4,4,topk*4,nhead]
        idx_ = rearrange(idx, "b l0 l1 t2 n -> b n (l0 l1) t2")
        topk_key = torch_gather_b2(key, idx_).reshape(bs, self.nhead, -1, 4, topk_prev * 4, cur_dim)  # [B,nhead,L//4,4,topk*4,D]
        topk_key = rearrange(topk_key, "b n l0 l1 t2 d -> b l0 l1 t2 n d").contiguous()  # [B,L//4,4,topk*4,nhead,D]
        QK = torch.sum(query.unsqueeze(-3) * topk_key, dim=-1)  # [B,L//4,4,topk*4,nhead]

        softmax_temp = 1.0 / cur_dim ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)  # [B,L//4,4,topk*4,nhead]

        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)  # [B, L//4, 4, topk_new, nhead]
        # A: [B, L//4, 4, topk*4, nhead]
        # value: [B, L, nhead, D]
        # idx: [B, L//4, 4, topk*4, nhead]
        value = rearrange(value, "b l h d -> b h l d")  # [B,L,nhead,D]->[B,nhead,L,D]
        topk_value = torch_gather_b2(value, idx_).reshape(bs, self.nhead, -1, 4, topk_prev * 4, cur_dim)  # [B,nhead,L//4,4,topk*4,D]
        topk_value = rearrange(topk_value, "b n l0 l1 t2 d -> b l0 l1 t2 n d").contiguous()  # [B,L//4,4,topk*4,nhead,D]
        message = torch.sum(A.unsqueeze(-1) * topk_value, dim=3).contiguous()
        topk_idx = torch.gather(idx, index=topk_idx, dim=-2)  # [B, L//4, 4, topk_new, nhead]
        # [B, L//4, 4, topk_new, nhead]->[B, L, topk_new, nhead] reshape back
        topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h0 // 2, t1=2).contiguous()
        topk_score = rearrange(topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h0 // 2, t1=2).contiguous()

        return A, message, topk_score, topk_idx

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

        bs = queries[0].shape[0]

        messages = []
        topk = self.topks[0]
        for i, (query, key, value) in enumerate(zip(reversed(queries), reversed(keys), reversed(values))):  # reversed!
            bs, c, h, w = key.shape
            if i == 0:  # Full attention for the coarest level
                A, message, topk_score, topk_idx = self.process_coarse_level(query, key, value, topk)
            else:
                topk_prev = topk
                topk = self.topks[i]
                final = True if i == len(queries) - 1 else False
                A, message, topk_score, topk_idx = self.process_fine_level(query, key, value, topk_score, topk_pos, topk_prev, topk, final)

            messages.append(message)
            # convert to coordinate, [2, B, HW, topk, nhead]
            topk_pos = torch.stack([topk_idx // w, topk_idx % w])

        # Merge messages of different layers
        final_message = 0

        weight = torch.softmax(self.weight, dim=0)
        for i, m in enumerate(messages):
            if self.lepe:
                H, W = values[-(i + 1)].shape[-2:]
                lepe = self.get_vs[i](values[-(i + 1)])

            if i == 0:
                if self.lepe:
                    lepe = rearrange(lepe, "b (hd d) H W -> b (H W) hd d", hd=self.nhead)
                    final_message = (m + lepe) * weight[i]
                else:
                    final_message = m * weight[i]
            else:
                # [B, L//4, nhead, D]->[B, L//4, 1, nhead, D]-->[B, L//4, 4, nhead, D]
                if self.lepe:
                    lepe = rearrange(lepe, "b (hd d) (H t1) (W t2) -> b (H W) (t1 t2) hd d", hd=self.nhead, t1=2, t2=2)
                    final_message = final_message.unsqueeze(2) + (m + lepe) * weight[i]
                else:
                    final_message = final_message.unsqueeze(2) + m * weight[i]

                final_message = rearrange(final_message, "b (H W) (t1 t2) h d -> b (H t1 W t2) h d", t1=2, t2=2, H=queries[-i].shape[2]).contiguous()

        return final_message
