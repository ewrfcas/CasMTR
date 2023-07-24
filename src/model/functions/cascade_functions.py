import fast_score_computation
import torch
import torch.nn.functional as F
from einops.einops import rearrange
from torch.autograd import Function


class ScoreComputation(Function):
    @staticmethod
    def forward(ctx, query, key, index):
        assert query.shape[1] % 16 == 0  # length must be divisible by 16
        assert query.shape[1] / 16 <= 32768
        x = fast_score_computation.score_forward(query, key, index)
        ctx.save_for_backward(query, key, index)
        return x[0]

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, index = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        x = fast_score_computation.score_backward(grad_output, input1, input2, index)
        return x[0], x[1], None

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


def convert_global_to_window_coordinate(global_idx, window_idx):
    # convert global idx to window idx:
    # global_idx: [L] global real 1d coordinate
    # window_idx: [L,ww] window based real 1d coordinate (ensure that it should be sorted in dim=2)
    assert global_idx.shape[0] == window_idx.shape[0]

    global_window_label = (global_idx.unsqueeze(-1) == window_idx).long()  # [L,ww]

    return global_window_label


def convert_global_to_dilated_window_coordinate(global_idx, window_idx, W, dilated):
    # convert global idx to dilated window idx. We should search nearest idx for the gt idx
    # global_idx: [L] global real 1d coordinate
    # window_idx: [L,ww] window based real 1d coordinate (ensure that it should be sorted in dim=2)
    assert global_idx.shape[0] == window_idx.shape[0]
    if dilated == 2:
        threshold = 2
    else:
        raise NotImplementedError('Wrong dilated for', dilated)

    # convert 1d to 2d coordinate
    global_idx_yx = torch.stack([torch.div(global_idx, W, rounding_mode='trunc'), global_idx % W], dim=-1)  # [L,2]
    window_idx_yx = torch.stack([torch.div(window_idx, W, rounding_mode='trunc'), window_idx % W], dim=-1)  # [L,ww,2]

    distance = ((global_idx_yx[:, None] - window_idx_yx) ** 2).sum(-1)  # sum of yx distance [L,ww]
    min_dis, min_idx = torch.min(distance, dim=1)  # [L,]*2
    label_mask = (min_dis <= threshold)
    global_window_label = torch.zeros_like(window_idx)  # [L,ww]
    global_window_label[label_mask, min_idx[label_mask]] = 1

    return global_window_label


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def mask_window_border(mask, idx_2d, b, v, H1, W1):
    """ Mask borders with value
    Args:
        mask (torch.Tensor): [B,H0,W0]
        idx_2d: [B,H0,W0,2] 2d_coordinate (y,x)
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return mask

    mask[:, :b] = v
    mask[:, :, :b] = v
    mask[:, -b:] = v
    mask[:, :, -b:] = v

    tgt_x_mask = (idx_2d[:, :, :, 1] < b) + (idx_2d[:, :, :, 1] > W1 - b)
    tgt_y_mask = (idx_2d[:, :, :, 0] < b) + (idx_2d[:, :, :, 0] > H1 - b)
    tgt_mask = tgt_x_mask + tgt_y_mask
    mask[tgt_mask] = v

    return mask


def mask_window_border_with_padding(mask, idx_2d, b, v, p_m0, p_m1):
    """ Mask borders with value
    Args:
        mask (torch.Tensor): [B,H0,W0]
        idx_2d: [B,H0,W0,2] 2d_coordinate (y,x)
        b (int)
        v (m.dtype)
        p_m0: [B,H0,W0]
    """
    if b <= 0:
        return mask

    mask[:, :b] = v
    mask[:, :, :b] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        mask[b_idx, h0 - b:] = v
        mask[b_idx, :, w0 - b:] = v

        # mask targets
        tgt_x_mask = (idx_2d[b_idx, :, :, 1] < b) + (idx_2d[b_idx, :, :, 1] > w1 - b)
        tgt_y_mask = (idx_2d[b_idx, :, :, 0] < b) + (idx_2d[b_idx, :, :, 0] > h1 - b)
        tgt_mask = tgt_x_mask + tgt_y_mask
        mask[b_idx, tgt_mask] = v

    return mask


def detect_keypoints(heatmap0, conf_matrix01, mode, grid_size):
    # split into grid, conf_matrix01:[B,HW0,k]
    b, _, h, w = heatmap0.shape
    k = conf_matrix01.shape[-1]
    heatmap0 = heatmap0.reshape(b, 1, h // grid_size, grid_size, w // grid_size, grid_size)
    heatmap0 = rearrange(heatmap0, "b c h t1 w t2 -> b c h w (t1 t2)")  # [B,1,h,w,gg]

    # ret:[B,1,h,w,gg](0,1)
    if mode == 'gumbel':
        ret = F.gumbel_softmax(heatmap0, tau=1.0, hard=True, dim=-1)
    elif mode == 'ST':
        y_soft = torch.softmax(heatmap0, dim=-1)
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(heatmap0, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        raise NotImplementedError
    ret = ret.squeeze(1)  # [B,h,w,gg]
    conf_matrix01 = conf_matrix01.reshape(b, h, w, k).reshape(b, h // grid_size, grid_size, w // grid_size, grid_size, k)
    conf_matrix01 = rearrange(conf_matrix01, "b h t1 w t2 k -> b h w (t1 t2) k")  # [B,h,w,gg,k]
    conf_matrix01 = conf_matrix01 * ret.unsqueeze(-1)
    conf_matrix01 = rearrange(conf_matrix01, "b h w (t1 t2) k -> b (h t1) (w t2) k", t1=grid_size)  # [B,H0,W0,k]
    conf_matrix01 = conf_matrix01.reshape(b, -1, k).contiguous()

    return conf_matrix01
