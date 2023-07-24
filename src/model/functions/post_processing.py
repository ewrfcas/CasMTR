import kornia
import torch
import torch.nn.functional as F
from einops.einops import rearrange
from kornia.feature import *


def get_laf_pts_to_draw_(LAF: torch.Tensor, img_idx: int = 0):
    """Return numpy array for drawing LAFs (local features).

    Args:
        LAF:
        n_pts: number of boundary points to output.

    Returns:
        tensor of boundary points.

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, n_pts, 2)`

    Examples:
        x, y = get_laf_pts_to_draw(LAF, img_idx)
        plt.figure()
        plt.imshow(kornia.utils.tensor_to_image(img[img_idx]))
        plt.plot(x, y, 'r')
        plt.show()
    """
    # TODO: Refactor doctest
    # KORNIA_CHECK_LAF(LAF)
    pts = laf_to_boundary_points(LAF[img_idx: img_idx + 1])[0]
    return (pts[..., 0], pts[..., 1])


class PostProcess(object):
    def __init__(self, post_config):
        self.config = post_config
        self.method = post_config['method']
        self.detector = None

    def apply(self, data, axes_lengths, next_idx_c01, next_conf_c01, test_thr, level):
        bs = next_idx_c01.shape[0]
        if self.method is None:
            mask = next_conf_c01 > test_thr
        elif self.method == 'sift':  # sift filtering
            if self.detector is None:
                self.detector = ScaleSpaceDetector(4096, resp_module=BlobHessian(),
                                                   nms_module=kornia.geometry.ConvQuadInterp3d(10),
                                                   scale_pyr_module=kornia.geometry.ScalePyramid(3, 1.6, 64, double_image=True),
                                                   ori_module=kornia.feature.LAFOrienter(19),
                                                   mr_size=0.1).to(data['image0'].device)
            timg_gray = kornia.color.rgb_to_grayscale(data['image0'])  # [B,N,2,3]
            # return for unpadded
            mask0_origin = data['mask0_origin']
            valid_h = mask0_origin.sum(1).max(1)[0]
            valid_w = mask0_origin.sum(2).max(1)[0]
            lafs = []
            for i in range(bs):
                timg_gray_ = timg_gray[i:i + 1, :, :valid_h[i], :valid_w[i]]
                lafs_, _ = self.detector.forward(timg_gray_)
                lafs.append(lafs_)
            lafs = torch.cat(lafs, dim=0)
            sift_idx0 = []
            for i in range(bs):
                x0, y0 = get_laf_pts_to_draw_(lafs, i)
                sift_idx0.append(torch.cat([torch.mean(y0, dim=1, keepdim=True), torch.mean(x0, dim=1, keepdim=True)], dim=1))  # [N,2]
            sift_idx0 = torch.stack(sift_idx0, dim=0) / int(level.replace('c', ''))  # [B,N,2]
            if torch.isnan(sift_idx0.mean()):
                mask = next_conf_c01 > test_thr
            else:
                sift_idx0 = sift_idx0[:, :, 0] * axes_lengths['w0c'] + sift_idx0[:, :, 1]
                sift_idx0 = torch.clamp(sift_idx0, 0, axes_lengths['h0c'] * axes_lengths['w0c'] - 1).round().long()
                mask = torch.zeros_like(next_conf_c01).bool()  # [B,HW0]
                mask = torch.scatter(mask, index=sift_idx0, dim=1, value=True)  # [B,HW0]
                mask[next_conf_c01 <= test_thr] = False
        elif self.method == 'local_window_nms':  # topk filtering in local patches
            window_size = self.config['window_size']
            topk = self.config['topk']
            hw0 = axes_lengths['h0c'] * axes_lengths['w0c']
            next_conf_c01_ = next_conf_c01.reshape(bs, axes_lengths['h0c'], axes_lengths['w0c'])  # [B,H0,W0]
            next_conf_c01_ = next_conf_c01_.reshape(bs, axes_lengths['h0c'] // window_size, window_size, axes_lengths['w0c'] // window_size, window_size)
            next_conf_c01_ = rearrange(next_conf_c01_, "b h0 t1 w0 t2 -> b (h0 w0) (t1 t2)")
            # argmax_next_conf_c01 = torch.argmax(next_conf_c01_, dim=2)  # [B,HW0//4]
            topk_next_conf_c01 = torch.topk(next_conf_c01_, k=topk, dim=2)[1]  # [B,HW0//4,4]->[B,HW0//4,k]
            conf_idx = torch.arange(0, hw0, device=next_conf_c01.device)[None].repeat(bs, 1)  # [B,HW0]
            conf_idx = conf_idx.reshape(bs, axes_lengths['h0c'], axes_lengths['w0c']).reshape(bs, axes_lengths['h0c'] // window_size, window_size,
                                                                                              axes_lengths['w0c'] // window_size, window_size)
            conf_idx = rearrange(conf_idx, "b h0 t1 w0 t2 -> b (h0 w0) (t1 t2)")
            conf_topk_idx = torch.gather(conf_idx, index=topk_next_conf_c01, dim=2).reshape(bs, -1)
            mask = torch.zeros_like(next_conf_c01).bool()
            mask = torch.scatter(mask, index=conf_topk_idx, dim=1, value=True)
            mask[next_conf_c01 <= test_thr] = False
        elif self.method == 'softargmax_nms':
            window_size = self.config['window_size']
            stride = self.config.get('stride', 1)
            temperature = self.config.get('temperature', 1.0)
            assert stride == 1 or stride == window_size
            padding = window_size // 2 if stride == 1 else 0
            if self.detector is None:
                self.detector = kornia.geometry.ConvSoftArgmax2d((window_size, window_size), (stride, stride), (padding, padding),
                                                                 temperature=temperature, normalized_coordinates=False, output_value=True)
            next_conf_c01_ = next_conf_c01.reshape(bs, axes_lengths['h0c'], axes_lengths['w0c']).unsqueeze(1)  # [B,1,H0,W0]
            softargmax_coords, softargmax_scores = self.detector(next_conf_c01_)
            softargmax_coords = softargmax_coords[:, 0].round().long()  # [B,2,H0,W0]
            softargmax_coords = softargmax_coords[:, 0] * axes_lengths['w0c'] + softargmax_coords[:, 1]
            softargmax_coords = softargmax_coords.reshape(bs, -1)  # [B,HW0]
            softargmax_scores = softargmax_scores[:, 0]  # [B,H0,W0]
            mask = torch.zeros_like(next_conf_c01).bool()
            mask = torch.scatter(mask, index=softargmax_coords, dim=1, value=True)
            mask[next_conf_c01 <= test_thr] = False
        elif self.method == 'maxpool_nms':
            window_size = self.config['window_size']
            stride = self.config.get('stride', 1)
            next_conf_c01_ = next_conf_c01.reshape(bs, axes_lengths['h0c'], axes_lengths['w0c'])  # [B,H0,W0]
            _, ixs_1d = F.max_pool2d(next_conf_c01_, kernel_size=window_size, stride=stride, padding=window_size // 2,
                                     return_indices=True)  # [B,H0,W0]
            coords = torch.arange(axes_lengths['h0c'] * axes_lengths['w0c'],
                                  device=next_conf_c01.device).reshape(1, axes_lengths['h0c'], axes_lengths['w0c'])
            mask = (ixs_1d == coords)
            mask = mask.reshape(bs, -1)
            mask[next_conf_c01 <= test_thr] = False
        elif self.method == 'd2d':
            S_d2d = data['S_d2d']  # [B,hw,1]
            window_size = self.config['window_size']
            stride = self.config.get('stride', 1)
            next_conf_c01_ = next_conf_c01.reshape(bs, axes_lengths['h0c'], axes_lengths['w0c'])  # [B,H0,W0]
            _, ixs_1d = F.max_pool2d(next_conf_c01_, kernel_size=window_size, stride=stride, padding=window_size // 2,
                                     return_indices=True)  # [B,H0,W0]
            coords = torch.arange(axes_lengths['h0c'] * axes_lengths['w0c'],
                                  device=next_conf_c01.device).reshape(1, axes_lengths['h0c'], axes_lengths['w0c'])
            mask = (ixs_1d == coords)
            mask = mask.reshape(bs, -1)
            num = mask.sum(dim=1)  # [B,]
            mask_d2d = torch.zeros_like(mask).bool()  # [B,HW]
            S_d2d = S_d2d.squeeze(dim=-1)  # [B,hw]
            for i in range(num.shape[0]):
                topk_d2d = torch.topk(S_d2d[i], k=min(S_d2d.shape[1], num[i].item()), largest=True, dim=0)[1]
                topk_d2d_y = topk_d2d // data['d2d_w'] * 4
                topk_d2d_x = topk_d2d % data['d2d_w'] * 4
                topk_d2d_1d = (topk_d2d_y * (data['d2d_w'] * 4) + topk_d2d_x).long()
                mask_d2d[i, topk_d2d_1d] = True
            mask = mask_d2d
            mask[next_conf_c01 <= test_thr] = False
        else:
            raise NotImplementedError

        return mask
