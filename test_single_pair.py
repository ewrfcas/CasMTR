import cv2
import numpy as np
import torch

from configs.default import get_cfg_defaults
from src.model.cascade_model_stage3 import CasMTR as CasMTR_4c
from src.model.cascade_model_stage4 import CasMTR as CasMTR_2c
from src.utils.misc import lower_config, load_im_padding


def get_args():
    import argparse

    parser = argparse.ArgumentParser("test quadtree attention-based feature matching")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--confidence_thresh", type=float, default=0.5)
    parser.add_argument("--imsize", type=int, default=1024, help="min side of the image")
    parser.add_argument("--NMS", action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    config = lower_config(config)
    model_config = config["loftr"]

    if len(model_config['cascade_levels']) == 2:
        model_config['coarse3']['post_config']['method'] = "maxpool_nms" if args.NMS else None
        model_config['coarse3']['post_config']['window_size'] = 5 if args.NMS else None
        matcher = CasMTR_2c(config=model_config)
    else:
        model_config['coarse2']['post_config']['method'] = "maxpool_nms" if args.NMS else None
        model_config['coarse2']['post_config']['window_size'] = 5 if args.NMS else None
        matcher = CasMTR_4c(config=model_config)

    state_dict = torch.load(args.weight_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)

    # load pair
    ori_rgb1, ori_rgb2, rgb1, rgb2, mask1, mask2, sc1, sc2 = load_im_padding(args.query_path, args.ref_path, args.imsize)

    with torch.no_grad():
        batch = {'image0': rgb1, 'image1': rgb2}
        if mask1 is not None:
            batch['mask0_origin'] = mask1
        if mask2 is not None:
            batch['mask1_origin'] = mask2

        matcher.eval()
        matcher.to("cuda")
        matcher(batch)
        batch['mconf'] = batch[f"stage_{model_config['cascade_levels'][-1]}c"]['mconf']

        query_kpts = batch["mkpts0_f"].cpu().numpy()
        ref_kpts = batch["mkpts1_f"].cpu().numpy()
        confidences = batch["mconf"].cpu().numpy()
        del batch

        conf_mask = np.where(confidences > args.confidence_thresh)
        query_kpts = query_kpts[conf_mask]
        ref_kpts = ref_kpts[conf_mask]

    def _np_to_cv2_kpts(np_kpts):
        cv2_kpts = []
        for np_kpt in np_kpts:
            cur_cv2_kpt = cv2.KeyPoint()
            cur_cv2_kpt.pt = tuple(np_kpt)
            cv2_kpts.append(cur_cv2_kpt)
        return cv2_kpts

    query_kpts = resample_kpts(query_kpts, sc1[1], sc1[0])
    ref_kpts = resample_kpts(ref_kpts, sc2[1], sc2[0])
    query_kpts, ref_kpts = _np_to_cv2_kpts(query_kpts), _np_to_cv2_kpts(ref_kpts)

    matched_image = cv2.drawMatches(
        ori_rgb1,
        query_kpts,
        ori_rgb2,
        ref_kpts,
        [
            cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
            for idx in range(len(query_kpts))
        ],
        None,
        flags=2,
    )
    cv2.imwrite("result.jpg", matched_image[:, :, ::-1])


def resample_kpts(kpts: np.ndarray, height_ratio, width_ratio):
    kpts[:, 0] *= width_ratio
    kpts[:, 1] *= height_ratio

    return kpts


if __name__ == "__main__":
    main()
