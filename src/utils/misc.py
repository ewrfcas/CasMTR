import contextlib
import os
from itertools import chain
from typing import Union

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms.functional as F
from loguru import _Logger, logger
from pytorch_lightning.utilities import rank_zero_only
from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level,
                    lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """ A temporary fix for pytorch-lighting 1.3.x """
    gpus = str(gpus)
    gpu_ids = []
    
    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']
    
    # setup environment variables
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        logger.warning(f'[Temporary Fix] manually set CUDA_VISIBLE_DEVICES when specifying gpus to use: {visible_devices}')
    else:
        logger.warning('[Temporary Fix] CUDA_VISIBLE_DEVICES already set by user or the main process.')
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    
    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))
            
    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((pad_size, pad_size, inp.shape[2]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def resize_im_(wo, ho, imsize=None, dfactor=1, value_to_scale=max):
    wt, ht = wo, ho
    if imsize is not None and imsize > 0:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])
    scale = (wo / wt, ho / ht)
    return wt, ht, scale


def load_im_padding(im_path0, im_path1, imsize=1024, max_img_size=1344, device="cuda", dfactor=32):
    im0 = cv2.imread(im_path0)[:, :, ::-1]
    origin_im0 = im0.copy()
    ho, wo, _ = im0.shape
    if imsize > 0:
        posible_max = imsize / min(ho, wo) * max(ho, wo)
    else:
        posible_max = -1
    if posible_max > max_img_size:
        wt0, ht0, scale0 = resize_im_(wo, ho, imsize=max_img_size, dfactor=dfactor, value_to_scale=max)
    else:
        wt0, ht0, scale0 = resize_im_(wo, ho, imsize=imsize, dfactor=dfactor, value_to_scale=min)
    im0 = cv2.resize(im0, (wt0, ht0))

    im1 = cv2.imread(im_path1)[:, :, ::-1]
    origin_im1 = im1.copy()
    ho, wo, _ = im1.shape
    wt1, ht1, scale1 = resize_im_(wo, ho, imsize=imsize, dfactor=dfactor, value_to_scale=min)
    im1 = cv2.resize(im1, (wt1, ht1))

    if im0.shape != im1.shape:
        pad_to = max(ht0, wt0)
        im0, mask0 = pad_bottom_right(im0, pad_to, ret_mask=True)
        wt1, ht1, scale1_ = resize_im_(wt1, ht1, imsize=pad_to, dfactor=dfactor, value_to_scale=max)
        scale1 = (scale1[0] * scale1_[0], scale1[1] * scale1_[1])
        im1 = cv2.resize(im1, (wt1, ht1))
        pad_to = max(ht1, wt1)
        im1, mask1 = pad_bottom_right(im1, pad_to, ret_mask=True)
        mask0 = torch.from_numpy(mask0).unsqueeze(0).to(device)
        mask1 = torch.from_numpy(mask1).unsqueeze(0).to(device)
    else:
        mask0, mask1 = None, None
    im0 = F.to_tensor(im0).unsqueeze(0).to(device)
    im1 = F.to_tensor(im1).unsqueeze(0).to(device)

    print(im0.shape, im1.shape)

    return origin_im0, origin_im1, im0, im1, mask0, mask1, scale0, scale1

