import argparse
import math
import os.path
import pprint
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from loguru import logger as loguru_logger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from configs.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_cascade import PLCascadeMatcher
from src.lightning.lightning_cascade_refine import PLCascadeRefineMatcher
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--train_img_size', type=int, default=704, help='training image size')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--training_stage', type=int, default=1, help='training stage, 1:1/8 only, 2:1/4, 3:1/2')
    parser.add_argument(
        '--reset_lr', action='store_true')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--seed', type=int, default=66)
    parser.add_argument(
        '--refine', action='store_true', help='whether to finetune model based on quadtree')
    parser.add_argument(
        '--quadtree_path', type=str, default=None, help='quadtree model path (used for refine)')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    config.TRAINER.SEED = args.seed
    pl.seed_everything(config.TRAINER.SEED + args.training_stage)  # reproducibility
    config.DATASET.MGDPT_IMG_RESIZE = args.train_img_size
    config.LOFTR.TRAIN_SIZE = args.train_img_size
    config.LOFTR.TRAINING_STAGE = args.training_stage
    config.main_cfg_path = args.main_cfg_path
    config.data_cfg_path = args.data_cfg_path
    # make sure training use no post-process
    config.LOFTR.COARSE2.POST_CONFIG.METHOD = None
    config.LOFTR.COARSE3.POST_CONFIG.METHOD = None
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    if config.DATASET.TRAINVAL_DATA_SOURCE == 'ScanNet':
        _scaling = float(np.sqrt(_scaling))
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    loguru_logger.info(f"True LR = {config.TRAINER.TRUE_LR} !")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    if args.refine:
        if args.quadtree_path is not None:
            assert os.path.exists(args.quadtree_path)
            config.LOFTR.QUADTREE_PATH = args.quadtree_path
        model = PLCascadeRefineMatcher(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, reset_lr=args.reset_lr)
    else:
        model = PLCascadeMatcher(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, reset_lr=args.reset_lr)
    loguru_logger.info(f"LoFTR LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"LoFTR DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'check_points'

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=3, mode='max',
                                    save_last=True, dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
