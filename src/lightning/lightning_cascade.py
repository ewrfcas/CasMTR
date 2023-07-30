import collections
import copy
import math
import os
import pprint
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

from src.losses.cascade_loss import CascadeLoss
from src.model.cascade_model_stage3 import CasMTR
from src.model.cascade_model_stage4 import CasMTR as CasMTR4
from src.model.functions.supervision import compute_supervision_coarse, compute_supervision_fine
from src.optimizers import build_optimizer, build_scheduler
from src.utils.comm import gather, all_gather
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.misc import lower_config, flattenList
from src.utils.plotting import make_matching_figures
from src.utils.profiler import PassThroughProfiler

total_times = []
total_times_ransac = []


def get_lr_schedule_with_stepwise_cosine(optimizer, total_steps, min_lr, previous_steps=0, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        lr_weight = min_lr + (1. - min_lr) * 0.5 * (1. + math.cos(math.pi * current_step / (total_steps - previous_steps)))
        return lr_weight

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class PLCascadeMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, reset_lr=False):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.stage = self.loftr_cfg['training_stage']
        self.reset_lr = reset_lr

        # Matcher: LoFTR
        if len(self.loftr_cfg['cascade_levels']) == 2:
            self.matcher = CasMTR4(config=_config['loftr'])
        else:
            self.matcher = CasMTR(config=_config['loftr'])
        self.loss = CascadeLoss(_config)
        self.loss_dict = collections.defaultdict(float)
        self.first_train = True
        self.infer_times = []

        self.ema = getattr(self.config.TRAINER, 'EMA', False)
        self.test_ema = getattr(self.config.TRAINER, 'TEST_EMA', False)
        if self.test_ema:
            self.matcher_ema = copy.deepcopy(self.matcher)
        else:
            self.matcher_ema = None
        self.restore_step = 0

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items['lr'] = min(self.lr_schedulers().get_lr())
        if self.trainer.accelerator_connector.precision == 16:
            items['scale'] = self.trainer.accelerator_connector.precision_plugin.scaler.get_scale()
        for k in self.loss_dict:
            if k != 'loss':
                items[k] = self.loss_dict[k]
        return items

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        import copy
        new_config = copy.deepcopy(self.config)
        new_config.TRAINER.SCHEDULER = 'MultiStepLR'
        scheduler = build_scheduler(new_config, optimizer)
        return [optimizer], [scheduler]

    # def on_train_epoch_start(self) -> None:
    #     if self.global_rank == 0:
    #         self.trainer.save_checkpoint("ckpt_modify/example_restore.ckpt")

    def on_train_start(self) -> None:
        self.restore_step = copy.deepcopy(self.global_step)
        if self.global_rank == 0:
            os.makedirs(self.trainer.logger.log_dir, exist_ok=True)
            self.config.dump(stream=open(Path(self.trainer.logger.log_dir) / 'config.yaml', 'w'))
        shutil.copy(self.config.main_cfg_path, Path(self.trainer.logger.log_dir) / 'main_cfg.py')
        shutil.copy(self.config.data_cfg_path, Path(self.trainer.logger.log_dir) / 'data_cfg.py')
        if self.ema and self.matcher_ema is None:
            self.matcher_ema = copy.deepcopy(self.matcher)

        optimizer = self.optimizers()
        if self.reset_lr and self.stage > 1:
            # reset scheduler
            if self.config.TRAINER.SCHEDULER == 'stepwise_cosine':
                max_steps = self.config.TRAINER.STEPS_RANGE[-1]
                scheduler = {"scheduler": get_lr_schedule_with_stepwise_cosine(optimizer, total_steps=max_steps, min_lr=self.config.TRAINER.MIN_LR,
                                                                               previous_steps=self.global_step),
                             "interval": "step", "name": None, "frequency": 1, "reduce_on_plateau": False, "monitor": None, "strict": True, "opt_idx": None}
                self.trainer.lr_schedulers[0] = scheduler
                scheduler = self.lr_schedulers()
            else:
                scheduler = self.lr_schedulers()
                new_scheduler = build_scheduler(self.config, optimizer)['scheduler']
                scheduler.milestones = new_scheduler.milestones
            scheduler.base_lrs = []
            scheduler._last_lr = []
            # reset learning rate for new stages; should *0.5 to learning rate
            decay = 1.0
            milestones = self.config.TRAINER.MSLR_MILESTONES
            gamma = self.config.TRAINER.MSLR_GAMMA
            for m in milestones:
                if self.current_epoch >= m:
                    decay *= gamma
            decay_initial = decay if self.config.TRAINER.SCHEDULER == 'stepwise_cosine' else 1.0
            for pg in optimizer.param_groups:
                pg['initial_lr'] = self.config.TRAINER.TRUE_LR * decay_initial
                if pg['vit_param']:
                    pg['initial_lr'] *= self.config.TRAINER.VIT_LR_SCALE
                pg['lr'] = self.config.TRAINER.TRUE_LR * decay
                if pg['vit_param']:
                    pg['lr'] *= self.config.TRAINER.VIT_LR_SCALE

                scheduler.base_lrs.append(pg['initial_lr'])
                scheduler._last_lr.append(pg['lr'])

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        warmup_step_stages = self.config.TRAINER.WARMUP_STEP_STAGES
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                for pg in optimizer.param_groups:
                    base_lr = self.config.TRAINER.WARMUP_RATIO * pg['initial_lr']
                    lr = base_lr + (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * abs(pg['initial_lr'] - base_lr)
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')
        elif self.stage > 1 and self.trainer.global_step < (self.restore_step + warmup_step_stages):
            for pg in optimizer.param_groups:
                if '8c' not in pg['layer_name'] and 'backbone' not in pg['layer_name']:
                    init_lr = pg['initial_lr'] / 2
                    base_lr = self.config.TRAINER.WARMUP_RATIO_STAGES * init_lr
                    lr = base_lr + ((self.trainer.global_step - self.restore_step) / warmup_step_stages) * abs(init_lr - base_lr)
                    pg['lr'] = lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        # if self.global_rank == 0:
        #     self.trainer.save_checkpoint("ckpt_modify/example_restore_opt.ckpt")
        #     raise RuntimeError

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("LoFTR"):
            if not self.training and self.ema:
                self.matcher_ema.forward(batch)
            else:
                self.matcher.forward(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)

        if self.training:
            with self.profiler.profile("Compute losses"):
                self.loss.forward(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        for k in batch['loss_scalars']:
            if k != 'loss':
                self.loss_dict[k.replace('loss_', '')] = batch['loss_scalars'][k].item()

        # logging
        if self.trainer.global_rank == 0 and (self.global_step % self.trainer.log_every_n_steps == 0 or self.first_train):
            self.first_train = False
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            for level in self.config['LOFTR']['CASCADE_LEVELS']:
                if f'stage_{level}c' in batch and 'valid_patch_num' in batch[f'stage_{level}c']:
                    self.logger.experiment.add_scalar(f'train/valid_n_{level}c', batch[f'stage_{level}c']['valid_patch_num'], self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(f'skh_bin_score',
                                                  self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING and self.global_step % (self.trainer.log_every_n_steps * 10) == 0:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)

            # loss scale for fp16 and DEBUG
            if self.trainer.accelerator_connector.precision == 16:
                self.logger.experiment.add_scalar(f'train/loss_scale', self.trainer.accelerator_connector.precision_plugin.scaler.get_scale(),
                                                  self.global_step)

        if torch.isnan(batch['loss'].mean()):
            ss = 'Rank ' + str(self.local_rank) + ' : NaN loss' + ' of step ' + str(self.global_step) + '!!!'
            print(ss)
            with open(f'fp16_debug_rank{self.local_rank}.txt', 'a') as w:
                w.write(ss + '\n')
            torch.save(self.matcher.state_dict(), f'model_debug_rank{self.local_rank}.pth')
            raise RuntimeError

        return {'loss': batch['loss']}

    def training_step_end(self, training_step_outputs):
        # Update G_ema.
        if self.ema:
            min_steps = self.config.TRAINER.STEPS_RANGE[0]
            warmup = self.config.TRAINER.EMA_WARMUP
            if (self.global_step - min_steps) < warmup:
                ema_beta = (self.global_step - min_steps) / warmup * self.config.TRAINER.EMA_BETA
            else:
                ema_beta = self.config.TRAINER.EMA_BETA
            with torch.no_grad():
                for p_ema, p in zip(self.matcher_ema.parameters(), self.matcher.parameters()):
                    if p.requires_grad:
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(self.matcher_ema.buffers(), self.matcher.buffers()):
                    b_ema.copy_(b)

        return training_step_outputs

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/avg_loss_on_epoch', avg_loss, global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        if self.ema and self.matcher_ema is None:
            self.matcher_ema = copy.deepcopy(self.matcher)
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            # 'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("Model Matching"):
            with torch.cuda.amp.autocast():
                if self.ema and self.test_ema:
                    self.matcher_ema(batch)
                else:
                    self.matcher(batch)

        batch['m_bids'] = batch[f"stage_{self.loftr_cfg['cascade_levels'][-1]}c"]['m_bids']
        batch['mconf'] = batch[f"stage_{self.loftr_cfg['cascade_levels'][-1]}c"]['mconf']

        with self.profiler.profile("RANSAC"):
            ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        print('Infer time:', np.mean(self.infer_times))

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
