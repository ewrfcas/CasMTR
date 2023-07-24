import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    # param_groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and '.vit.' in n],
    #                  'lr': lr * config.TRAINER.VIT_LR_SCALE, 'weight_decay': config.TRAINER.ADAM_DECAY, 'vit_param': True}]
    # param_groups.append({"params": [p for n, p in model.named_parameters() if p.requires_grad and '.vit.' not in n],
    #                      'lr': lr, 'weight_decay': config.TRAINER.ADAM_DECAY, 'vit_param': False})

    param_groups = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            is_vit = '.vit.' in n
            param_groups.append({'params': [p], 'lr': lr * config.TRAINER.VIT_LR_SCALE if is_vit else lr,
                                 'vit_param': is_vit, 'layer_name': n})

    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
