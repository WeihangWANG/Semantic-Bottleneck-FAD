# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler

def build_scheduler(config, optimizer, n_iter_per_epoch):
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    # 两个周期：每个周期 50 epoch
    cycle_epochs = 45
    cycle_steps = int(cycle_epochs * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cycle_steps,               # 关键：一个周期的步数（50 epoch）
        cycle_mul=1.0,                         # 关键：等长周期（替代 t_mul）
        lr_min=config.TRAIN.MIN_LR,          # 1e-6
        warmup_lr_init=config.TRAIN.WARMUP_LR,  # 1e-6
        warmup_t=warmup_steps,               # 10 epoch warmup
        warmup_prefix=True,                  # 关键：warmup 放在最前，不占 cosine 周期
        cycle_limit=2,                       # 关键：2 个周期
        t_in_epochs=False,                   # 你用 step_update，保持 False
    )
    return lr_scheduler

# def build_scheduler(config, model):
#     """
#     Build optimizer, set weight decay of normalization to 0 by default.
#     """
#     skip = {}
#     skip_keywords = {}
#     if hasattr(model, 'no_weight_decay'):
#         skip = model.no_weight_decay()
#     if hasattr(model, 'no_weight_decay_keywords'):
#         skip_keywords = model.no_weight_decay_keywords()
#     parameters = set_weight_decay(model, skip, skip_keywords)
#
#     lr = float(config.TRAIN.BASE_LR)
#     wd = float(getattr(config.TRAIN, "WEIGHT_DECAY", 0.005))
#
#     opt_lower = 'adamw'
#     if opt_lower == 'sgd':
#         optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True,
#                               lr=lr, weight_decay=wd)
#     elif opt_lower == 'adamw':
#         optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
#                                 lr=lr, weight_decay=wd)
#     else:
#         raise ValueError(f"Unsupported optimizer: {opt_lower}")
#
#     return optimizer


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
