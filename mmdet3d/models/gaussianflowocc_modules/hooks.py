# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from mmcv import runner
from mmcv.runner.hooks import HOOKS, CosineAnnealingLrUpdaterHook, Hook
from mmcv.runner.hooks.lr_updater import annealing_cos

@HOOKS.register_module()
class CustomCosineAnealingLrUpdaterHook(CosineAnnealingLrUpdaterHook):

    def __init__(self,
                 start_at: int = 0,
                 **kwargs) -> None:
        self.start_at = start_at
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch - self.start_at
            max_progress = runner.max_epochs - self.start_at
        else:
            iter_per_epoch = runner.max_iters // runner.max_epochs
            progress = runner.iter - (iter_per_epoch * self.start_at)
            max_progress = runner.max_iters - (iter_per_epoch * self.start_at)

        if runner.epoch < self.start_at:
            return base_lr

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        return annealing_cos(base_lr, target_lr, progress / max_progress)