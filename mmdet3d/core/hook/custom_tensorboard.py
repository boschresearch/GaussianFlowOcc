# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner.hooks.hook import HOOKS


@HOOKS.register_module()
class CustomTensorboardLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True,
                 min_grad_norm=2.):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir
        self.min_grad_norm = min_grad_norm

    def before_run(self, runner) -> None:
        super().before_run(runner)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                'the dependencies to use torch.utils.tensorboard '
                '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

        # Write param grads is available
        if hasattr(runner, 'grad_buffer') and runner.grad_buffer is not None:
            for name, grad in runner.grad_buffer.items():
                if grad > self.min_grad_norm:
                    self.writer.add_scalar(f'grad/{name}', grad, self.get_iter(runner))
            runner.grad_buffer = None

    def after_run(self, runner) -> None:
        self.writer.close()
