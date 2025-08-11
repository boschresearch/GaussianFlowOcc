import bisect
import os.path as osp

from torch.utils.data import DataLoader
import torch.distributed as dist
from mmcv.runner import DistEvalHook
from mmdet.core.evaluation.eval_hooks import _calc_dynamic_intervals
from torch.nn.modules.batchnorm import _BatchNorm

class CustomDistEvalHook(DistEvalHook):
    """Distributed evaluation hook that will only evaluate on rank 0."""

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        if runner.rank == 0:
            from mmdet3d.apis import custom_single_gpu_test
            results = custom_single_gpu_test(runner.model, self.dataloader)
            self.latest_results = results
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            # the key_score may be `None` so it needs to skip the action to save
            # the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
                
        dist.barrier()