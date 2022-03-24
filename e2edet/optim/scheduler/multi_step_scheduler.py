from bisect import bisect_right

from e2edet.optim.scheduler import register_scheduler, BaseScheduler


@register_scheduler("multi_step")
class MultiStepScheduler(BaseScheduler):
    def __init__(self, config, optimizer):
        self.use_warmup = config["use_warmup"]
        self.mode = config["mode"]
        self.lr_steps = config["lr_steps"]
        self.lr_ratio = config["lr_ratio"]
        self.warmup_iterations = config.get("warmup_iterations", 0)
        self.warmup_factor = config.get("warmup_factor", 1)
        assert self.warmup_iterations < self.lr_steps[0]
        assert self.mode in ["iter", "epoch"], "Only iter|epoch are accepted!"
        super(MultiStepScheduler, self).__init__(config, optimizer)

    def get_iter_lr(self):
        if self.last_iter <= self.warmup_iterations and self.use_warmup is True:
            alpha = float(self.last_iter) / float(self.warmup_iterations)
            lr_ratio = self.warmup_factor * (1.0 - alpha) + alpha

            return [base_lr * lr_ratio for base_lr in self.base_lrs]

        if self.mode == "iter":
            return [
                base_lr * self.lr_ratio ** bisect_right(self.lr_steps, self.last_iter)
                for base_lr in self.base_lrs
            ]

        return [None for _ in self.base_lrs]

    def get_epoch_lr(self):
        if self.mode == "epoch":
            return [
                base_lr * self.lr_ratio ** bisect_right(self.lr_steps, self.last_epoch)
                for base_lr in self.base_lrs
            ]

        return [None for _ in self.base_lrs]
