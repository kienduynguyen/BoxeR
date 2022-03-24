import warnings
import weakref
from functools import wraps

import torch


class BaseScheduler(object):
    def __init__(self, config, optimizer):
        super(BaseScheduler, self).__init__()

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be an instance of torch.optim.Optimizer")
        self.config = config
        self.optimizer = optimizer
        last_epoch = config.get("last_epoch", -1)
        last_iter = config.get("last_iter", -1)

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0

        if last_epoch == -1:
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )

        if last_epoch == -1 and last_iter != -1:
            raise ValueError(
                "Found last_epoch = -1 but last_iter = {}".format(last_iter)
            )
        elif last_epoch != -1 and last_iter == -1:
            raise ValueError(
                "Found last_epoch = {} but last_iter = -1".format(last_epoch)
            )
        elif last_epoch == -1 and last_iter == -1:
            last_epoch = 0
            last_iter = 0

        self.base_lrs = list(
            map(lambda group: group["initial_lr"], self.optimizer.param_groups)
        )
        self.last_iter = last_iter
        self.last_epoch = last_epoch

        self._step_count = 0
        self._step_epoch_count = 0

        self.step(last_iter)
        self.step_epoch(last_epoch)

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "config")
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        print(
            "Step to the last epoch {}, last iter {}".format(
                self.last_epoch, self.last_iter
            )
        )
        self.step(self.last_iter)
        self.step_epoch(self.last_epoch)

    def get_iter_lr(self):
        return [None for _ in self.base_lrs]

    def get_epoch_lr(self):
        return [None for _ in self.base_lrs]

    def step(self, iter=None):
        if self._step_count == 1:
            if self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of 'lr_scheduler.step()' before 'optimizer.step()'. "
                    "In Pytorch 1.1.0 and later, you should call them in the opposite order: "
                    "'optimizer.step()' before 'lr_scheduler.step()'. Failure to do this "
                    "will result in Pytorch skipping the first value of the learning rate schedule."
                )

        self._step_count += 1
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter

        for param_group, lr in zip(self.optimizer.param_groups, self.get_iter_lr()):
            if lr is not None:
                param_group["lr"] = lr

    def step_epoch(self, epoch=None):
        if self._step_epoch_count == 1:
            if self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of 'lr_scheduler.step()' before 'optimizer.step()'. "
                    "In Pytorch 1.1.0 and later, you should call them in the opposite order: "
                    "'optimizer.step()' before 'lr_scheduler.step()'. Failure to do this "
                    "will result in Pytorch skipping the first value of the learning rate schedule."
                )

        self._step_epoch_count += 1
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_epoch_lr()):
            if lr is not None:
                param_group["lr"] = lr
