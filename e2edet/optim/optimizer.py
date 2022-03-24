# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from fairscale (https://github.com/facebookresearch/fairscale)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import math
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

import torch

if TYPE_CHECKING:  # pragma: no cover
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class BaseOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: _params_t,
        optim: Type[torch.optim.Optimizer] = torch.optim.SGD,
        **default: Any
    ):
        self.in_super_constructor = True
        super().__init__(params, default)
        self.in_super_constructor = False

        self._params = list(chain(*[group["params"] for group in self.param_groups]))

        self.optim = optim(params, **default)

    def state_dict(self) -> Dict[str, Any]:
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.optim.load_state_dict(state_dict)

        BaseOptimizer._sync_param_groups(state_dict["param_groups"], self.param_groups)
        BaseOptimizer._sync_param_groups(self.param_groups, self.optim.param_groups)

    def step(
        self, closure: Optional[Callable[[], float]] = None, **kwargs: Any
    ) -> Optional[float]:
        BaseOptimizer._sync_param_groups(self.param_groups, self.optim.param_groups)

        if closure is not None:
            loss = self.optim.step(closure=closure, **kwargs)
        else:
            loss = self.optim.step(**kwargs)

        BaseOptimizer._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def clip_grad_norm(self, max_norm):
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self._params, max_norm)
        else:
            return torch.tensor(
                [
                    math.sqrt(
                        sum(
                            p.grad.data.norm().item() ** 2
                            for p in self._params
                            if p.grad is not None
                        )
                    )
                ]
            )

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if not self.in_super_constructor:
            self.optim.add_param_group(param_group)

    @staticmethod
    def _sync_param_groups(
        source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]
    ) -> None:
        """Sync learning rate and other optimizer attributes (needed to support schedulers)."""

        for source_group, destination_group in zip(source, destination):
            # Sync everything but the parameters
            for k in filter(lambda x: x != "params", source_group.keys()):
                destination_group[k] = source_group[k]
