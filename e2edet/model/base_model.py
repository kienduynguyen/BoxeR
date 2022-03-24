# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import collections
import warnings

from torch import nn

from e2edet.criterion import build_loss, build_metric


class BaseDetectionModel(nn.Module):
    """For integration with Pythia's trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.

    """

    def __init__(self, config, num_classes, **kwargs):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self._global_config = kwargs["global_config"]
        self._iter_per_update = self._global_config.training.iter_per_update
        self._logged_warning = {"losses_present": False, "metrics_present": False}

    def _build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError(
            "Build method not implemented in the child model class."
        )

    def build(self):
        self._build()
        self.inference(False)

    def inference(self, mode=True):
        if mode:
            super().train(False)
        self.inferencing = mode
        for module in self.modules():
            if hasattr(module, "inferencing"):
                module.inferencing = mode
            else:
                setattr(module, "inferencing", mode)

    def train(self, mode=True):
        if mode:
            self.inferencing = False
            for module in self.modules():
                if hasattr(module, "inferencing"):
                    module.inferencing = False
                else:
                    setattr(module, "inferencing", False)
        super().train(mode)

    def init_losses_and_metrics(self):
        """Initializes loss and metrics for the model based ``losses`` key
        and ``metrics`` keys. Automatically called by MMF internally after
        building the model.
        """
        loss = self.config.get("loss", None)
        metric = self.config.get("metric", None)

        if loss is None:
            warnings.warn(
                "No losses are defined in model configuration. You are expected "
                "to return loss in your return dict from forward."
            )

        if metric is None:
            warnings.warn(
                "No metrics are defined in model configuration. You are expected "
                "to return metrics in your return dict from forward."
            )

        self.losses = build_loss(loss, self.num_classes, self._iter_per_update)
        self.metrics = build_metric(metric)

        aux_weight_dict = {}
        aux_loss = self.config["aux_loss"]
        num_layers = self.config["transformer"]["params"]["dec_layers"]
        if hasattr(self.losses, "weight_dict"):
            aux_weight_dict.update(
                {k + f"_enc_0": v for k, v in self.losses.weight_dict.items()}
            )

            if aux_loss:
                for i in range(num_layers - 1):
                    aux_weight_dict.update(
                        {k + f"_{i}": v for k, v in self.losses.weight_dict.items()}
                    )

            self.losses.weight_dict.update(aux_weight_dict)

    def forward(self, sample):
        raise NotImplementedError(
            "Forward of the child model class needs to be implemented."
        )

    def __call__(self, sample, target=None):
        model_output = super().__call__(sample, target)

        if target is None or self.inferencing:
            return model_output

        # Make sure theat the output from the model is a Mapping
        assert isinstance(
            model_output, collections.abc.Mapping
        ), "A dict must be returned from the forward of the model."

        if "losses" in model_output:
            if not self._logged_warning["losses_present"]:
                warnings.warn(
                    "'losses' already present in model output. "
                    "No calculation will be done in base model."
                )
                self._logged_warning["losses_present"] = True

            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."
        else:
            losses_stat = {}
            if "num_boxes" in sample:
                model_output["num_boxes"] = sample["num_boxes"]

            loss_dict = self.losses(model_output, target)
            if hasattr(self.losses, "weight_dict"):
                weight_dict = self.losses.weight_dict
                total_loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
                losses_stat.update({f"{k}_unscaled": v for k, v in loss_dict.items()})
                losses_stat.update(
                    {
                        k: v * weight_dict[k]
                        for k, v in loss_dict.items()
                        if k in weight_dict
                    }
                )
            else:
                total_loss = sum(loss_dict[k] for k in loss_dict.keys())
                losses_stat.update({k: v for k, v in loss_dict.items()})
            losses_stat["total_loss"] = total_loss
            model_output["losses"] = total_loss
            model_output["losses_stat"] = losses_stat

        if "metrics" in model_output:
            if not self._logged_warning["metrics_present"]:
                warnings.warn(
                    "'metrics' already present in model output. "
                    "No calculation will be done in base model."
                )
                self._logged_warning["metrics_present"] = True

            assert isinstance(
                model_output["metrics"], collections.abc.Mapping
            ), "'metrics' must be a dict."
        else:
            metrics = {}
            for name, metric in self.metrics.items():
                if name == "accuracy":
                    metrics.update(metric(*self.losses.get_target_classes()))
                else:
                    metrics.update(metric(model_output, target))
            model_output["metrics"] = metrics

        return model_output
