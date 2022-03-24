import importlib
import os

from e2edet.model.base_model import BaseDetectionModel


ARCH_REGISTRY = {}


__all__ = ["BaseDetectionModel"]


def build_model(config, num_classes):
    model_name = config.model
    model_config = config.model_config[model_name]

    if model_name not in ARCH_REGISTRY:
        raise ValueError("Model architecture ({}) is not found.".format(model_name))
    model = ARCH_REGISTRY[model_name](model_config, num_classes, global_config=config)

    if hasattr(model, "build"):
        model.build()
        model.init_losses_and_metrics()

    return model


def register_model(name):
    def register_model_cls(cls):
        if name in ARCH_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        elif not issubclass(cls, BaseDetectionModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseDetectionModel",
                format(name, cls.__name__),
            )

        ARCH_REGISTRY[name] = cls
        return cls

    return register_model_cls


def get_arch_list():
    return tuple(ARCH_REGISTRY.keys())


models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("e2edet.model." + model_name)
