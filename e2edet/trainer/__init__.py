import importlib
import os


TRAINER_REGISTRY = {}


def build_trainer(configuration, *rest, **kwargs):
    configuration.freeze()

    config = configuration.get_config()
    trainer = config.training.trainer
    trainer = TRAINER_REGISTRY[trainer](configuration)

    return trainer


def register_trainer(name):
    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))

        TRAINER_REGISTRY[name] = cls
        return cls

    return register_trainer_cls


trainers_dir = os.path.dirname(__file__)
for file in os.listdir(trainers_dir):
    path = os.path.join(trainers_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        trainer_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("e2edet.trainer." + trainer_name)
