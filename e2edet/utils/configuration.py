# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import collections
import json
import os
import warnings
from ast import literal_eval

import torch
from omegaconf import OmegaConf

from e2edet.utils.general import get_root


def load_yaml(f):
    # Convert to absolute path for loading includes
    f = os.path.abspath(f)
    mapping = OmegaConf.load(f)

    if mapping is None:
        mapping = OmegaConf.create()

    includes = mapping.get("includes", [])

    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    include_mapping = OmegaConf.create()

    root_dir = get_root()

    for include in includes:
        original_include_path = include
        include = os.path.join(root_dir, include)
        print("include path:", include)

        if not os.path.exists(include):
            include = os.path.join(os.path.dirname(f), original_include_path)

        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)

    mapping = OmegaConf.merge(include_mapping, mapping)

    return mapping


class Configuration:
    def __init__(self, args):
        self.config = {}
        self.args = args
        self._register_resolvers()

        default_config = self._build_default_config()
        user_config = self._build_user_config(args.config)

        self._default_config = default_config
        self._user_config = user_config
        self.config = OmegaConf.merge(default_config, user_config)

        self.config = self._merge_with_dotlist(self.config, args.opts)
        self._update_specific(self.config, args)

    def _build_default_config(self):
        self.default_config_path = self._get_default_config_path()
        default_config = load_yaml(self.default_config_path)
        return default_config

    def _build_user_config(self, config_path):
        user_config = {}

        # Update user_config with opts if passed
        self.config_path = config_path
        if self.config_path is not None:
            user_config = load_yaml(self.config_path)

        return user_config

    def get_config(self):
        self._register_resolvers()
        return self.config

    def _register_resolvers(self):
        OmegaConf.clear_resolvers()
        # Device count resolver
        device_count = max(1, torch.cuda.device_count())
        OmegaConf.register_new_resolver("device_count", lambda: device_count)

    def _merge_with_dotlist(self, config, opts):
        # TODO: To remove technical debt, a possible solution is to use
        # struct mode to update with dotlist OmegaConf node. Look into this
        # in next iteration
        if opts is None:
            opts = []

        if len(opts) == 0:
            return config

        # Support equal e.g. model=visual_bert for better future hydra support
        has_equal = opts[0].find("=") != -1

        if has_equal:
            opt_values = [opt.split("=") for opt in opts]
        else:
            assert len(opts) % 2 == 0, "Number of opts should be multiple of 2"
            opt_values = zip(opts[0::2], opts[1::2])

        for opt, value in opt_values:
            splits = opt.split(".")
            current = config
            for idx, field in enumerate(splits):
                array_index = -1
                if field.find("[") != -1 and field.find("]") != -1:
                    stripped_field = field[: field.find("[")]
                    array_index = int(field[field.find("[") + 1 : field.find("]")])
                else:
                    stripped_field = field
                if stripped_field not in current:
                    raise AttributeError(
                        "While updating configuration"
                        " option {} is missing from"
                        " configuration at field {}".format(opt, stripped_field)
                    )
                if isinstance(current[stripped_field], collections.abc.Mapping):
                    current = current[stripped_field]
                elif (
                    isinstance(current[stripped_field], collections.abc.Sequence)
                    and array_index != -1
                ):
                    current_value = current[stripped_field][array_index]

                    # Case where array element to be updated is last element
                    if not isinstance(
                        current_value,
                        (collections.abc.Mapping, collections.abc.Sequence),
                    ):
                        print("Overriding option {} to {}".format(opt, value))
                        current[stripped_field][array_index] = self._decode_value(value)
                    else:
                        # Otherwise move on down the chain
                        current = current_value
                else:
                    if idx == len(splits) - 1:
                        print("Overriding option {} to {}".format(opt, value))
                        current[stripped_field] = self._decode_value(value)
                    else:
                        raise AttributeError(
                            "While updating configuration",
                            "option {} is not present "
                            "after field {}".format(opt, stripped_field),
                        )

        return config

    def _decode_value(self, value):
        # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L400
        if not isinstance(value, str):
            return value

        if value == "None":
            value = None

        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def freeze(self):
        OmegaConf.set_struct(self.config, True)

    def defrost(self):
        OmegaConf.set_struct(self.config, False)

    def pretty_print(self, writer=None):
        if not self.config.training.log_detailed_config:
            return

        if writer is not None:
            self.writer = writer
        self.writer.write("=====  Training Parameters    =====", "info")
        self.writer.write(self._convert_node_to_json(self.config.training), "info")
        self.writer.write(self._convert_node_to_json(self.config.distributed), "info")

        self.writer.write("======  Dataset Attributes  ======", "info")
        task = self.config.task

        if task in self.config.dataset_config:
            self.writer.write("======== {} =======".format(task), "info")
            dataset_config = self.config.dataset_config[task]
            self.writer.write(self._convert_node_to_json(dataset_config), "info")
        else:
            self.writer.write(
                "No dataset named '{}' in config. Skipping".format(task),
                "warning",
            )

        self.writer.write("======  Optimizer Attributes  ======", "info")
        self.writer.write(self._convert_node_to_json(self.config.optimizer), "info")

        self.writer.write("======  LR_Scheduler Attributes  ======", "info")
        self.writer.write(self._convert_node_to_json(self.config.scheduler), "info")

        if self.config.model not in self.config.model_config:
            raise ValueError(
                "{} not present in model attributes".format(self.config.model)
            )

        self.writer.write(
            "======  Model ({}) Attributes  ======".format(self.config.model), "info"
        )
        self.writer.write(
            self._convert_node_to_json(self.config.model_config[self.config.model]),
            "info",
        )

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def _get_default_config_path(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(directory, "..", "config", "default.yaml")

    def _update_specific(self, config, args):
        if not torch.cuda.is_available() and "cuda" in config.training.device:
            warnings.warn(
                "Device specified is 'cuda' but cuda is not present. Switching to CPU version"
            )
            config.training.device = "cpu"

        # update task and model to config
        config.task = args.task
        config.model = args.model

        return config
