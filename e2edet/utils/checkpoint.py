# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import glob
import warnings

import torch
from omegaconf import OmegaConf

from e2edet.utils.distributed import is_master


def load_pretraind_state_dict(state_dict, trainer, strict=False):
    model_dict = {}

    # remove 'module.' in module keys of state_dict
    for name in state_dict["model"]:
        processed_name = name

        if trainer.parallel is False and name.startswith("module."):
            processed_name = processed_name.replace("module.", "", 1)
        elif trainer.parallel is True and not name.startswith("module."):
            processed_name = "module." + processed_name

        model_dict[processed_name] = state_dict["model"][name]

    print(trainer.model.load_state_dict(model_dict, strict=strict))

    if "optimizer" in state_dict:
        trainer.optimizer.load_state_dict(state_dict["optimizer"])
    else:
        warnings.warn(
            "'optimizer' key is not present in the "
            "checkpoint asked to be loaded. Skipping."
        )

    if "update" in state_dict:
        trainer.current_update = state_dict["update"]
        trainer.current_epoch = state_dict["epoch"]
    else:
        warnings.warn(
            "'update' and 'epoch' key is not present in the "
            "checkpoint asked to be loaded. Skipping."
        )

    if "lr_scheduler" in state_dict:
        trainer.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    else:
        warnings.warn(
            "'lr_scheduler' key is not present in the "
            "checkpoint asked to be loaded. Skipping."
        )

    if "grad_scaler" in state_dict and trainer.grad_scaler is not None:
        trainer.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class Checkpoint:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = self.trainer.config
        self.save_dir = self.config.training.save_dir
        self.num_checkpoint = self.config.training.num_checkpoint
        self.model_name = self.config.model

        self.ckpt_foldername = self.save_dir
        self.device = trainer.device

        self.ckpt_prefix = ""

        if hasattr(self.trainer.model, "get_ckpt_name"):
            self.ckpt_prefix = self.trainer.model.get_ckpt_name() + "_"

        self.pth_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + self.model_name + "_final.pth"
        )

        self.models_foldername = os.path.join(self.ckpt_foldername, "models")
        if is_master():
            if not os.path.exists(self.models_foldername):
                os.makedirs(self.models_foldername, exist_ok=True)

        self.save_config()

    def _process_config(self):
        save_config = OmegaConf.create(OmegaConf.to_yaml(self.config, resolve=True))
        save_config.distributed.init_method = None
        save_config.distributed.rank = 0
        save_config.distributed.port = -1
        save_config.distributed.backend = "nccl"
        save_config.distributed.world_size = 1
        save_config.distributed.no_spawn = False

        return save_config

    def save_config(self):
        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        save_config = self._process_config()
        
        with open(cfg_file, "w") as f:
            f.write(OmegaConf.to_yaml(save_config, resolve=True))

    def _extract_iter(self, path):
        return int(path.split("_")[-1].split(".")[0])

    def load_state_dict(self):
        tp = self.config.training

        if tp.resume:
            if tp.resume_file is not None:
                self.trainer.writer.write(
                    "Loading weights from {}".format(tp.resume_file)
                )

                if os.path.exists(tp.resume_file):
                    self._load(tp.resume_file)
                    return True
                else:
                    raise RuntimeError("{} doesn't exist".format(tp.resume_file))
            else:
                self.trainer.writer.write("Loading weights the last checkpoint")
                ckpt_file_paths = sorted(
                    glob.glob(os.path.join(self.models_foldername, "model_*.ckpt")),
                    key=self._extract_iter,
                )

                if len(ckpt_file_paths) > 0:
                    ckpt_filepath = ckpt_file_paths[-1]
                    self._load(ckpt_filepath)
                    return True
                else:
                    warnings.warn("No checkpoint found!")

        return False

    def _load(self, file):
        self.trainer.writer.write("Loading checkpoint")

        ckpt = self._torch_load(file)
        if "model" in ckpt:
            state_dict = ckpt
        else:
            state_dict = {"model": ckpt}

        load_pretraind_state_dict(state_dict, self.trainer)
        self.trainer.writer.write("Checkpoint loaded")

    def _torch_load(self, file):
        if "cuda" in str(self.device):
            return torch.load(file, map_location=self.device)
        else:
            return torch.load(file, map_location=lambda storage, loc: storage)

    def save(self, update):
        # Only save in main process
        if not is_master():
            return

        ckpt_filepath = os.path.join(self.models_foldername, "model_%d.ckpt" % update)

        if self.trainer.parallel:
            model = self.trainer.model.module
        else:
            model = self.trainer.model

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "epoch": self.trainer.current_epoch,
            "update": self.trainer.current_update,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        if self.trainer.grad_scaler is not None:
            ckpt["grad_scaler"] = self.trainer.grad_scaler.state_dict()
        torch.save(ckpt, ckpt_filepath)

        ckpt_file_paths = sorted(
            glob.glob(os.path.join(self.models_foldername, "model_*.ckpt")),
            key=self._extract_iter,
            reverse=True,
        )
        while len(ckpt_file_paths) > self.num_checkpoint:
            file_path = ckpt_file_paths.pop()
            os.remove(file_path)

    def finalize(self):
        if is_master():
            torch.save(self.trainer.model.state_dict(), self.pth_filepath)
