# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from e2edet.dataset import get_task_list
from e2edet.model import get_arch_list

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def set_seed(seed):
    if seed:
        if seed == -1:
            # From detectron2
            seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
            )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    return seed


def _configure_libraries():
    # Disable opencl in opencv since its interaction with cuda often has negative effects
    # This envvar is supported after OpenCV 3.4.0
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        import cv2

        if int(cv2.__version__.split(".")[0]) >= 3:
            cv2.ocl.setUseOpenCL(False)
    except ModuleNotFoundError:
        # Other types of ImportError, if happened, should not be ignored.
        # Because a failed opencv import could mess up address space
        # https://github.com/skvark/opencv-python/issues/381
        pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    import yaml

    assert get_version(yaml) >= (5, 1), "Requires pyyaml>=5.1"
    assert get_version(torch) >= (1, 4), "Requires torch>=1.4"


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Core Arguments")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=get_task_list(),
        help="The working task.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_arch_list(),
        help="The architecture for training and testing.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    return parser
