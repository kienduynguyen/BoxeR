# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import numpy as np
import socket
import subprocess
import warnings
import functools

import torch
from torch import distributed as dist


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def get_rank():
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def broadcast_tensor(tensor, src=0):
    world_size = get_world_size()
    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.broadcast(tensor, src=0)

    return tensor


def broadcast_scalar(scalar, src=0, device="cpu"):
    scalar_tensor = torch.tensor(scalar).to(device)
    scalar_tensor = broadcast_tensor(scalar_tensor, src)
    return scalar_tensor.item()


def reduce_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if dist.get_rank() == 0:
            tensor = tensor.div(world_size)

    return tensor


def gather_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.stack(tensor_list, dim=0)
    return tensor_list


def reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        dist.reduce(values, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def all_reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        dist.all_reduce(values)
        values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = (
            _get_global_gloo_group()
        )  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)

    return output


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def infer_init_method(config):
    if config.distributed.init_method is not None:
        return

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        print("support launch")
        config.distributed.init_method = "env://"
        config.distributed.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed.rank = int(os.environ["RANK"])

    # we can determine the init method automatically for Slurm
    else:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            if config.distributed.port < 0:
                config.distributed.port = 16749
            try:
                nnodes = int(os.environ.get("SLURM_NNODES"))

                # don't need to initialize distributed training on a single gpu
                if nnodes == 1 and torch.cuda.device_count() == 1:
                    return

                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config.distributed.init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config.distributed.port,
                )

                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)

                gpus_per_node = torch.cuda.device_count()
                config.distributed.world_size = nnodes * gpus_per_node
                if ntasks_per_node == 1:
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    config.distributed.rank = node_id * gpus_per_node
                else:
                    assert gpus_per_node == ntasks_per_node
                    config.distributed.no_spawn = True
                    config.distributed.rank = int(os.environ.get("SLURM_PROCID"))
                    config.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def distributed_init(config):
    if config.distributed.world_size == 1:
        raise ValueError("Cannot initialize distributed with distributed_world_size=1")

    if dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
    else:
        print(
            "Distributed Init (Rank {}): {}".format(
                config.distributed.rank, config.distributed.init_method
            ),
            flush=True,
        )
        dist.init_process_group(
            backend=config.distributed.backend,
            init_method=config.distributed.init_method,
            world_size=config.distributed.world_size,
            rank=config.distributed.rank,
        )
        print(
            "Initialized Host {} as Rank {}".format(
                socket.gethostname(), config.distributed.rank
            ),
            flush=True,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        dist.all_reduce(torch.zeros(1).cuda())

        suppress_output(is_master())

    config.distributed.rank = dist.get_rank()
    return config.distributed.rank


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)
