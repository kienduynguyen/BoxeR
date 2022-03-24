import os
import pickle

import torch
import numba
import numpy as np


def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def shape_mergeable(x, expected_shape):
    mergeable = True

    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable


def mask_points_by_range(points, pc_range):
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] <= pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] <= pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] <= pc_range[5])
    )

    return mask


def read_from_file(info, nsweeps=1):
    path = info["path"]
    with open(path, "rb") as f:
        obj = pickle.load(f)

    points = read_single_waymo(obj)
    times = None

    if nsweeps > 1:
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) == len(
            info["sweeps"]
        ), "nsweeps {} should be equal to the list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in range(nsweeps - 1):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = read_single_waymo_sweep(sweep)

            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    if times is not None:
        points = np.hstack([points, times])

    return points


def read_pc_annotations(info):
    if "gt_boxes" in info:
        annos = {
            "boxes": info["gt_boxes"].astype(np.float32),
            "names": info["gt_names"],
            "difficulty": info["difficulty"].astype(np.int8),
            "num_points_in_gt": info["num_points_in_gt"].astype(np.int64),
        }
        return annos

    return None


def _read_file(path, num_points=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split("/")[:-2], "painted_" + path.split("")[-2])
        painted_path = os.path.join(dir_path, path.split("/")[-1] + ".npy")
        points = np.load(painted_path)
        points = points[
            :, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]  # remove ring_index from features
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_points]

    return points


def _remove_close(points, radius: float):
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius

    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]

    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = _read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = _remove_close(points_sweep, min_distance)

    num_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(num_points)))
        )[:3, :]

    cur_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, cur_times.T


def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)

    return points


def read_single_waymo_sweep(sweep):
    with open(sweep["path"], "rb") as f:
        obj = pickle.load(f)

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T  # 5 x N

    num_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(num_points)))
        )[:3, :]

    cur_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, cur_times.T


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def points_to_voxel_cuda(
    points, voxel_size, coord_range, max_points=35, max_voxel=20000, reverse=True,
):
    """
    Params:
    :points: (N, ndim). points[:, 3] contain xyz points and
        points[:, 3:] contain other information such as reflectivity.
    :voxel_size: (3,) list/tuple or array, xyz, indicate voxel size.
    :coord_range: (6,) list/tuple or array indicates voxel range.

    Return:
    :voxels: (M, max_points, ndim)
    :coords: (M, 3)
    :num_points_per_voxel: (M,)
    """
    grid_shape = (coord_range[3:] - coord_range[:3]) / voxel_size
    grid_shape = torch.round(grid_shape).int()

    points_to_voxel_coord = torch.floor(
        (points[:, :3] - coord_range[:3]) / voxel_size
    ).int()
    assert (
        ((points_to_voxel_coord >= 0) & (points_to_voxel_coord < grid_shape))
        .all()
        .item()
    )
    points_to_voxel_idx = (
        points_to_voxel_coord[:, 0] * grid_shape[1] * grid_shape[2]
        + points_to_voxel_coord[:, 1] * grid_shape[2]
        + points_to_voxel_coord[:, 2]
    )
    sorted_indices = points_to_voxel_idx.argsort()
    points_to_voxel_idx = points_to_voxel_idx[sorted_indices]
    points_to_voxel_coord = points_to_voxel_coord[sorted_indices]
    points = points[sorted_indices]

    unique_voxel_idx, voxel_idx_count = torch.unique(
        points_to_voxel_idx, sorted=True, return_counts=True
    )
    num_voxels = unique_voxel_idx.shape[0]
    num_voxels = min(max_voxel, num_voxels)

    voxels = torch.zeros(
        num_voxels,
        max_points,
        points.shape[-1],
        device=points.device,
        dtype=points.dtype,
    )
    coords = torch.zeros(num_voxels, 3, device=points.device, dtype=torch.int32)
    num_points_per_voxel = torch.zeros(
        num_voxels, device=points.device, dtype=torch.int32
    )

    points_idx = 0
    for i in range(num_voxels):
        num_points = voxel_idx_count[i].item()

        num_points_in_voxel = min(num_points, max_points)
        voxels[i, :num_points_in_voxel] = points[
            points_idx : points_idx + num_points_in_voxel
        ]
        if reverse:
            coords[i] = points_to_voxel_coord[points_idx].flip(0)
        else:
            coords[i] = points_to_voxel_coord[points_idx]
        num_points_per_voxel[i] = num_points_in_voxel
        points_idx += num_points

    return voxels, coords, num_points_per_voxel


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(
    points, voxel_size, coors_range, max_points=35, reverse_index=True, max_voxels=20000
):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype
    )
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    else:
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N,), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices


@numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    coor_to_voxelidx,
    # coors_2d,
    bev_map,
    height_lowers,
    # density_norm_num=16,
    with_reflectivity=False,
    max_voxels=40000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = points.shape[1] - 1
    # ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3,), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]
        ) / height_slice_size
        if incomimg_height_norm > height_norm:
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(
    points,
    voxel_size,
    coors_range,
    with_reflectivity=False,
    density_norm_num=16,
    max_voxels=40000,
):
    """convert kitti points(N, 4) to a bev map. return [C, H, W] map.
    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor.
            `WARNING`: bev_map[-1] is num_points map, NOT density map,
            because calculate density map need more time in cpu rather than gpu.
            if with_reflectivity is True, bev_map[-2] is intensity map.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False
    )
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        coor_to_voxelidx,
        bev_map,
        height_lowers,
        with_reflectivity,
        max_voxels,
    )

    return bev_map


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """Convert boxes_corner to aligned (min-max) boxes.
    Args:
        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.
    Returns:
        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (
                                D[0] - A[0]
                            )
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (
                                D[0] - B[0]
                            )
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    C[0] - A[0]
                                )
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    D[0] - A[0]
                                )
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret
