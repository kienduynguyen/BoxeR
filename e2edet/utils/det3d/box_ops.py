import torch
import numpy as np

from e2edet.utils.det3d.geometry import points_in_convex_polygon_3d_jit


def box_cxcyczlwh_to_xyxyxy(x):
    x_c, y_c, z_c, l, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * l),
        (y_c - 0.5 * w),
        (z_c - 0.5 * h),
        (x_c + 0.5 * l),
        (y_c + 0.5 * w),
        (z_c + 0.5 * h),
    ]

    return torch.stack(b, dim=-1)


def box_vol_wo_angle(boxes):
    vol = (
        (boxes[:, 3] - boxes[:, 0])
        * (boxes[:, 4] - boxes[:, 1])
        * (boxes[:, 5] - boxes[:, 2])
    )

    return vol


def box_intersect_wo_angle(boxes1, boxes2):
    ltb = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    lwh = (rbf - ltb).clamp(min=0)  # [N,M,3]
    inter = lwh[:, :, 0] * lwh[:, :, 1] * lwh[:, :, 2]  # [N,M]

    return inter


def box_iou_wo_angle(boxes1, boxes2):
    vol1 = box_vol_wo_angle(boxes1)
    vol2 = box_vol_wo_angle(boxes2)
    inter = box_intersect_wo_angle(boxes1, boxes2)

    union = vol1[:, None] + vol2 - inter
    iou = inter / union

    return iou, union


def generalized_box3d_iou(boxes1, boxes2):
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()

    iou, union = box_iou_wo_angle(boxes1, boxes2)

    ltb = torch.min(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whl = (rbf - ltb).clamp(min=0)  # [N,M,3]
    vol = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]

    return iou - (vol - union) / vol


def rotate_points_along_z(points, angle):
    if isinstance(points, torch.Tensor):
        is_tensor = True
    elif isinstance(points, np.ndarray):
        is_tensor = False
        points = torch.from_numpy(points).float()
        angle = torch.from_numpy(angle).float()
    else:
        raise ValueError("Only support tensor or ndarray!")

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack([cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones], dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, :3], rot_matrix)
    points_rot = torch.cat([points_rot, points[:, :, 3:]], dim=-1)

    return points_rot if is_tensor else points_rot.numpy()


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """

    template = (
        np.array(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        ).astype(boxes3d.dtype)
        / 2
    )

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.reshape(-1, 8, 3), boxes3d[:, 6]
    ).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, :3]

    return corners3d


def mask_boxes_outside_range(boxes, limit_range, min_num_corners=8):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners: int
    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, [0, 1, 2, 3, 4, 5, -1]]

    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[:3]) & (corners <= limit_range[3:])).all(axis=-1)
    mask = mask.sum(axis=1) >= min_num_corners

    return mask


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    Args:
        val (np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.
    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period)
    """
    if isinstance(val, torch.Tensor):
        is_tensor = True
    elif isinstance(val, np.ndarray):
        is_tensor = False
        val = torch.from_numpy(val).float()
    else:
        raise ValueError("Only support tensor or ndarray!")

    val = val - torch.floor(val / period + offset) * period

    if not ((val >= -offset * period) & (val <= offset * period)).all().item():
        val = torch.clamp(val, min=-offset * period, max=offset * period)

    return val if is_tensor else val.numpy()


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])

    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.
    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)

    rot_mat_T = np.stack([rot_cos, rot_sin, -rot_sin, rot_cos]).reshape([2, 2, -1])

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_3d(points, angles):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    rot_mat_T = np.stack(
        [[rot_cos, rot_sin, zeros], [-rot_sin, rot_cos, zeros], [zeros, zeros, ones],]
    )

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    Returns:
        [type]: [description]
    """
    corners = corners_nd(dims, origin=origin)

    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])

    return corners


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """convert kitti locations, dimensions and angles to corners
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d(corners, angles)
    corners += centers.reshape([-1, 1, 3])

    return corners


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.
    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    """Check points in rotated bbox and return indicces.
    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int): Indicate which axis is height.
        origin (tuple[int]): Indicate the position of box center.
    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)

    return indices
