import argparse
import pickle
import os
import sys

sys.path.insert(0, os.path.abspath("./"))

import numpy as np
from pyquaternion import Quaternion

from detector.utils.det3d import box_ops
from detector.utils.det3d.general import read_from_file, read_pc_annotations


def transform_matrix(translation, rotation, inverse=False):
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def veh_pos_to_transform(veh_pos):
    """
    Convert vehicle pose to two transformation matrix
    """
    rotation = veh_pos[:3, :3]
    translation = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        translation, Quaternion(matrix=rotation), inverse=False
    )
    car_from_global = transform_matrix(
        translation, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global


def _fill_infos(root_path, frames, split="train", nsweeps=1):
    # load all train infos
    infos = []
    for frame_name in frames:  # global id
        lidar_path = os.path.join(split, "lidar", frame_name)
        ref_path = os.path.join(split, "annos", frame_name)

        with open(os.path.join(root_path, ref_path), "rb") as f:
            ref_obj = pickle.load(f)
        ref_time = 1e-6 * int(ref_obj["frame_name"].split("_")[-1])

        ref_pose = np.reshape(ref_obj["veh_to_global"], [4, 4])
        _, ref_from_global = veh_pos_to_transform(ref_pose)

        info = {
            "path": lidar_path,
            "anno_path": ref_path,
            "token": frame_name,
            "timestamp": ref_time,
            "sweeps": [],
        }

        sequence_id = int(frame_name.split("_")[1])
        frame_id = int(frame_name.split("_")[3][:-4])  # remove .pkl

        prev_id = frame_id
        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if prev_id <= 0:
                if len(sweeps) == 0:
                    sweep = {
                        "path": lidar_path,
                        "token": frame_name,
                        "transform_matrix": None,
                        "time_lag": 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_id = prev_id - 1

                # global identifier

                curr_name = "seq_{}_frane_{}.pkl".format(sequence_id, prev_id)
                curr_lidar_path = os.path.join(split, "lidar", curr_name)
                curr_label_path = os.path.join(split, "annos", curr_name)

                with open(os.path.join(root_path, curr_label_path), "rb") as f:
                    curr_obj = pickle.load(f)

                curr_pose = np.reshape(curr_obj["veh_to_global"], [4, 4])
                global_from_car, _ = veh_pos_to_transform(curr_pose)

                # tm = reduce(np.dot, [ref_from_global, global_from_car])
                tm = ref_from_global.dot(global_from_car)

                curr_time = 1e-6 * int(curr_obj["frame_name"].split("_")[-1])
                time_lag = ref_time - curr_time

                sweep = {
                    "path": curr_lidar_path,
                    "transform_matrix": tm,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        if split != "test":
            # read boxes
            TYPE_LIST = ("UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST")
            annos = ref_obj["objects"]
            num_points_in_gt = np.array([ann["num_points"] for ann in annos])
            gt_boxes = np.array([ann["box"] for ann in annos]).reshape(-1, 9)
            difficulty = np.array([ann["detection_difficulty_level"] for ann in annos])

            # Return: x, y, z, length, width, height, rotation from positive x axis clockwisely
            # if len(gt_boxes) != 0:
            # transform from Waymo to KITTI coordinate
            # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
            # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely
            # gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
            # gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

            gt_names = np.array([TYPE_LIST[ann["label"]] for ann in annos])
            mask_not_zero = (num_points_in_gt > 0).reshape(-1)

            # filter boxes without lidar points
            info["gt_boxes"] = gt_boxes[mask_not_zero, :].astype(np.float32)
            info["gt_names"] = gt_names[mask_not_zero].astype(str)
            info["difficulty"] = difficulty[mask_not_zero].astype(np.int32)
            info["num_points_in_gt"] = num_points_in_gt[mask_not_zero].astype(np.int64)

        infos.append(info)

    return infos


def _sort_frame(frames):
    indices = []

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id = int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]

    return frames


def _get_available_frames(root, split):
    dir_path = os.path.join(root, split, "lidar")
    available_frames = list(os.listdir(dir_path))

    sorted_frames = _sort_frame(available_frames)
    print(split, " split exist frame num:", len(sorted_frames))

    return sorted_frames


def create_waymo_infos(root_path, split="train", nsweeps=1):
    frames = _get_available_frames(root_path, split)

    waymo_infos = _fill_infos(root_path, frames, split, nsweeps)

    print(f"sample: {len(waymo_infos)}")

    with open(
        os.path.join(
            root_path,
            "infos_" + split + "_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
        ),
        "wb",
    ) as f:
        pickle.dump(waymo_infos, f)


def _get_sensor_data(
    index, dataset_infos, root_path, nsweeps=1, point_features=5, test_mode=True
):
    info = dataset_infos[index]

    sample = {
        "lidar": {
            "type": "lidar",
            "points": None,
            "annotations": None,
            "nsweeps": nsweeps,
        },
        "metadata": {
            "image_prefix": root_path,
            "num_point_features": point_features,
            "token": info["token"],
        },
        "calib": None,
        "cam": {},
        "mode": "val" if test_mode else "train",
    }

    if not os.path.isabs(info["path"]):
        info["path"] = os.path.join(root_path, info["path"])
    points = read_from_file(info, nsweeps=nsweeps)
    sample["lidar"]["points"] = points

    annos = read_pc_annotations(info)
    sample["lidar"]["annotations"] = annos

    return sample


def create_groundtruth_database(
    data_path,
    info_path=None,
    used_classes=("VEHICLE", "CYCLIST", "PEDESTRIAN"),
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    nsweeps=None,
):
    if nsweeps is None:
        test_mode = True
        nsweeps = 1
    else:
        test_mode = False

    root_path = data_path
    if db_path is None:
        db_path = os.path.join(root_path, f"gt_database_{nsweeps}sweeps_withvelo")
    if dbinfo_path is None:
        dbinfo_path = os.path.join(
            root_path, f"dbinfos_train_{nsweeps}sweeps_withvelo.pkl"
        )

    # WAYMO dataset setting
    point_features = 5 if nsweeps == 1 else 6

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    all_db_infos = {}
    group_counter = 0

    with open(info_path, "rb") as f:
        dataset_infos_all = pickle.load(f)

    for index in range(len(dataset_infos_all)):
        image_idx = index

        sensor_data = _get_sensor_data(
            index,
            dataset_infos_all,
            root_path,
            nsweeps=nsweeps,
            point_features=point_features,
            test_mode=test_mode,
        )

        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]

        gt_boxes = annos["boxes"]
        names = annos["names"]

        # waymo dataset contains millions of objects and it is not possible to store
        # all of them into a single folder
        # we randomly sample a few objects for gt augmentation
        # We keep all cyclists as they are rare
        if index % 4 != 0:
            mask = names == "VEHICLE"
            mask = np.logical_not(mask)
            names = names[mask]
            gt_boxes = gt_boxes[mask]

        if index % 2 != 0:
            mask = names == "PEDESTRIAN"
            mask = np.logical_not(mask)
            names = names[mask]
            gt_boxes = gt_boxes[mask]

        num_obj = gt_boxes.shape[0]

        if num_obj == 0:
            continue

        group_dict = {}
        # group_ids = np.full([num_obj], -1, dtype=np.int64)

        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(num_obj, dtype=np.int64)

        difficulty = np.zeros(num_obj, dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        point_indices = box_ops.points_in_rbbox(points, gt_boxes)

        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(db_path, names[i])

                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)

                filepath = os.path.join(db_path, names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points[:, :point_features].tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

                if relative_path:
                    db_dump_path = os.path.join(
                        f"gt_database_{nsweeps}sweeps_withvelo", names[i], filename
                    )
                else:
                    db_dump_path = filepath

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }

                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1

                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset_infos_all))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WayMo dataset preparation")
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--nsweeps", default=1, type=int)

    args = parser.parse_args()

    create_waymo_infos(args.root_path, args.split, args.nsweeps)

    info_path = os.path.join(
        args.root_path,
        "infos_" + args.split + "_{:02d}sweeps_filter_zero_gt.pkl".format(args.nsweeps),
    )

    if args.split == "train":
        create_groundtruth_database(
            args.root_path, info_path=info_path, nsweeps=args.nsweeps
        )
