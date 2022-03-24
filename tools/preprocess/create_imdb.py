import argparse
import os
import json
import re
import sys

sys.path.insert(0, "/home/dknguyen/3D-ObjectDect")
import math
import matplotlib
import matplotlib.pyplot as plt
import traceback
from collections import defaultdict, Counter
from typing import List, Dict, Mapping
from threading import Thread, Lock
from queue import Queue

import torch
import numpy as np

from detector.utils.kitti.general import generate_3d_corners, read_calib_file
from detector.utils.kitti.transform3d import corners3d_to_img_boxes


def get_img_path_from_id(img_id, include_prev=False, include_camera_3=False) -> dict:
    item = {}

    item["img_path"] = os.path.join("image_2", "{}.png".format(img_id))
    item["img_prev_path"] = None

    if include_prev:
        item["img_prev_path"] = [
            os.path.join("prev_2", "{}_01.png".format(img_id)),
            os.path.join("prev_2", "{}_02.png".format(img_id)),
            os.path.join("prev_2", "{}_03.png".format(img_id)),
        ]

    if include_camera_3:
        raise NotImplementedError

    return item


def get_calib_path_from_id(img_id) -> str:
    return os.path.join("calib", "{}.txt".format(img_id))


def get_label_path_from_id(img_id) -> str:
    return os.path.join("label_2", "{}.txt".format(img_id))


def get_lidar_path_from_id(img_id) -> str:
    return os.path.join("velodyne", "{}.bin".format(img_id))


def get_obj_level(box2d, truncation, occlusion) -> int:
    height = box2d[3] - box2d[1] + 1

    if height >= 40 and truncation <= 0.15 and occlusion == 0:
        return 1  # Easy
    elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
        return 2  # Moderate
    elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
        return 3  # Hard
    else:
        return 4  # Unknown


def read_img_ids(id_file) -> List[str]:
    with open(id_file) as f:
        lines = f.readlines()

    img_ids = []
    for img_id in lines:
        img_ids.append(img_id.lower().strip())

    return img_ids


def read_label_from_file(label_file, P2) -> Dict[str, List]:
    gts = defaultdict(list)

    with open(label_file) as f:
        lines = f.readlines()

    for line in lines:
        label = line.strip().split(" ")
        assert label[0] != "", "Found empty object in label file: {}".format(label_file)

        obj_cls = (label[0]).strip().lower()
        truncation = float(label[1])
        occlusion = int(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        alpha = float(label[3])

        # left, top, right, bottom
        box2d = [float(label[4]), float(label[5]), float(label[6]), float(label[7])]

        h3d = float(label[8])
        w3d = float(label[9])
        l3d = float(label[10])

        center3d = [float(label[11]), float(label[12]) - (h3d / 2), float(label[13])]
        rotY = float(label[14])
        score = float(label[15]) if len(label) == 16 else -1.0

        level = get_obj_level(box2d, truncation, occlusion)

        corners3d = generate_3d_corners(torch.tensor(center3d), w3d, h3d, l3d, rotY)
        _, _, depth_rect = corners3d_to_img_boxes(corners3d.unsqueeze(0), P2)

        ign = False
        if (depth_rect < 0).any():
            ign = True

        item = {
            "class": obj_cls,
            "truncation": truncation,
            "occlusion": occlusion,
            "alpha": alpha,
            "box2d": box2d,
            "size3d": [w3d, h3d, l3d],
            "center3d": center3d,
            "rotY": rotY,
            "score": score,
            "level": level,
            "ign": ign,
        }

        if item["class"] != "dontcare":
            gts["objects"].append(item)
        else:
            gts["igns"].append(item)

    return gts


def generate_imdb_item(img_id, data_root_dir, data_split, include_prev=True) -> Dict:
    if data_split != "test":
        data_split_dir = "training"

    item = {"image_id": int(img_id)}

    img_dict = get_img_path_from_id(img_id, include_prev=include_prev)
    item.update(img_dict)

    item["calib_path"] = get_calib_path_from_id(img_id)
    item["lidar_path"] = get_lidar_path_from_id(img_id)

    if data_split != "test":
        calib = read_calib_file(
            os.path.join(data_root_dir, data_split_dir, item["calib_path"])
        )
        label_path = get_label_path_from_id(img_id)
        label_file = os.path.join(data_root_dir, data_split_dir, label_path)
        gts = read_label_from_file(label_file, calib["P2"])
        item.update(gts)

    return item


def generate_imdb(
    num_workers, data_root_dir, data_split, id_file, include_prev=True
) -> List[Dict]:
    imdb = []
    idx_queue = Queue()
    lock = Lock()

    if os.path.isabs(id_file):
        img_ids = read_img_ids(id_file)
    else:
        img_ids = read_img_ids(os.path.join(data_root_dir, id_file))

    for idx, img_id in enumerate(img_ids):
        idx_queue.put((idx, img_id))

    def _worker():
        while True:
            idx, img_id = idx_queue.get()
            if img_id is None:
                break

            try:
                item = generate_imdb_item(
                    img_id, data_root_dir, data_split, include_prev=include_prev
                )
            except Exception as e:
                exc_info = sys.exc_info()
                exc_type = exc_info[0]
                exc_msg = "".join(traceback.format_exception(*exc_info))
                print("{}: {}".format(exc_type, exc_msg))
                break

            with lock:
                if idx % 100 == 0:
                    print("Processing {} / {}".format(idx, len(img_ids)))
                imdb.append(item)
            idx_queue.task_done()

    print("Spawning threads...")
    for _ in range(num_workers):
        thread = Thread(target=_worker)
        thread.daemon = True
        thread.start()
    idx_queue.join()

    print("Terminating threads...")
    for _ in range(2 * num_workers):
        idx_queue.put((None, None))

    assert len(imdb) == len(
        img_ids
    ), "Number of items doesn't match with number of images"

    return imdb


def generate_class_vocab(imdb) -> Mapping[str, int]:
    obj_vocab = Counter()
    for item in imdb:
        obj_vocab.update([obj["class"] for obj in item["objects"]])

    return obj_vocab


def draw_hist_figure(
    hist_data, title, xticks, xlabel, ylabel, bins, filename, figsize=(12, 8)
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.hist(hist_data, bins=bins)
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.savefig(filename, bbox_inches="tight")


def draw_histogram(img_stats, obj_stats, save_dir="data/kitti/stats") -> None:
    img_stats = np.array(img_stats)
    draw_hist_figure(
        img_stats,
        title="histogram of dataset",
        xticks=list(range(max(img_stats) + 2)),
        xlabel="# object",
        ylabel="# image",
        bins=list(range(max(img_stats) + 2)),
        filename=os.path.join(save_dir, "num_obect_dataset.png"),
    )

    for obj_cls in obj_stats.keys():
        draw_hist_figure(
            obj_stats[obj_cls]["num_object"],
            title="histogram of {}".format(obj_cls),
            xticks=list(range(max(obj_stats[obj_cls]["num_object"]) + 2)),
            xlabel="# object",
            ylabel="# image",
            bins=list(range(max(obj_stats[obj_cls]["num_object"]) + 2)),
            filename=os.path.join(save_dir, "num_obect_in_{}.png".format(obj_cls)),
        )

    for obj_cls in obj_stats.keys():
        draw_hist_figure(
            obj_stats[obj_cls]["area_object"],
            title="histogram of {} area".format(obj_cls),
            xticks=np.linspace(
                math.floor(min(obj_stats[obj_cls]["area_object"])),
                math.ceil(max(obj_stats[obj_cls]["area_object"])),
                num=20,
            ).tolist(),
            xlabel="area",
            ylabel="# object",
            bins=np.linspace(
                math.floor(min(obj_stats[obj_cls]["area_object"])),
                math.ceil(max(obj_stats[obj_cls]["area_object"])),
                num=20,
            ).tolist(),
            filename=os.path.join(save_dir, "area_in_{}.png".format(obj_cls)),
        )

    for obj_cls in obj_stats.keys():
        draw_hist_figure(
            obj_stats[obj_cls]["volume_object"],
            title="histogram of {} volume".format(obj_cls),
            xticks=np.linspace(
                math.floor(min(obj_stats[obj_cls]["volume_object"])),
                math.ceil(max(obj_stats[obj_cls]["volume_object"])),
                num=20,
            ).tolist(),
            xlabel="volume",
            ylabel="# object",
            bins=np.linspace(
                math.floor(min(obj_stats[obj_cls]["volume_object"])),
                math.ceil(max(obj_stats[obj_cls]["volume_object"])),
                num=20,
            ).tolist(),
            filename=os.path.join(save_dir, "volume_in_{}.png".format(obj_cls)),
        )


def compute_stats(imdb, obj_vocab) -> None:
    # No. object per class (min - max)
    # No. object per image (min - max)
    # Distribution of bbox2d area per class
    # Distribution of bbox3d volume per class
    # Distribution of number of objects across dataset and per class
    obj_stats = {}
    img_stats = []

    for obj_cls in obj_vocab:
        obj_stats[obj_cls] = {"num_object": [], "area_object": [], "volume_object": []}

    for item in imdb:
        num_total_obj = 0
        num_obj = defaultdict(int)

        for obj in item["objects"]:
            if obj["class"] != "dontcare":
                num_total_obj += 1
                num_obj[obj["class"]] += 1

                box2d = obj["box2d"]
                obj_area = (box2d[2] - box2d[0]) * (box2d[3] - box2d[1])
                obj_stats[obj["class"]]["area_object"].append(obj_area)

                size3d = obj["size3d"]
                obj_volume = size3d[0] * size3d[1] * size3d[2]
                obj_stats[obj["class"]]["volume_object"].append(obj_volume)

        for obj_cls in obj_vocab:
            obj_stats[obj_cls]["num_object"].append(num_obj[obj_cls])

        img_stats.append(num_total_obj)

    print("# image: {}".format(len(img_stats)))
    print(
        "average # object per image: {} (min: {}, max: {})".format(
            sum(img_stats) / len(img_stats),
            min(img_stats),
            max(img_stats),
        )
    )
    for obj_cls in obj_vocab:
        print(
            "average # {} per image: {} (min: {}, max: {})".format(
                obj_cls,
                sum(obj_stats[obj_cls]["num_object"])
                / len(obj_stats[obj_cls]["num_object"]),
                min(obj_stats[obj_cls]["num_object"]),
                max(obj_stats[obj_cls]["num_object"]),
            )
        )

    draw_histogram(img_stats, obj_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default="./")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--id_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--include_prev", type=bool, default=True)
    parser.add_argument("--imdb_name", type=str)

    args = parser.parse_args()
    params = vars(args)
    print("Parsed input parameters:")
    print(json.dumps(params, indent=2))

    imdb = generate_imdb(
        args.num_workers,
        args.data_root_dir,
        args.data_split,
        args.id_file,
        args.include_prev,
    )

    if not os.path.exists(os.path.join(args.data_root_dir, "imdb")):
        os.makedirs(os.path.join(args.data_root_dir, "imdb"))
    output_file = os.path.join(
        args.data_root_dir, "imdb", "{}.npy".format(args.imdb_name)
    )
    np.save(output_file, imdb, allow_pickle=True)

    if args.data_split != "test":
        obj_vocab = generate_class_vocab(imdb)

        print("Total number of each object in dataset: ", obj_vocab)
        with open(
            os.path.join(
                args.data_root_dir, "vocab/{}_vocab.txt".format(args.imdb_name)
            ),
            "w",
        ) as f:
            f.write("\n".join(list(obj_vocab.keys())))

        compute_stats(imdb, obj_vocab)
