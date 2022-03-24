import argparse
import os

import torch
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2


tf.get_logger().setLevel("INFO")


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    Args:
        val (np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.
    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    val = val - np.floor(val / period + offset) * period

    if not ((val >= -offset * period) & (val <= offset * period)).all().item():
        val = np.clip(val, -offset * period, offset * period)

    return val


class WaymoEvaluator(tf.test.TestCase):

    WAYMO_CLASSES = ("UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST")

    def generate_waymo_type_results(self, infos, token_list, class_names, is_gt=False):
        score, overlap_nlz, difficulty = [], [], []
        frame_id, boxes3d, obj_type = [], [], []
        for token in token_list:
            info = infos[token]
            if not is_gt:
                info = {"scores": info["pred_scores"], "labels": info["pred_labels"], "boxes3d": info["pred_boxes3d"]}
                
            info["boxes"] = info["boxes3d"][:, [0, 1, 2, 3, 4, 5, -1]].numpy()
            info["names"] = np.array(
                [self.WAYMO_CLASSES[i] for i in info["labels"].tolist()]
            )

            if is_gt:
                info["difficulty"] = info["difficulty"].numpy()
                info["num_points_in_gt"] = info["num_points_in_gt"].numpy()

                box_mask = np.array(
                    [
                        self.WAYMO_CLASSES[i] in class_names
                        for i in info["labels"].tolist()
                    ],
                    dtype=np.bool_,
                )
                if "num_points_in_gt" in info:
                    zero_difficulty_mask = info["difficulty"] == 0
                    info["difficulty"][
                        (info["num_points_in_gt"] > 5) & zero_difficulty_mask
                    ] = 1
                    info["difficulty"][
                        (info["num_points_in_gt"] <= 5) & zero_difficulty_mask
                    ] = 2
                    nonzero_mask = info["num_points_in_gt"] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    print(
                        "Please provide the num_points_in_gt for evaluating on Waymo Dataset "
                        "(If you create Waymo Infos before 20201126, please re-create the validation infos "
                        "with version 1.2 Waymo dataset to get this attribute). SSS of OpenPCDet"
                    )
                    raise NotImplementedError

                num_boxes = box_mask.sum()
                box_name = info["names"][box_mask]

                difficulty.append(info["difficulty"][box_mask])
                score.append(np.ones(num_boxes))
                boxes3d.append(info["boxes"][box_mask])
            else:
                info["scores"] = info["scores"].numpy()

                num_boxes = info["boxes"].shape[0]
                difficulty.append([0] * num_boxes)
                score.append(info["scores"])
                boxes3d.append(np.array(info["boxes"]))
                box_name = info["names"]

            obj_type += [
                self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)
            ]

            seq_id = int(token.split("_")[1])
            f_id = int(token.split("_")[3][:-4])

            idx = seq_id * 1000 + f_id
            frame_id.append(np.array([idx] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

    def build_config(self):
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """

        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config

    def build_graph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)

            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self.build_config(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=self._pd_overlap_nlz,
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
            )
            return metrics

    def run_eval_ops(
        self,
        sess,
        graph,
        metrics,
        prediction_frame_id,
        prediction_bbox,
        prediction_type,
        prediction_score,
        prediction_overlap_nlz,
        ground_truth_frame_id,
        ground_truth_bbox,
        ground_truth_type,
        ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._pd_overlap_nlz: prediction_overlap_nlz,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            },
        )

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, :2], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def waymo_evaluation(self, infos, class_name, distance_thresh=100):
        print("Start the waymo evaluation...")

        token_list = tuple(infos.keys())

        tf.compat.v1.disable_eager_execution()
        (
            pd_frameid,
            pd_boxes3d,
            pd_type,
            pd_score,
            pd_overlap_nlz,
            _,
        ) = self.generate_waymo_type_results(infos, token_list, class_name, is_gt=False)
        (
            gt_frameid,
            gt_boxes3d,
            gt_type,
            gt_score,
            gt_overlap_nlz,
            gt_difficulty,
        ) = self.generate_waymo_type_results(infos, token_list, class_name, is_gt=True)

        (
            pd_boxes3d,
            pd_frameid,
            pd_type,
            pd_score,
            pd_overlap_nlz,
        ) = self.mask_by_distance(
            distance_thresh,
            pd_boxes3d,
            pd_frameid,
            pd_type,
            pd_score,
            pd_overlap_nlz,
        )
        (
            gt_boxes3d,
            gt_frameid,
            gt_type,
            gt_score,
            gt_difficulty,
        ) = self.mask_by_distance(
            distance_thresh,
            gt_boxes3d,
            gt_frameid,
            gt_type,
            gt_score,
            gt_difficulty,
        )

        print("Number: (pd, %d) VS. (gt, %d)" % (len(pd_boxes3d), len(gt_boxes3d)))
        print(
            "Level 1: %d, Level2: %d)"
            % ((gt_difficulty == 1).sum(), (gt_difficulty == 2).sum())
        )

        if pd_score.max() > 1:
            # assert pd_score.max() <= 1.0, 'Waymo evaluation only supports normalized scores'
            pd_score = 1 / (1 + np.exp(-pd_score))
            print("Warning: Waymo evaluation only supports normalized scores")

        graph = tf.Graph()
        metrics = self.build_graph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self.run_eval_ops(
                sess,
                graph,
                metrics,
                pd_frameid,
                pd_boxes3d,
                pd_type,
                pd_score,
                pd_overlap_nlz,
                gt_frameid,
                gt_boxes3d,
                gt_type,
                gt_difficulty,
            )
            with tf.compat.v1.variable_scope("detection_metrics", reuse=True):
                aps = self.eval_value_ops(sess, graph, metrics)
        return aps


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--root-path", type=str, default=None, help="pickle file")
    args = parser.parse_args()

    infos = torch.load(os.path.join(args.root_path, "results.pth"), map_location="cpu")

    classes = ["VEHICLE", "PEDESTRIAN"]
    if "classes" in infos:
        classes = infos["classes"]

    print("Start to evaluate the waymo format results...")
    eval = WaymoEvaluator()

    waymo_AP = eval.waymo_evaluation(infos, classes)

    print(waymo_AP)


if __name__ == "__main__":
    main()
