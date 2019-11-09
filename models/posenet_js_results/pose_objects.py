import collections
import typing

import numpy as np
from sklearn import preprocessing

from models.openpose.normalize import resize_reshape

KeyPoint = collections.namedtuple("KeyPoint", ["name", "score", "x", "y"])


def get_l2_norm(x_vals, y_vals):
    pose_vector = np.array([x_vals, y_vals]).flatten()
    pose_vector = preprocessing.normalize([pose_vector], norm="l2")

    x_vals, y_vals = pose_vector.reshape((2, -1))
    return x_vals, y_vals


class Pose:
    def __init__(self, score, key_points: typing.List[KeyPoint]):
        self.score = score
        self.key_points = key_points

        self.names, self.scores, self.x_vals, self.y_vals = np.transpose(
            np.array(key_points)
        )

        self.scores = self.scores.astype(float)
        self.x_vals = self.x_vals.astype(float)
        self.y_vals = self.y_vals.astype(float)

        self.norm_x_vals, self.norm_y_vals = resize_reshape(self.x_vals, self.y_vals)
        self.keep_aspect_ratio_x_vals, self.keep_aspect_ratio_y_vals = resize_reshape(
            self.x_vals, self.y_vals, keep_aspect_ratio=True
        )

        self.norm_x_vals, self.norm_y_vals = get_l2_norm(
            self.norm_x_vals, self.norm_y_vals
        )
        self.keep_aspect_ratio_x_vals, self.keep_aspect_ratio_y_vals = get_l2_norm(
            self.keep_aspect_ratio_x_vals, self.keep_aspect_ratio_y_vals
        )

    @staticmethod
    def from_json_result(model_result):
        max_scored_pose = max(model_result, key=lambda x: x["score"])
        score = max_scored_pose["score"]
        key_points = [
            KeyPoint(
                point["part"],
                point["score"],
                point["position"]["y"],
                point["position"]["x"],
            )
            for point in max_scored_pose["keypoints"]
        ]

        return Pose(score, key_points)
