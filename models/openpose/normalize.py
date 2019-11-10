import typing

import numpy as np
from sklearn import preprocessing

from .opencv_openpose import Point


def resize_reshape(
    x_vals: np.array, y_vals: np.array, keep_aspect_ratio=False
) -> typing.Tuple[np.array, np.array]:
    """

    Args:
        x_vals: list of x values of each point in a pose
        y_vals: list of y values of each point in a pose
        keep_aspect_ratio (bool): if not, x and y will be scaled independently of each other.
            otherwise, will scale based on longest side

    Returns:
        numpy array of updated x values, and y values of each point in the pose
    """

    x_min = np.min(x_vals)
    x_max = np.max(x_vals)
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)

    if keep_aspect_ratio:
        x_len = x_max - x_min
        y_len = y_max - y_min

        if x_len > y_len:
            y_min -= (x_len - y_len) / 2
            y_max += (x_len - y_len) / 2
        else:
            x_min -= (y_len - x_len) / 2
            x_max += (y_len - x_len) / 2

    x_vals = (x_vals - x_min) / (x_max - x_min)
    y_vals = (y_vals - y_min) / (y_max - y_min)

    return x_vals, y_vals


def normalize_pose(points: typing.List[Point]):
    confidences, x_vals, y_vals = np.transpose(np.array(points))
    x_vals, y_vals = resize_reshape(x_vals, y_vals)

    # flatten so that vector is [x1, x2, ... xn, y1, y2 ..., yn]
    # normalization result shouldn't be different from [x1, y1, x2, y2, ...]
    pose_vector = np.array([x_vals, y_vals]).flatten()
    pose_vector = preprocessing.normalize([pose_vector], norm="l2")

    x_vals, y_vals = pose_vector.reshape((2, -1))

    return [
        Point(confidence, x, y) for confidence, x, y in zip(confidences, x_vals, y_vals)
    ]
