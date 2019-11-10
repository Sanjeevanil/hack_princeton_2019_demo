from typing import Optional, Union
import json
import sys
import pdb
from os.path import dirname, abspath

import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))

from models.posenet_js_results.pose_objects import Pose


class ClusterImagePoint(object):
    def __init__(
        self,
        json_path: str,
        yoga_class: str,
        data_subgroup: str,
        pose: Optional[Union[Pose, None]] = None,
    ) -> None:
        self._json_path = json_path
        self.model_result_json = json.load(open(self._json_path))
        self.yoga_class = yoga_class
        self.data_subgroup = data_subgroup

        if not pose:
            self.pose = Pose.from_json_result(self.model_result_json)
        else:
            self.pose = pose

    def set_pose(self, pose: Pose):
        self.pose = pose

    def get_position_features(self) -> tuple:
        features = np.concatenate((self.pose.norm_x_vals, self.pose.norm_y_vals, self.pose.scores))
        return features, self.yoga_class
