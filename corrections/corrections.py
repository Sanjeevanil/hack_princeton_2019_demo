from joblib import dump, load
from os.path import dirname, abspath
import json
import sys
import pdb
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np
import math

from models.posenet_js_results.pose_objects import Pose
from clustering_experiments.nearest_centroid import calc_centroid_stats, id_to_name_map, euclidian_dist_2d
from clustering_experiments.visualize_results import POSE_NAMES

MODEL = "./models/trained_yoga_classifier.joblib"
model_centroids = json.load(open("%s/models/model_centroids.json" 
    % dirname(dirname(abspath(__file__)))))

def predict(json_result): 
    pose = Pose.from_json_result(json_result)
    features = pose.get_position_features()
    features = features.reshape(1, -1) 
    class_id = model.predict(features)[0]
    classname = id_to_name_map[class_id]
    c_x, c_y, _ = np.array(model_centroids[classname]).reshape((3,-1))

    dummy_scores = [None for _ in range(len(POSE_NAMES))]
    keypoints = zip(POSE_NAMES, dummy_scores, c_x, c_y)
    centroid = Pose(1, list(keypoints))
    Pose.yoga_class = classname
    Pose.centroid = centroid  
    return pose 

def calculate_max_error(pose): 
    error = [] 
    for i in range(len(POSE_NAMES)): 
        if i < 5:
            # do not count facial features
            error.append(0)
            continue 
         # will be normalized or aspect ratio - whatever is fed into model to begin with
        centroid_vec = (pose.centroid.x_vals[i], pose.centroid.y_vals[i])
        pose_vec = (pose.norm_x_vals[i], pose.norm_y_vals[i])
        error.append(pose.scores[i]*euclidian_dist_2d(centroid_vec, pose_vec))
    pose.error = np.array(error)
    # Returns indices with highest error in order of most to least error 
    return np.flip(pose.error.argsort()[-3:])

def get_error_data(indices, pose ):
    error = {}
    for i in indices:
        mag = math.sqrt((pose.norm_x_vals[i]**2 + pose.norm_y_vals[i]**2 ))
        error[pose.names[i]] = {
            "key_id": i,
            "x_correct": pose.norm_x_vals[i]/mag,
            "y_correct": pose.norm_y_vals[i]/mag,
            "x_coord": pose.x_vals[i],
            "y_coord": pose.y_vals[i]
        }
    return error 
    
def return_error(json_features):
    pose = predict(json_features)
    err_indices = calculate_max_error(pose)
    error_data = get_error_data(err_indices, pose)
    return error_data


if __name__ == "__main__":
    global model 
    model = load(MODEL)
    # json_features = json.load(open("./corrections/19-0.json"))
    # return_error(json_features)


