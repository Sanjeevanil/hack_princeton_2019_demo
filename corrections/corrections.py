from joblib import dump, load
from os.path import dirname, abspath
import json
import sys
import pdb
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np
import math

from models.posenet_js_results.pose_objects import Pose
from clustering_experiments.nearest_centroid import euclidian_dist_2d
from clustering_experiments.visualize_results import POSE_NAMES

id_to_name_map = {0: 'virabhadrasana_i', 1: 'utkatasana', 2: 'ardha_uttanasana', 3: 'bhujangasana', 4: 'paschimottanasana', 5: 'adho_mukha_svanasana', 6: 'purvottanasana', 7: 'marjaryasana', 8: 'chaturanga_dandasana', 9: 'ananda-balasana', 10: 'anjaneyasana', 11: 'malasana', 12: 'marichyasana_iii', 13: 'ustrasana', 14: 'salabhasana', 15: 'bitilasana', 16: 'uttanasana', 17: 'agnistambhasana', 18: 'balasana', 19: 'urdhva_hastasana'}

MODEL = "./models/trained_yoga_classifier.joblib"
model_centroids = json.load(open("%s/models/model_centroids.json" 
    % dirname(dirname(abspath(__file__)))))

model=None

def predict(json_result, classname): 
    pose = Pose.from_json_result(json_result)
    features = pose.get_position_features()
    features = features.reshape(1, -1) 
    if not classname:
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
         # centroid will be normalized or aspect ratio - whatever is fed into 
         # model for training 
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
        if mag <= 0.0005:
            continue
        error[pose.names[i]] = {
            "name": pose.names[i],
            "key_id": int(i),
            "x_correct": float(pose.norm_x_vals[i]/mag),
            "y_correct": float(pose.norm_y_vals[i]/mag),
            "x_coord": int(pose.x_vals[i]),
            "y_coord": int(pose.y_vals[i])
        }
    return error 
    
def return_error(json_features, classname=None):
    global model 
    if not model:
        model = load(MODEL)
    pose = predict(json_features, classname)
    err_indices = calculate_max_error(pose)
    error_data = get_error_data(err_indices, pose)
    return error_data

if __name__ == "__main__":
    model = load(MODEL)
    # json_features = json.load(open("./corrections/19-0.json"))
    # return_error(json_features)


