from os.path import dirname, basename, splitext, abspath, exists
from os import makedirs
import glob
import pdb
import json
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import cv2

from models.posenet_js_results.pose_objects import Pose

IMG_FILE_DIRECTORY = "./static/images"
MODEL_RESULT_DIRECTORY = "./model_result"
JSON_FILE_DIRECTORY = "%s/**/images" % MODEL_RESULT_DIRECTORY


POSE_NAMES= [
    "nose", 
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle"
]

POSE_PAIRS= [
    (5,7),
    (6,8),
    (7,9),
    (8,10),
    (11,13),
    (12,14),
    (13,15),
    (14,16), 
    (5,6), 
    (6,12),
    (12,11), 
    (11,5)
]

def map_json_to_img(json_file_path): 
    rel_path_start = json_file_path.find("images/")
    rel_path = json_file_path[rel_path_start+7:] # 7 is length of substring search query
    img_name, _ = splitext(rel_path)
    img_matches = glob.glob("%s/%s*" % (IMG_FILE_DIRECTORY, img_name), recursive=True)
    if len(img_matches) > 1:
        img_full_path = None
        for match in img_matches:
            match_name, _ = splitext(basename(match))
            if basename(img_name) == match_name:
                print("Found match for ambiguous name %s, \n\tMatches: %s"
                    "\n\tFinal match: %s" %(img_name, img_matches, match))
                img_full_path = match
                break 
        if not img_full_path:
            print("Unable to identify img! Ambiguous name %s, \n\tMatches: %s" 
                % (img_name, img_matches))
    else:
        img_full_path = img_matches[0]
    
    return img_full_path, json_file_path

def parse_json(img_dict):
    keys_to_delete = []
    for img in img_dict.keys():
        metadata = json.load(open(img_dict[img]))
        if not metadata:
            keys_to_delete.append(img)
        else:
            img_dict[img] = Pose.from_json_result(metadata)
    
    for img_key in keys_to_delete:
        print("No Keypoints: %s" % img_key) 
        del img_dict[img_key]
    
    return img_dict


def draw_img(img_filepath, pose, threshold=0.1):
    # opencv x and y are tranpose of posenet's 
    frame = cv2.imread(img_filepath)
    for i in range(len(POSE_NAMES)):
        y = int(pose.x_vals[i]) 
        x = int(pose.y_vals[i])
        prob = pose.scores[i]
        if prob > threshold:
            cv2.circle(
                frame,
                (x,y),
                8,
                (0, 255, 255),
                thickness=-1,
                lineType=cv2.FILLED,
            )
            cv2.putText(
                frame,
                "{}".format(i),
                (x,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        else:
            continue

    # Draw Skeleton
    for pair in POSE_PAIRS:
        indexA = pair[0]
        indexB = pair[1]
        pAy = int(pose.x_vals[indexA])
        pAx = int(pose.y_vals[indexA])
        pBy = int(pose.x_vals[indexB])
        pBx = int(pose.y_vals[indexB])

        if pose.scores[indexA] > threshold and pose.scores[indexB] > threshold:
            cv2.line(
                frame, (pAx, pAy),(pBx, pBy), (0, 255, 255), 2
            )
            cv2.circle(
                frame,
                (pAx, pAy),
                8,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.FILLED,
            )

    return frame

if __name__ == "__main__":

    JSON_FILE_PATHS = glob.glob("%s/**/*.json" % JSON_FILE_DIRECTORY, recursive=True)
    img_pairs = {}
    for json_file in JSON_FILE_PATHS:
        img_full_path, _ = map_json_to_img(json_file)
        if img_full_path:
            img_pairs[img_full_path] = json_file
    
    img_dict = parse_json(img_pairs)
    for key, pose in img_dict.items():
        if (len(pose.norm_x_vals) != 17 or len(pose.norm_y_vals) != 17 or
            len(pose.names) != 17 or len(pose.scores) != 17):
            print(key)
            print(len(pose.norm_x_vals), len(pose.norm_y_vals),
            len(pose.names), len(pose.scores))

    output_base_dir = "%s/visualize_results" % MODEL_RESULT_DIRECTORY

    existing_dir_count = len(glob.glob(output_base_dir + "*"))
    if existing_dir_count:
        output_base_dir += f"_{existing_dir_count}"

    makedirs(output_base_dir)

    for img, pose in img_dict.items():
        rel_path_start = img.find("images/")
        rel_path = img[rel_path_start+7:] # 7 is length of substring search 
        output_dir = "%s/%s" % (output_base_dir, dirname(rel_path))
        if not exists(output_dir):
            makedirs(output_dir)
        frame = draw_img(img, pose)
        cv2.imwrite("%s/%s" % (output_dir, basename(img)), frame)
        # cv2.imshow(rel_path,frame)
        # cv2.waitKey(0)