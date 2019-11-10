import os
import json
import glob

from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, Response, render_template

from models.posenet_js_results.pose_objects import Pose

app = Flask(__name__)


POSE_TO_IMAGE_MAP = {
    "anjaneyasana": "/static/media/anjaneyasana_71-0.png",
    "balasana": "/static/media/balasana_41._childs-pose.png",
    "bitilasana": "/static/media/bitilasana_50-0.png",
    "malasana": "/static/media/malasana_8-0.png",
    "marichyasana iii": "/static/media/marichyasana_iii_6-1.png",
    "marjaryasana": "/static/media/marjaryasana_77-0.png",
    "paschimottanasana": "/static/media/paschimottanasana_97-0.png",
    "purvottanasana": "/static/media/purvottanasana_35-0.png",
    "salabhasana": "/static/media/salabhasana _57-0.png",
    "ustrasana": "/static/media/ustrasana_21-0.png",
    "utkatasana": "/static/media/utkatasana_15._yoga-pose-101-utkatasana-or-chair-pose.png",
    "virabhadrasana": "/static/media/virabhadrasana_i_32-0.png",
}


def load_image_into_numpy_array(pil_image):
    (im_width, im_height) = pil_image.size
    return (
        np.array(pil_image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    )


# for CORS
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add(
        "Access-Control-Allow-Methods", "GET,POST"
    )  # Put any other methods you need here
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/learn_poses")
def learn_poses():
    return render_template("learn_poses.html", pose_images=POSE_TO_IMAGE_MAP)


@app.route("/learn/<pose_name>")
def learn_single_pose(pose_name):
    pose_img = POSE_TO_IMAGE_MAP[pose_name]

    return render_template("learn_single_pose.html", image_src=pose_img, pose_name=pose_name)


@app.route("/pose_correct/<pose_name>", methods=["POST"])
def pose_correct(pose_name):
    try:
        pose_result = request.json["value"]
        if pose_result:
            pose = Pose.from_json_result(pose_result)

            return json.dumps(pose.get_keypoint_dict())
        else:
            return json.dumps([])

    except Exception as e:
        print("POST /show-pose error: %e" % e)
        return e


@app.route("/local")
def local():
    return render_template("local.html")


@app.route("/show-pose", methods=["POST"])
def show_pose():
    try:
        pose_result = request.json["value"]
        if pose_result:
            pose = Pose.from_json_result(pose_result)

            return json.dumps(pose.get_keypoint_dict())
        else:
            return json.dumps([])

    except Exception as e:
        print("POST /show-pose error: %e" % e)
        return e


# save-pose and get_multiple_poses are used to get the model output
# for whatever images are saved in static/images
@app.route("/save-pose", methods=["POST"])
def save_pose():
    try:
        pose = request.json["value"]
        src = request.json["src"]
        print(src)
        out_filename = os.path.join(
            "model_result", os.path.splitext(src)[0].split(":")[-1] + ".json"
        )
        os.makedirs(os.path.split(out_filename)[0], exist_ok=True)
        print(out_filename)
        json.dump(pose, open(out_filename, "w+"), indent=4)

        return "nice!"

    except Exception as e:
        print("POST /show-pose error: %e" % e)
        return e


@app.route("/get_multiple_poses")
def get_multiple_poses_from_images():
    file_extensions = ("png", "jpeg", "jpg")
    image_files = []
    for file_ext in file_extensions:
        image_files.extend(glob.glob(f"static/images/*.{file_ext}"))
        image_files.extend(glob.glob(f"static/images/**/*.{file_ext}", recursive=True))

    return render_template("get_multiple_poses.html", image_files=image_files)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
