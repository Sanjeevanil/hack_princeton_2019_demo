import os
import json
import glob

from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, Response, render_template

app = Flask(__name__)


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
    return Response("Tensor Flow object detection")


@app.route("/local")
def local():
    return Response(open("./static/local.html").read(), mimetype="text/html")


@app.route("/show-pose", methods=["POST"])
def show_pose():
    try:
        pose = request.json["value"]
        print(pose)

        return "nice!"

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
