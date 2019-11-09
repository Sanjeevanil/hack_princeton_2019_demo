import os
import json

from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, Response

from models.openpose import opencv_openpose

app = Flask(__name__)

protoFile, weightsFile, nPoints, POSE_PAIRS = opencv_openpose.get_mode_details("MPI")
input_width = 368
input_height = 368
get_model_output = opencv_openpose.model_output_function(
    protoFile, weightsFile, input_width, input_height
)


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


@app.route("/image", methods=["POST"])
def image():
    try:
        image_file = request.files["image"]  # get the image

        # finally run the image through tensor flow object detection`
        frame = Image.open(image_file)
        frame = load_image_into_numpy_array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        output = get_model_output(frame)
        H = output.shape[2]
        W = output.shape[3]

        points = opencv_openpose.parse_points(
            output, nPoints, frame_width, frame_height, H, W
        )

        return json.dumps({"points": [point._asdict() for point in points]})

    except Exception as e:
        print("POST /image error: %e" % e)
        return e


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
