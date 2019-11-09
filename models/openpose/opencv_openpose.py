"""
using posenet with opencv
credit: https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
"""

import argparse
import collections

import cv2
import time
import numpy as np

Point = collections.namedtuple("Point", ["confidence", "x", "y"])


def get_mode_details(mode):
    if mode == "COCO":
        protoFile = "models/openpose/pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "models/openpose/pose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [
            [1, 0],
            [1, 2],
            [1, 5],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [1, 11],
            [11, 12],
            [12, 13],
            [0, 14],
            [0, 15],
            [14, 16],
            [15, 17],
        ]

    elif mode == "MPI":
        protoFile = (
            "models/openpose/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        )
        weightsFile = "models/openpose/pose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 14],
            [14, 8],
            [8, 9],
            [9, 10],
            [14, 11],
            [11, 12],
            [12, 13],
        ]

    else:
        raise ValueError("MODE must be either COCO or MPI")

    return protoFile, weightsFile, nPoints, POSE_PAIRS


def model_output_function(proto_file, weights_file, in_width, in_height):
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    def get_model_output(frame):
        inpBlob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False
        )

        net.setInput(inpBlob)

        return net.forward()

    return get_model_output


def parse_points(output, n_points, frame_width, frame_height, H, W, scale=True):
    points = []
    for i in range(n_points):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if scale:
            # Scale the point to fit on the original image
            x = int((frame_width * point[0]) / W)
            y = int((frame_height * point[1]) / H)
        else:
            x = point[0]
            y = point[1]

        points.append(Point(prob, x, y))

    return points


def draw_points(points, threshold, frame, pose_pairs):
    frame_copy_1 = np.copy(frame)
    frame_copy_2 = np.copy(frame)
    for i, (prob, x, y) in enumerate(points):

        if prob > threshold:
            cv2.circle(
                frame_copy_1,
                (int(x), int(y)),
                8,
                (0, 255, 255),
                thickness=-1,
                lineType=cv2.FILLED,
            )
            cv2.putText(
                frame_copy_1,
                "{}".format(i),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        else:
            continue

    # Draw Skeleton
    for pair in pose_pairs:
        partA = pair[0]
        partB = pair[1]

        if points[partA][0] > threshold and points[partB][0] > threshold:
            cv2.line(
                frame_copy_2, points[partA][1:], points[partB][1:], (0, 255, 255), 2
            )
            cv2.circle(
                frame_copy_2,
                points[partA][1:],
                8,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.FILLED,
            )

    return frame_copy_1, frame_copy_2


def get_frame_results(frame, model_forward_pass, n_points):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    t = time.time()

    output = model_forward_pass(frame)
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    points = parse_points(output, n_points, frame_width, frame_height, H, W)

    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="MPI")
    parser.add_argument("--input-width", type=int, default=368)
    parser.add_argument("--input-height", type=int, default=368)
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="confidence threshold for display"
    )
    parser.add_argument("-i", "--image-path")
    parser.add_argument("--output-name", type=str, default="images/test/Output")

    args = parser.parse_args()

    protoFile, weightsFile, nPoints, POSE_PAIRS = get_mode_details(args.mode)

    frame = cv2.imread(args.image_path)

    model_forward_pass = model_output_function(
        protoFile, weightsFile, args.input_width, args.input_height
    )

    points = get_frame_results(frame, model_forward_pass, nPoints)

    frame_copy_1, frame_copy_2 = draw_points(points, args.threshold, frame, POSE_PAIRS)

    cv2.imshow("Output-Keypoints", frame_copy_1)
    cv2.imshow("Output-Skeleton", frame_copy_2)

    cv2.imwrite(args.output_name + "-Keypoints.jpg", frame_copy_1)
    cv2.imwrite(args.output_name + "-Skeleton.jpg", frame_copy_2)

    cv2.waitKey(0)
