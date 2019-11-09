import argparse
import os
import glob
import json

import cv2

from opencv_openpose import (
    get_mode_details,
    model_output_function,
    get_frame_results,
    draw_points,
)
from normalize import normalize_pose


def get_image_files(base_dir, file_extensions=("png", "jpeg", "jpg"), recursive=False):
    image_files = []
    for file_ext in file_extensions:
        image_files.extend(glob.glob(os.path.join(base_dir, f"*.{file_ext}")))
        if recursive:
            image_files.extend(glob.glob(os.path.join(base_dir, f"**/*.{file_ext}")))

    return image_files


def label_image_folders(image_files, model_forward_pass, n_points, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in image_files:
        img_name = os.path.splitext(img_path)[0]
        if img_name[-10:] == "-Keypoints" or img_name[-9:] == "-Skeleton":
            continue

        img_base_name = os.path.basename(img_name)

        print(img_path)
        frame = cv2.imread(img_path)

        points = get_frame_results(frame, model_forward_pass, n_points)

        frame_copy_1, frame_copy_2 = draw_points(
            points, args.threshold, frame, POSE_PAIRS
        )

        cv2.imwrite(img_name + "-Keypoints.jpg", frame_copy_1)
        cv2.imwrite(img_name + "-Skeleton.jpg", frame_copy_2)

        normalized_points = normalize_pose(points)

        json.dump(
            {
                "image_path": img_path,
                "points": [point._asdict() for point in points],
                "normalized_points": [point._asdict() for point in normalized_points],
            },
            open(os.path.join(output_dir, img_base_name + ".json"), "w+"),
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="MPI")
    parser.add_argument("--input-width", type=int, default=368)
    parser.add_argument("--input-height", type=int, default=368)
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="confidence threshold for display"
    )
    parser.add_argument(
        "-b", "--base-dir", type=str, help="base directory of where to look for images"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="find images recursively from base directory",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="where to save all the json results"
    )

    args = parser.parse_args()

    protoFile, weightsFile, nPoints, POSE_PAIRS = get_mode_details(args.mode)
    model_forward_pass = model_output_function(
        protoFile, weightsFile, args.input_width, args.input_height
    )

    image_files = get_image_files(args.base_dir, recursive=args.recursive)
    label_image_folders(image_files, model_forward_pass, nPoints, args.output_dir)
