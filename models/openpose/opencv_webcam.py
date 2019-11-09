import argparse

import cv2
from opencv_openpose import (
    get_mode_details,
    model_output_function,
    get_frame_results,
    draw_points,
)


def main(mode="MPI", input_width=368, input_height=368, threshold=0.1):
    proto_file, weights_file, n_points, pose_pairs = get_mode_details(mode)

    model_forward_pass = model_output_function(
        proto_file, weights_file, input_width, input_height
    )
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        points = get_frame_results(frame, model_forward_pass, n_points)

        frame_copy_1, frame_copy_2 = draw_points(points, threshold, frame, pose_pairs)

        # Display the resulting frame
        cv2.imshow("frame1", frame_copy_1)
        cv2.imshow("frame2", frame_copy_2)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="MPI")
    parser.add_argument("--input-width", type=int, default=368)
    parser.add_argument("--input-height", type=int, default=368)
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="confidence threshold for display"
    )

    args = parser.parse_args()

    main(args.mode, args.input_width, args.input_height, args.threshold)
