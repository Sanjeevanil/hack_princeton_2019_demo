# hack_princeton_2019_demo

## resources

(OpenPose models)[https://drive.google.com/open?id=1ofJ-SL1MxhfYmgHsqqIozwrocIWEmPmc]
(Source)[https://github.com/DhruvJawalkar/yoga-poses-dataset] for original yoga pose dataset


## setup

to setup a conda environment with the required packages, run:
```$ conda env create -f requirements.yml```

Download .caffemodel files to load model weights from the google drive into the corresponding folders

## testing openpose

run `models/openpose/opencv_openpose.py` from the base directory with flags for input and output images to test on a specific image

run `models/openpose/opencv_webcam.py` from the base directory to start running with webcam

`models/openpose` might have to be marked as a sources root if working on pycharm
