# hack_princeton_2019_demo

## Setup

to setup a conda environment with the required packages, run:
```$ conda install -f -y -q --name hack_princeton_2019 -c conda-forge --file requirements.txt```

## Obtaining posenet output 
1. Place images into `static/images` folder in the same file structure as you
    would want the output
2. From the main directory, run `python app.py` 
3. Go to `localhost:5000/get_multiple_poses` 
4. Wait until all json outputs are processed. They will be in `./model_result/`
5. Run `python3 ./process_data/create_datasets.py` to aggregate the json data
    into csv files and partitioning the training and validation sets (10% of 
    total set is put into validation set). It will also filter out any images
    for which no keypoints were found, before partitioning. These csv files
    will also be under `./model_result` 
6. Run `python3 clustering_experiments/visualize_results.py` to see how the 
    keypoints map on the images. The frames with keypoints will be outputted
    to `./model_result/visualize_results*` depending on how many times it is
    run (creates a new folder if it detects on is already there) 
- NOTES: 
    - All image names must be devoid of spaces, `%` and `+` characters 
    (alphanumeric characters and underscores preferred)
    - Must be of `.png`, `.jpg` or `.jpeg` formats

## Clustering posenet output
- The functions in `./clustering_experiments/data_processing.py` can be used
    to organize the csv data into the ClusterImagePoint data structure. 
    The function `read_into_dictionary($CSV_FILE)` returns a list of the 
    ClusterImagePoint objects. This list can then be passed into 
    `get_cluster_dataset` to get the `(features, labels)` which can 
    then be passed into any clustering algorithm 