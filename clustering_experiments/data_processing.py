
import csv 
import pdb 
import os 
from typing import List

import argparse
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
import pandas as pd

from cluster_image_point import ClusterImagePoint

def read_into_cluster_list(filepath):
    df = pd.read_csv(filepath)
    cluster_list = []
    for index, row in df.iterrows():
        filepath = "%s/%s" %(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            row['json_path'][2:])
        cluster_list.append(ClusterImagePoint(filepath, 
            row['yoga_class'], row['img_group']))
        
    return cluster_list

def get_cluster_dataset(cluster_list: List[ClusterImagePoint]): 
    datapoints = []
    labels = []    
    for point in cluster_list:
        features, classname = point.get_position_features()
        datapoints.append(features)     
        labels.append(classname)
    
    datapoints = np.array(datapoints)
    labels = np.array(labels)
    return datapoints, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="File path to csv metadata")

    args = parser.parse_args()
    cluster_list = read_into_cluster_list(args.csv_file)
    datapoints, labels = get_cluster_dataset(cluster_list)
    pdb.set_trace()