from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
import csv 
import argparse
import pdb 
import os 
from typing import List

from cluster_image_point import ClusterImagePoint

def read_into_dictionary(filepath):
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cluster_list = []
        index = 0
        for record in csv_reader:
            if index == 0:
                index+=1
                continue
            else: 
                # Parsing out relative directory into absolute directory 
                filepath = "%s%s" %(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    record[1][2:])
                try: 
                    cluster_list.append(ClusterImagePoint(filepath, 
                        record[2], record[3]))
                except: 
                    print("No keypoints: %s"% filepath)
                index+=1
        
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
    cluster_list = read_into_dictionary(args.csv_file)
    get_cluster_dataset(cluster_list)

