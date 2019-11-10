
from sklearn.neighbors.nearest_centroid import NearestCentroid

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np
import csv
import json
from clustering_experiments.data_processing import read_into_dictionary, get_cluster_dataset

MODEL_RESULT_FOLDER = "model_result"

def move_mirror_dist(pt1, pt2):
	confidence_meas_spacing = 3
	confidence_meas_idx_counter = confidence_meas_spacing  
	
	distance_summation_1 = 0
	distance_summation_2 = 0

	
	for feature_idx in range(0, len(pt1), confidence_meas_spacing):
		distance_summation_1 += pt1[feature_idx]
		distance_summation_2 += pt1[feature_idx] * np.sum(abs(np.array(pt1[feature_idx+1:feature_idx+2]) - \
			np.array(pt2[feature_idx+1:feature_idx+2])))

	distance = distance_summation_2/distance_summation_1
	return distance

def model_output_to_csv(predicted_classes, true_classes, validation_set_file):

	def str_arr_to_comma_sep_list(str_arr):
		str_build = str_arr[0]

		for string in str_arr[1:]:
			str_build += "," + string
		
		return str_build

	results_file = open("../"+MODEL_RESULT_FOLDER+"/model_performance_logs.csv", 'w')

	if len(predicted_classes) != len(true_classes):
		raise Exception("Error! Number of json paths needs to be" + \
			"same as number of category guesses")
	
	idx = 0
	num_correct_guesses = 0
	is_header_row = True
	for row in validation_set_file:
		if is_header_row:
			is_header_row = False
			continue

		result_text = "FAIL"
		#if idx >= len(predicted_classes):
		#	break
		if predicted_classes[idx] == true_classes[idx]:
			result_text = "PASS"
			num_correct_guesses +=1

		results_file.write(str_arr_to_comma_sep_list(row) + "," + \
			predicted_classes[idx] + "," + result_text + "\n")
		idx += 1

	print(num_correct_guesses, " CORRECT OUT OF ", len(predicted_classes))
	return num_correct_guesses, len(predicted_classes)

def map_names_to_ids(names_list):
	name_to_id_map = {}
	ids = []

	names_counter = 0 
	for name in names_list:
		if name not in name_to_id_map:
			name_to_id_map[name] = names_counter
			names_counter += 1
		ids.append(name_to_id_map[name])

	return ids, name_to_id_map


cluster_list = read_into_dictionary("../"+MODEL_RESULT_FOLDER+"/training_set.csv")
x, y = get_cluster_dataset(cluster_list)

class_name_to_class_id_map = {}
y_ids, name_to_id_map = map_names_to_ids(y)
y_ids = np.array(y_ids)


id_to_name_map = dict([[v,k] for k,v in name_to_id_map.items()])

model = NearestCentroid(metric=move_mirror_dist)
model.fit(x, y_ids)

cluster_list = read_into_dictionary("../"+MODEL_RESULT_FOLDER+"/validation_set.csv")
validation_x, validation_y = get_cluster_dataset(cluster_list)

quick_and_dirty_csv_read = csv.reader(open("../"+MODEL_RESULT_FOLDER+"/validation_set.csv"))

predictions_y_ids = model.predict(validation_x)

predictions = []

for id_num in predictions_y_ids:
	predictions.append(id_to_name_map[id_num])	

print(predictions_y_ids)
model_output_to_csv(predictions, validation_y, quick_and_dirty_csv_read)
