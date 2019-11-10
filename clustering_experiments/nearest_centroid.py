
from sklearn.neighbors.nearest_centroid import NearestCentroid

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np
import csv
import json
from clustering_experiments.data_processing import read_into_cluster_list, get_cluster_dataset
import math
import visualize_pose_data

MODEL_RESULT_FOLDER = "model_result"

def euclidian_dist_2d(vec1, vec2):
	if len(vec1) != 2 or len(vec2) != 2:
		raise Exception("euclidian_dist_2d expects 2D vectors only")

	return math.sqrt(\
			(vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 \
		)

def move_mirror_dist(pt1, pt2):
	confidence_meas_spacing = 3
	confidence_meas_idx_counter = confidence_meas_spacing  
	
	distance_summation_1 = 0
	distance_summation_2 = 0

	for feature_idx in range(0, len(pt1), confidence_meas_spacing):
		pt1_coord = [pt1[feature_idx+1], pt1[feature_idx+2]]
		pt2_coord = [pt2[feature_idx+1], pt2[feature_idx+2]]

		distance_summation_1 += pt1[feature_idx]
		distance_summation_2 += pt1[feature_idx] * euclidian_dist_2d(pt1_coord, pt2_coord)

	distance = distance_summation_2/distance_summation_1
	return distance

def str_arr_to_comma_sep_list(str_arr):
		str_build = str(str_arr[0])

		for string in str_arr[1:]:
			str_build += "," + str(string)
		
		return str_build

def model_output_to_csv(predicted_classes, true_classes, validation_set_file):
	
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

def dict_to_csv(dictionary, csv_file_name):
	out_file = open(csv_file_name, 'w')
	
	str_builder = ""
	for key in dictionary.keys():
		str_builder += key + ","

	str_builder = str_builder[:-1] + "\n"
	out_file.write(str_builder)

	max_len = 0
	for key in dictionary.keys():
		if len(dictionary[key]) > max_len:
			max_len = len(dictionary[key])

	for idx in range(max_len):
		str_builder = ""
		for key in dictionary.keys():
			if idx < len(dictionary[key]):
				str_builder += str(dictionary[key][idx])
			str_builder += ","
		str_builder = str_builder[:-1] + "\n"
		out_file.write(str_builder)		
	out_file.close()

def map_model_space(feature_list, corresp_classes):
	model_space = {}

	for img_idx in range(len(feature_list)):
		if corresp_classes[img_idx] not in model_space.keys():
			model_space[corresp_classes[img_idx]] = []
		model_space[corresp_classes[img_idx]].append(feature_list[img_idx])
	return model_space

def map_model_space_w_ids(feature_list, corresp_classes, id_val):
	model_space = {}

	for img_idx in range(len(feature_list)):
		if corresp_classes[img_idx] not in model_space.keys():
			model_space[corresp_classes[img_idx]] = {}
		model_space[corresp_classes[img_idx]][id_val[img_idx]] = feature_list[img_idx]
	return model_space

def matr_to_csv(matr, idx_to_name, output_path, row_headings=None):
	output_file = open(output_path, 'w')

	header_arr = []
	if isinstance(matr, list):
		loop_num = len(matr[0])
	else:
		loop_num = matr.shape[1]

	for idx in range(loop_num):
		header_arr.append(idx_to_name[idx])

	output_file.write("," + str_arr_to_comma_sep_list(header_arr) + "\n")

	if row_headings == None:
		row_headings = header_arr

	row_num = 0
	for row in matr:
		output_file.write(row_headings[row_num] + "," + str_arr_to_comma_sep_list(row) + "\n")	
		row_num += 1
	output_file.close()

def invert_dict(dictionary):
	return dict([[v,k] for k,v in dictionary.items()])

def calc_inter_centroid_dist(centroids):
	key_list = list(centroids.keys())

	key_name_to_key_id_map = {}
	for idx in range(len(key_list)):
		key_name_to_key_id_map[key_list[idx]] = idx

	dist_matr = np.empty((len(key_list), len(key_list)))
	dist_matr[:] = np.nan

	for pose_class_outer_loop in centroids.keys():
		for pose_class_inner_loop in centroids.keys():
			
			dist_matr[key_name_to_key_id_map[pose_class_outer_loop], key_name_to_key_id_map[pose_class_inner_loop]] = \
				move_mirror_dist(centroids[pose_class_outer_loop], centroids[pose_class_inner_loop])

	matr_to_csv(dist_matr, invert_dict(key_name_to_key_id_map), "./inter_centroid_dists.csv")

def calc_dist_to_foreign_centroids(model_space_labelled, centroids):
	aggreg_indiv_pts_ids = []
	
	for pose_class in model_space_labelled.keys():
		aggreg_indiv_pts_ids.extend( \
			list(model_space_labelled[pose_class].keys()) \
			)	

	idxs_to_headings = {\
		0: "classification"\
	}
	counter = 1
	for pose_class in model_space_labelled.keys():
		idxs_to_headings[counter] = pose_class
		counter += 1

	matr = []

	for pose_class in model_space_labelled.keys():
		for point in model_space_labelled[pose_class].keys():

			dist_list = [pose_class]
			for foreign_pose_class in centroids.keys():
				
				if foreign_pose_class == pose_class:
					dist_list.append(" ")
				else:
					dist_list.append(str(move_mirror_dist(model_space_labelled[pose_class][point], centroids[foreign_pose_class])))

			matr.append(dist_list)
	matr_to_csv(matr, idxs_to_headings, "./img_dist_to_foreign_centroids.csv", row_headings=aggreg_indiv_pts_ids)

def eval_prediction_results(model_space_labelled, total_class_list, true_classes):
	aggreg_indiv_pts_ids = []
	
	for pose_class in model_space_labelled.keys():
		aggreg_indiv_pts_ids.extend( \
			list(model_space_labelled[pose_class].keys()) \
			)	

	idxs_to_headings = {\
		0: "actual class"\
	}
	counter = 1
	for pose_class in total_class_list:
		idxs_to_headings[counter] = pose_class
		counter += 1

	matr = []

	true_class_counter = 0
	for pose_class in model_space_labelled.keys():
		for point in model_space_labelled[pose_class].keys():

			dist_list = [true_classes[true_class_counter]]
			true_class_counter += 1
			for pose_class_inner_loop in centroids.keys():
				dist_list.append(str(move_mirror_dist(model_space_labelled[pose_class][point], centroids[pose_class_inner_loop])))

			matr.append(dist_list)
	matr_to_csv(matr, idxs_to_headings, "./prediction_results.csv", row_headings=aggreg_indiv_pts_ids)

def calc_centroid_stats(model_space, outp_dir):

	featurewise_variances = {}
	distance_variances_centroid_as_pt2 = {}
	distance_variances_centroid_as_pt1 = {}
	centroids = {}

	for pose_class in model_space.keys():
		point_accum = np.zeros((1, len(model_space[pose_class][0])))
		
		for point in model_space[pose_class]:
			point_accum += np.array(point)
		
		centroids[pose_class] = point_accum/len(model_space[pose_class])
		centroids[pose_class] = centroids[pose_class][0]
		featurewise_variances[pose_class] = np.var(model_space[pose_class], axis=0)


		dist_arr_centroid_as_pt2 = []
		dist_arr_centroid_as_pt1 = []
		for point in model_space[pose_class]:
			dist_arr_centroid_as_pt2.append(move_mirror_dist(point, centroids[pose_class].tolist()))
			dist_arr_centroid_as_pt1.append(move_mirror_dist(centroids[pose_class].tolist(), point))

		distance_variances_centroid_as_pt2[pose_class] = [np.var(np.array(dist_arr_centroid_as_pt2))]
		distance_variances_centroid_as_pt1[pose_class] = [np.var(np.array(dist_arr_centroid_as_pt1))]

	dict_to_csv(centroids, outp_dir + "centroids.csv")
	dict_to_csv(featurewise_variances, outp_dir + "featurewise_variances.csv")
	dict_to_csv(distance_variances_centroid_as_pt1, outp_dir + "distance_variances_centroid_as_pt1.csv")
	dict_to_csv(distance_variances_centroid_as_pt2, outp_dir + "distance_variances_centroid_as_pt2.csv")
	return centroids, featurewise_variances, distance_variances_centroid_as_pt1, distance_variances_centroid_as_pt2


cluster_list = read_into_cluster_list("../"+MODEL_RESULT_FOLDER+"/training_set.csv")
x, y, training_pt_ids= get_cluster_dataset(cluster_list)


class_name_to_class_id_map = {}
y_ids, name_to_id_map = map_names_to_ids(y)
y_ids = np.array(y_ids)


id_to_name_map = dict([[v,k] for k,v in name_to_id_map.items()])

model = NearestCentroid(metric=move_mirror_dist)
model.fit(x, y_ids)

cluster_list = read_into_cluster_list("../"+MODEL_RESULT_FOLDER+"/validation_set.csv")
validation_x, validation_y, validation_ids = get_cluster_dataset(cluster_list)

quick_and_dirty_csv_read = csv.reader(open("../"+MODEL_RESULT_FOLDER+"/validation_set.csv"))

predictions_y_ids = model.predict(validation_x)

predictions = []

for id_num in predictions_y_ids:
	predictions.append(id_to_name_map[id_num])	

v_y_ids, _ = map_names_to_ids(validation_y)

v_y_ids = np.array(v_y_ids)

model_output_to_csv(predictions, validation_y, quick_and_dirty_csv_read)

model_space = map_model_space(x,y)
model_space_labelled = map_model_space_w_ids(x, y, training_pt_ids)

predicted_model_space_labelled = map_model_space_w_ids(validation_x, predictions_y_ids, validation_ids)

centroids, \
featurewise_variances, \
distance_variances_centroid_as_pt1, \
distance_variances_centroid_as_pt2 = calc_centroid_stats(model_space, "./")

calc_inter_centroid_dist(centroids)
calc_dist_to_foreign_centroids(model_space_labelled, centroids)

eval_prediction_results(predicted_model_space_labelled, model_space_labelled.keys(), validation_y)
'''
for pose_class in centroids.keys():
	print(pose_class)
	visualize_pose(centroids[pose_class], pose_class)
'''

