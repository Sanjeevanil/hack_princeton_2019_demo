import csv
import math

percent_to_be_validation = 10

percent_to_be_validation /=100
dataset = csv.reader(open("./labelled_data.csv", 'r'))

data_by_category = {}

category_idx = 2

for row in dataset:
	row_category = row[category_idx]
	if row_category not in data_by_category:
		data_by_category[row_category] = []

	row_contents = row[0:category_idx]
	row_contents.append(row[category_idx + 1])
	data_by_category[row_category].append(row_contents)

training_set_file = open("training_set.csv", 'w')
validation_set_file = open("validation_set.csv", 'w')
for category_name in data_by_category.keys():
	total_samples = len(data_by_category[category_name])
	
	num_validation_data_points = math.ceil(total_samples * percent_to_be_validation)

	for item_num in range(0, num_validation_data_points - 1):
		row = data_by_category[category_name][item_num]
		validation_set_file.write(row[0] + "," + row[1] + "," + category_name  + "\n")

	for item_num in range(num_validation_data_points, total_samples - num_validation_data_points):
		row = data_by_category[category_name][item_num]
		training_set_file.write(row[0] + "," + row[1] + "," + category_name + "," + row[2] + "\n")

	
