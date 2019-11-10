import glob
import json

data_folder_name = "MPI_label_2"
json_file_paths = glob.glob("/home/felix-cheng/hpr_test/MPI_label_2/**/*.json", recursive=True)
category = ""
path_to_image = ""


output_file = open("./labelled_data.csv", 'w')

def replace_spaces_with_hyphens(string):
	newstr = ""
	for char in string:
		if char == " ":
			newstr += "-"
		else:	
			newstr += char
	return newstr

for file_path in json_file_paths:
	category_substring_idx_start = file_path.find(data_folder_name) + len(data_folder_name) + 1
	category_substring_idx_end = file_path.find("/", category_substring_idx_start)
	category = replace_spaces_with_hyphens(file_path[category_substring_idx_start:category_substring_idx_end])
	
	img_type_substr_idx_end = file_path.find("/", category_substring_idx_end + 1)
	img_type = file_path[category_substring_idx_end + 1:img_type_substr_idx_end]

	if img_type == "discard":
		continue

	json_contents = json.load(open(file_path))
	path_to_image = replace_spaces_with_hyphens(json_contents["image_path"])
	output_file.write(replace_spaces_with_hyphens(file_path) + "," + path_to_image + "," + category + "," + img_type + "\n")

