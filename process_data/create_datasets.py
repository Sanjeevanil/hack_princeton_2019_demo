import glob
import json
import pdb
from os.path import dirname, basename, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_RESULT_FOLDER = "./model_result"

JSON_FILE_PATHS = glob.glob("%s/**/*.json" % MODEL_RESULT_FOLDER, recursive=True)

def get_record(filepath_list):
	for file_path in filepath_list:
		contents = json.load(open(file_path))
		if not contents:
			print("No keypoints found: %s" % file_path)
			continue 

		category = basename(dirname(dirname(file_path)))
		img_type = basename(dirname(file_path))
		
		if img_type == "discard":
			continue

		yield file_path, category, img_type

def get_dataframe():
	df = pd.DataFrame(get_record(JSON_FILE_PATHS))
	df.columns = ["json_path", "yoga_class", "img_group"]
	return df 

def partition_data(full_df = None):
	train, test = train_test_split(full_df, test_size=.1, stratify=df['yoga_class'])
	train.to_csv("%s/training_set.csv" % MODEL_RESULT_FOLDER)
	test.to_csv("%s/validation_set.csv" % MODEL_RESULT_FOLDER)

if __name__ == "__main__":
	df = get_dataframe()
	partition_data(df)	