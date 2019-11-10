import glob
import json
import pdb
from os.path import dirname, basename

import pandas as pd
from sklearn.model_selection import train_test_split

json_file_paths = glob.glob("../**/*.json", recursive=True)


def get_record(filepath_list):
    for file_path in filepath_list:
        category = basename(dirname(dirname(file_path)))
        img_type = basename(dirname(file_path))

        if img_type == "discard":
            continue

        yield file_path, category, img_type


def get_dataframe():
    df = pd.DataFrame(get_record(json_file_paths))
    df.columns = ["json_path", "yoga_class", "img_group"]
    return df


def partition_data(full_df=None):
    train, test = train_test_split(full_df, test_size=0.1, stratify=df["yoga_class"])
    train.to_csv("training_set.csv")
    test.to_csv("validation_set.csv")


if __name__ == "__main__":
    df = get_dataframe()
    partition_data(df)
