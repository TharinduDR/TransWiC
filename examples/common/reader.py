import pandas as pd


def read_training_file(data_path, annotation_path):
    data_df = pd.read_json(data_path, orient='records')
    annotation_data = pd.read_json(annotation_path, orient='records')

    return data_df.merge(annotation_data, left_on='id', right_on='id', how='left')


def read_test_file(data_path):
    data_df = pd.read_json(data_path, orient='records')
    return data_df
