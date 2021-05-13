import pandas as pd


def read_training_file(data_path, annotation_path, args):
    data_df = pd.read_json(data_path, orient='records')
    annotation_data = pd.read_json(annotation_path, orient='records')

    complete_df = data_df.merge(annotation_data, left_on='id', right_on='id', how='left')

    if args["tagging"]:
        complete_df['sentence1'] = complete_df.apply(
            lambda row: include_tags(row['sentence1'], row['start1'], row['end1'], args), axis=1)
        complete_df['sentence2'] = complete_df.apply(
            lambda row: include_tags(row['sentence2'], row['start2'], row['end2'], args), axis=1)

    return complete_df


def read_test_file(data_path, args, cross_Lingual=False):
    data_df = pd.read_json(data_path, orient='records')

    if args["tagging"]:
        if cross_Lingual:
            data_df['sentence1'] = data_df.apply(
                lambda row: include_tags(row['sentence1'], int(row['ranges1'].split(",")[0].split("-")[0]),
                                         int(row['ranges1'].split(",")[0].split("-")[1]),
                                         args), axis=1)
            data_df['sentence2'] = data_df.apply(
                lambda row: include_tags(row['sentence2'], int(row['ranges2'].split(",")[0].split("-")[0]),
                                         int(row['ranges2'].split(",")[0].split("-")[1]),
                                         args), axis=1)
        else:
            data_df['sentence1'] = data_df.apply(
                lambda row: include_tags(row['sentence1'], row['start1'], row['end1'], args), axis=1)
            data_df['sentence2'] = data_df.apply(
                lambda row: include_tags(row['sentence2'], row['start2'], row['end2'], args), axis=1)
    return data_df


def read_data(data_path, annotation_path, args, cross_Lingual=False):
    data_df = pd.read_json(data_path, orient='records')
    annotation_data = pd.read_json(annotation_path, orient='records')

    complete_df = data_df.merge(annotation_data, left_on='id', right_on='id', how='left')

    if args["tagging"]:
        if cross_Lingual:
            complete_df['sentence1'] = complete_df.apply(
                lambda row: include_tags(row['sentence1'], int(row['ranges1'].split(",")[0].split("-")[0]),
                                         int(row['ranges1'].split(",")[0].split("-")[1]),
                                         args), axis=1)
            complete_df['sentence2'] = complete_df.apply(
                lambda row: include_tags(row['sentence2'], int(row['ranges2'].split(",")[0].split("-")[0]),
                                         int(row['ranges2'].split(",")[0].split("-")[1]),
                                         args), axis=1)
        else:
            complete_df['sentence1'] = complete_df.apply(
                lambda row: include_tags(row['sentence1'], row['start1'], row['end1'], args), axis=1)
            complete_df['sentence2'] = complete_df.apply(
                lambda row: include_tags(row['sentence2'], row['start2'], row['end2'], args), axis=1)
    return complete_df


def include_tags(sentence, start, end, args):
    return sentence[:start] + args["begin_tag"] + " " + sentence[start:end] + " " + args["end_tag"] + sentence[end:]
