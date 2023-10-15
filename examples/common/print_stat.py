from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report


def print_information(df, pred_column, real_column, pos_column, eval_file_path=None):
    f = None
    if eval_file_path is not None:
        f = open(eval_file_path, "w")

    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    cl_report = classification_report(real_values, predictions, digits=4)
    print("Classification report:\n")
    print(cl_report)
    if eval_file_path is not None:
        f.write("Default classification report:\n")
        f.write("{}\n\n".format(cl_report))

    # result = {
    #     "Weighted Recall": recall_score(real_values, predictions, average='weighted'),
    #     "Weighted Precision": precision_score(real_values, predictions, average='weighted'),
    #     "Weighted F1": f1_score(real_values, predictions, average='weighted')
    # }
    #
    # print()
    # print("Accuracy {}".format(accuracy_score(real_values, predictions)))
    # print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    # print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    # print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))
    #
    # print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))

    pos_values = sorted(list(set(df[pos_column].tolist())))

    for pos_value in pos_values:
        f.write("{}\n".format(pos_value))
        temp_df = df.loc[df[pos_column] == pos_value]
        temp_predictions = temp_df[pred_column].tolist()
        temp_real_values = temp_df[real_column].tolist()
        # print ()
        # print("Stat of the {} value".format(pos_value))
        # print("Accuracy {}".format(accuracy_score(temp_real_values, temp_predictions)))
        # print("Weighted Recall {}".format(recall_score(temp_real_values, temp_predictions, average='weighted')))
        # print("Weighted Precision {}".format(precision_score(temp_real_values, temp_predictions, average='weighted')))
        # print("Weighted F1 Score {}".format(f1_score(temp_real_values, temp_predictions, average='weighted')))
        # print("Macro F1 Score {}".format(f1_score(temp_real_values, temp_predictions, average='macro')))

        result = {
            "Accuracy": accuracy_score(temp_real_values, temp_predictions),
            "Weighted Recall": recall_score(temp_real_values, temp_predictions, average='weighted'),
            "Weighted Precision": precision_score(temp_real_values, temp_predictions, average='weighted'),
            "Weighted F1": f1_score(temp_real_values, temp_predictions, average='weighted'),
            "Macro Recall": recall_score(temp_real_values, temp_predictions, average='macro'),
            "Macro Precision": precision_score(temp_real_values, temp_predictions, average='macro'),
            "Macro F1": f1_score(temp_real_values, temp_predictions, average='macro')
        }

        for key in result.keys():
            print(f'{key} = {result[key]}')
            if eval_file_path is not None:
                f.write(f"{key} = {str(result[key])}\n")

    if eval_file_path is not None:
        f.close()