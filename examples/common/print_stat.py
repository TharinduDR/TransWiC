from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def print_information(df, pred_column, real_column, pos_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    print()
    print("Accuracy {}".format(accuracy_score(real_values, predictions)))
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))

    pos_values = sorted(list(set(df[pos_column].tolist())))

    for pos_value in pos_values:
        temp_df = df.loc[df[pos_column] == pos_value]
        temp_predictions = temp_df[pred_column].tolist()
        temp_real_values = temp_df[real_column].tolist()
        print ()
        print("Stat of the {} value".format(pos_value))
        print("Accuracy {}".format(accuracy_score(temp_real_values, temp_predictions)))
        print("Weighted Recall {}".format(recall_score(temp_real_values, temp_predictions, average='weighted')))
        print("Weighted Precision {}".format(precision_score(temp_real_values, temp_predictions, average='weighted')))
        print("Weighted F1 Score {}".format(f1_score(temp_real_values, temp_predictions, average='weighted')))
        print("Macro F1 Score {}".format(f1_score(temp_real_values, temp_predictions, average='macro')))


