from sklearn.metrics import recall_score, precision_score, f1_score


def print_information(df, pred_column, real_column, pos_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    labels = set(real_values)

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))

    for label in labels:
        print()
        print("Stat of the {} Class".format(label))
        print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label=label)))
        print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label=label)))
        print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label=label)))

    pos_values = sorted(list(set(df[pos_column].tolist())))

    for pos_value in pos_values:
        temp_df = df.loc[df[pos_column] == pos_value]
        temp_predictions = temp_df[pred_column].tolist()
        temp_real_values = temp_df[real_column].tolist()
        print ()
        print("Stat of the {} value".format(pos_value))
        print("Weighted Recall {}".format(recall_score(temp_real_values, temp_predictions, average='weighted')))
        print("Weighted Precision {}".format(precision_score(temp_real_values, temp_predictions, average='weighted')))
        print("Weighter F1 Score {}".format(f1_score(temp_real_values, temp_predictions, average='weighted')))

        for label in labels:
            print()
            print("Stat of the {} Class".format(label))
            print("Recall {}".format(recall_score(temp_real_values, temp_predictions, labels=labels, pos_label=label)))
            print("Precision {}".format(precision_score(temp_real_values, temp_predictions, labels=labels, pos_label=label)))
            print("F1 Score {}".format(f1_score(temp_real_values, temp_predictions, labels=labels, pos_label=label)))



