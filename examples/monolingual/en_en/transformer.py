import os
import shutil

import sklearn
import torch
import numpy as np

from sklearn.model_selection import train_test_split

from examples.common.evaluation import weighted_f1, macro_f1
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_training_file
from examples.monolingual.en_en.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE
from transwic.algo.transformer.monotranswic import MonoTransWiCModel


if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

train = read_training_file(os.path.join(DATA_DIRECTORY, "training.en-en.data"), os.path.join(DATA_DIRECTORY, "training.en-en.gold"))
dev = read_training_file(os.path.join(DATA_DIRECTORY, "dev.en-en.data"), os.path.join(DATA_DIRECTORY, "dev.en-en.gold"))

train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
train = train[['text_a', 'text_b', 'labels']]

dev = dev.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
dev = dev[['text_a', 'text_b', 'labels']]

train['labels'] = encode(train["labels"])
dev['labels'] = encode(dev["labels"])

if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        dev_preds = np.zeros((len(dev), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                      args=transformer_config)

            train_df, eval_df = train_test_split(train, test_size=0.1,
                                                 random_state=transformer_config['manual_seed'] * i)

            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              accuracy=sklearn.metrics.accuracy_score)

            model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                      use_cuda=torch.cuda.is_available(), args=transformer_config)

            result, model_outputs, wrong_predictions = model.eval_model(dev, macro_f1=macro_f1, weighted_f1=weighted_f1,
                                                                        accuracy=sklearn.metrics.accuracy_score)

            dev_preds[:, i] = model_outputs

        final_predictions = []
        for row in dev_preds:
            row = row.tolist()
            final_predictions.append(int(max(set(row), key=row.count)))
        dev['predictions'] = final_predictions

    else:
        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=transformer_config['manual_seed'])

        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                          accuracy=sklearn.metrics.accuracy_score)

        model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                  use_cuda=torch.cuda.is_available(), args=transformer_config)
        result, model_outputs, wrong_predictions = model.eval_model(dev, macro_f1=macro_f1, weighted_f1=weighted_f1,
                                                                    accuracy=sklearn.metrics.accuracy_score)
        dev['predictions'] = model_outputs

else:
    model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                              args=transformer_config)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    result, model_outputs, wrong_predictions = model.eval_model(dev, macro_f1=macro_f1, weighted_f1=weighted_f1,
                                                                accuracy=sklearn.metrics.accuracy_score)
    dev['predictions'] = model_outputs


dev['predictions'] = decode(dev['predictions'])
dev['labels'] = decode(dev['labels'])

print_information(dev, "predictions", "labels")