import os
import shutil

import numpy as np
import sklearn
import torch
from sklearn.model_selection import train_test_split

from examples.common.config_validator import validate_transformer_config
from examples.common.evaluation import weighted_f1, macro_f1
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_training_file, read_test_file
from examples.monolingual.en_en.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

train = read_training_file(os.path.join(DATA_DIRECTORY, "training.en-en.data"),
                           os.path.join(DATA_DIRECTORY, "training.en-en.gold"), args=transformer_config)

dev = read_training_file(os.path.join(DATA_DIRECTORY, "dev.en-en.data"), os.path.join(DATA_DIRECTORY, "dev.en-en.gold"),
                         args=transformer_config)

test = read_test_file(os.path.join(DATA_DIRECTORY, "test.en-en.data"), args=transformer_config)

train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
train = train[['text_a', 'text_b', 'labels']]

dev = dev.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b'}).dropna()

train['labels'] = encode(train["labels"])
dev['labels'] = encode(dev["labels"])

dev_sentence_pairs = list(map(list, zip(dev['text_a'].to_list(), dev['text_b'].to_list())))
test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

# validate  configs
transformer_config = validate_transformer_config(transformer_config, has_text_b=True)

if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        dev_preds = np.zeros((len(dev), transformer_config["n_fold"]))
        test_preds = np.zeros((len(test), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                      args=transformer_config, merge_type=transformer_config['merge_type'],
                                      merge_n=transformer_config['merge_n'])

            train_df, eval_df = train_test_split(train, test_size=0.1,
                                                 random_state=transformer_config['manual_seed'] * i)

            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              accuracy=sklearn.metrics.accuracy_score)

            model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                      use_cuda=torch.cuda.is_available(), args=transformer_config,
                                      merge_type=transformer_config['merge_type'],
                                      merge_n=transformer_config['merge_n'])

            predictions, raw_outputs = model.predict(dev_sentence_pairs)
            test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)

            dev_preds[:, i] = predictions
            test_preds[:, i] = test_predictions

        final_predictions = []
        for row in dev_preds:
            row = row.tolist()
            final_predictions.append(int(max(set(row), key=row.count)))
        dev['predictions'] = final_predictions

        final_test_predictions = []
        for row in test_preds:
            row = row.tolist()
            final_test_predictions.append(int(max(set(row), key=row.count)))
        test['predictions'] = final_test_predictions

    else:
        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                  args=transformer_config, merge_type=transformer_config['merge_type'],
                                  merge_n=transformer_config['merge_n'])
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=transformer_config['manual_seed'])

        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                          accuracy=sklearn.metrics.accuracy_score)

        model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                  use_cuda=torch.cuda.is_available(), args=transformer_config,
                                  merge_type=transformer_config['merge_type'], merge_n=transformer_config['merge_n'])
        predictions, raw_outputs = model.predict(dev_sentence_pairs)
        test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
        dev['predictions'] = predictions
        test['predictions'] = test_predictions

else:
    model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                              args=transformer_config, merge_type=transformer_config['merge_type'],
                              merge_n=transformer_config['merge_n'])
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(dev_sentence_pairs)
    test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
    dev['predictions'] = predictions
    test['predictions'] = test_predictions

dev['predictions'] = decode(dev['predictions'])
dev['labels'] = decode(dev['labels'])
print_information(dev, "predictions", "labels", "pos")

test['tag'] = decode(test['predictions'])


test = test[['id', 'tag']]
test.to_json(os.path.join(TEMP_DIRECTORY, transformer_config['output_dir'], 'test.en-en.labels'), orient='records', lines=True)
