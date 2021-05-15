import os
import shutil

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from examples.common.config_validator import validate_transformer_config
from examples.common.evaluation import weighted_f1, macro_f1
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_data
from examples.monolingual.other.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE, LANGUAGES, DATA_DIRECTORY_EN
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)


class TestInstance:
    def __init__(self, lang, df, sentence_pairs, preds):
        self.lang = lang
        self.df = df
        self.sentence_pairs = sentence_pairs
        self.preds = preds


train = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
test_instances = dict()

for lang in LANGUAGES:
    if lang == "en":
        temp_train = read_data(os.path.join(DATA_DIRECTORY_EN, f"training.{lang}-{lang}.data"),
                               os.path.join(DATA_DIRECTORY_EN, f"training.{lang}-{lang}.gold"),
                               args=transformer_config)
        temp_train = temp_train.append(read_data(os.path.join(DATA_DIRECTORY_EN, f"dev.{lang}-{lang}.data"),
                               os.path.join(DATA_DIRECTORY_EN, f"dev.{lang}-{lang}.gold"),
                               args=transformer_config))
        test = read_data(os.path.join(DATA_DIRECTORY_EN, f"test.{lang}-{lang}.data"),
                         os.path.join(DATA_DIRECTORY_EN, f"test.{lang}-{lang}.gold"),
                         args=transformer_config)
    else:
        temp_train = read_data(os.path.join(DATA_DIRECTORY, f"dev.{lang}-{lang}.data"),
                               os.path.join(DATA_DIRECTORY, f"dev.{lang}-{lang}.gold"),
                               args=transformer_config)
        test = read_data(os.path.join(DATA_DIRECTORY, f"test.{lang}-{lang}.data"),
                         os.path.join(DATA_DIRECTORY, f"test.{lang}-{lang}.gold"),
                         args=transformer_config)

    temp_train = temp_train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
    temp_train = temp_train[['text_a', 'text_b', 'labels']]
    train = train.append(temp_train)

    test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
    test_preds = np.zeros((len(test), transformer_config["n_fold"]))
    test_instances[lang] = TestInstance(lang, test, test_sentence_pairs, test_preds)

# shuffle data
train = train.sample(frac=1).reset_index(drop=True)
train['labels'] = encode(train["labels"])

# validate  configs
transformer_config = validate_transformer_config(transformer_config, has_text_b=True)

if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
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

            for k in test_instances.keys():
                test_predictions, test_raw_outputs = model.predict(test_instances[k].sentence_pairs)
                test_instances[k].preds[:, i] = test_predictions

            del model

        for k in test_instances.keys():
            final_test_predictions = []
            for row in test_instances[k].preds:
                row = row.tolist()
                final_test_predictions.append(int(max(set(row), key=row.count)))
            test_instances[k].df['predictions'] = final_test_predictions

    else:
        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                  args=transformer_config, merge_type=transformer_config['merge_type'],
                                  merge_n=transformer_config['merge_n'])
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=transformer_config['manual_seed'])

        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                          accuracy=sklearn.metrics.accuracy_score)

        model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                  use_cuda=torch.cuda.is_available(), args=transformer_config,
                                  merge_type=transformer_config['merge_type'],
                                  merge_n=transformer_config['merge_n'])

        for k in test_instances.keys():
            test_predictions, test_raw_outputs = model.predict(test_instances[k].sentence_pairs)
            test_instances[k].df['predictions'] = test_predictions

        del model

else:
    model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                              args=transformer_config, merge_type=transformer_config['merge_type'],
                              merge_n=transformer_config['merge_n'])
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)

    for k in test_instances.keys():
        test_predictions, test_raw_outputs = model.predict(test_instances[k].sentence_pairs)
        test_instances[k].df['predictions'] = test_predictions

    del model

# evaluate test data
for k in test_instances.keys():
    print(f'\n Evaluating {k}')
    test_instances[k].df['tag'] = decode(test_instances[k].df['predictions'])
    print_information(test_instances[k].df, "tag", "labels", "pos")

    test_temp = test_instances[k].df[['id', 'tag']]
    test_temp.to_json(os.path.join(TEMP_DIRECTORY, f"test.{k}-{k}"), orient='records')
