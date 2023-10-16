import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from examples.common.evaluation import weighted_f1, macro_f1, weighted_recall, weighted_precision, cls_report
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_data
from examples.monolingual.other.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE, DATA_DIRECTORY_EN
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='''evaluates multiple models  ''')
parser.add_argument('--wandb_api_key', required=False, help='wandb api key', default=None)
arguments = parser.parse_args()

if arguments.wandb_api_key is not None:
    os.environ['WANDB_API_KEY'] = arguments.wandb_api_key

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

random.seed(transformer_config['manual_seed'])
np.random.seed(transformer_config['manual_seed'])
torch.manual_seed(transformer_config['manual_seed'])

LANGUAGES = ["en", "ar", "fr", "ru", "zh"]


class TestInstance:
    def __init__(self, lang, df, sentence_pairs, preds):
        self.lang = lang
        self.df = df
        self.sentence_pairs = sentence_pairs
        self.preds = preds


def combine_train_data(train_lst, transformer_config, i):
    train_df = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
    eval_df = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])

    for lst in train_lst:
        temp_train_df, temp_eval_df = train_test_split(lst, test_size=0.1,
                                                       random_state=transformer_config['manual_seed'] * i)
        train_df = pd.concat([train_df, temp_train_df], ignore_index=True)
        eval_df = pd.concat([eval_df, temp_eval_df], ignore_index=True)

    # shuffle data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_df['labels'] = encode(train_df["labels"])
    eval_df = eval_df.sample(frac=1).reset_index(drop=True)
    eval_df['labels'] = encode(eval_df["labels"])

    return train_df, eval_df


# train = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
train_lst = []
test_instances = dict()

for lang in LANGUAGES:
    if lang == "en":
        temp_train = read_data(os.path.join(DATA_DIRECTORY_EN, f"training.{lang}-{lang}.data"),
                               os.path.join(DATA_DIRECTORY_EN, f"training.{lang}-{lang}.gold"),
                               args=transformer_config)
        temp_dev = read_data(os.path.join(DATA_DIRECTORY_EN, f"dev.{lang}-{lang}.data"),
                             os.path.join(DATA_DIRECTORY_EN, f"dev.{lang}-{lang}.gold"),
                             args=transformer_config)
        temp_train = pd.concat([temp_train, temp_dev], ignore_index=True)

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
    # train = pd.concat([train, temp_train], ignore_index=True)
    train_lst.append(temp_train)

    test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
    test_preds = np.zeros((len(test), transformer_config["n_fold"]))
    test_instances[lang] = TestInstance(lang, test, test_sentence_pairs, test_preds)

if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        for i in range(transformer_config["n_fold"]):
            transformer_config['wandb_kwargs'] = {
                'group': f'all_{MODEL_NAME}_{transformer_config["strategy"]}_{transformer_config["merge_type"]}',
                'job_type': str(i)}

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                      args=transformer_config)
            train_df, eval_df = combine_train_data(train_lst, transformer_config, i)

            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              weighted_r=weighted_recall, weighted_p=weighted_precision,
                              accuracy=sklearn.metrics.accuracy_score, cls_report=cls_report)

            model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                      use_cuda=torch.cuda.is_available(), args=transformer_config)

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
        transformer_config['wandb_kwargs'] = {
            'name': f'all_{MODEL_NAME}_{transformer_config["strategy"]}_{transformer_config["merge_type"]}'}

        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                  args=transformer_config)
        train_df, eval_df = combine_train_data(train_lst, transformer_config, 1)

        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                          weighted_r=weighted_recall, weighted_p=weighted_precision,
                          accuracy=sklearn.metrics.accuracy_score, cls_report=cls_report)

        model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                  use_cuda=torch.cuda.is_available(), args=transformer_config)

        for k in test_instances.keys():
            test_predictions, test_raw_outputs = model.predict(test_instances[k].sentence_pairs)
            test_instances[k].df['predictions'] = test_predictions

        del model

else:
    model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                              args=transformer_config)
    train = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
    for lst in train_lst:
        train = pd.concat([train, lst], ignore_index=True)

    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      weighted_r=weighted_recall, weighted_p=weighted_precision,
                      accuracy=sklearn.metrics.accuracy_score, cls_report=cls_report)

    for k in test_instances.keys():
        test_predictions, test_raw_outputs = model.predict(test_instances[k].sentence_pairs)
        test_instances[k].df['predictions'] = test_predictions

    del model

# evaluate test data
for k in test_instances.keys():
    print(f'\n Evaluating {k}')
    test_instances[k].df['tag'] = decode(test_instances[k].df['predictions'])
    print_information(test_instances[k].df, "tag", "labels", "pos",
                      eval_file_path=os.path.join(transformer_config['best_model_dir'], f'test_eval_{k}.txt'))

    test_temp = test_instances[k].df[['id', 'tag']]
    test_temp.to_json(os.path.join(TEMP_DIRECTORY, f"test.{k}-{k}"), orient='records')
