import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from examples.common.config_validator import validate_transformer_config
from examples.common.evaluation import weighted_f1, macro_f1, weighted_recall, weighted_precision, cls_report
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_data
from examples.monolingual.en_en.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE
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

if __name__ == '__main__':
    train = read_data(os.path.join(DATA_DIRECTORY, "training.en-en.data"),
                      os.path.join(DATA_DIRECTORY, "training.en-en.gold"), args=transformer_config)
    dev = read_data(os.path.join(DATA_DIRECTORY, "dev.en-en.data"), os.path.join(DATA_DIRECTORY, "dev.en-en.gold"),
                    args=transformer_config)
    # combine train and dev
    # train = train.append(dev)
    train = pd.concat([train, dev], ignore_index=True)

    test = read_data(os.path.join(DATA_DIRECTORY, "test.en-en.data"), os.path.join(DATA_DIRECTORY, "test.en-en.gold"),
                     args=transformer_config)

    train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
    train = train[['text_a', 'text_b', 'labels']]

    test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()

    train['labels'] = encode(train["labels"])
    test['labels'] = encode(test["labels"])

    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))


    if transformer_config["evaluate_during_training"]:
        if transformer_config["n_fold"] > 1:
            test_preds = np.zeros((len(test), transformer_config["n_fold"]))
            for i in range(transformer_config["n_fold"]):
                transformer_config['wandb_kwargs'] = {
                    'group': f'en-en_{MODEL_NAME}_{transformer_config["strategy"]}_{transformer_config["merge_type"]}', 'job_type': str(i)}

                if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                    shutil.rmtree(transformer_config['output_dir'])

                model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                          args=transformer_config)

                train_df, eval_df = train_test_split(train, test_size=0.1,
                                                     random_state=transformer_config['manual_seed'] * i)

                model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              weighted_r=weighted_recall, weighted_p=weighted_precision,
                              accuracy=sklearn.metrics.accuracy_score, cls_report=cls_report)

                model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                          use_cuda=torch.cuda.is_available(), args=transformer_config)

                test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
                test_preds[:, i] = test_predictions

            final_test_predictions = []
            for row in test_preds:
                row = row.tolist()
                final_test_predictions.append(int(max(set(row), key=row.count)))
            test['predictions'] = final_test_predictions

        else:
            transformer_config['wandb_kwargs'] = {
                'name': f'en-en_{MODEL_NAME}_{transformer_config["strategy"]}_{transformer_config["merge_type"]}'}

            model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                      args=transformer_config)

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=transformer_config['manual_seed'])

            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              weighted_r=weighted_recall, weighted_p=weighted_precision,
                              accuracy=sklearn.metrics.accuracy_score, cls_report=cls_report)

            model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                      use_cuda=torch.cuda.is_available(), args=transformer_config)

            test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
            test['predictions'] = test_predictions
    else:
        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                  args=transformer_config,)
        model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              weighted_r=weighted_recall, weighted_p=weighted_precision,
                              accuracy=sklearn.metrics.accuracy_score, cls_report=cls_report)

        test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
        test['predictions'] = test_predictions

    test['tag'] = decode(test['predictions'])
    test['labels'] = decode(test['labels'])
    print_information(test, "tag", "labels", "pos", eval_file_path=os.path.join(transformer_config['best_model_dir'], 'test_eval.txt'))

    test = test[['id', 'tag']]
    test.to_json(os.path.join(TEMP_DIRECTORY, 'test.en-en'), orient='records')
