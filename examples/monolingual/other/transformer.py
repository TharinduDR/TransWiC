import os
import random
import shutil

import numpy as np
import sklearn
import torch
from sklearn.model_selection import train_test_split

from examples.common.config_validator import validate_transformer_config
from examples.common.evaluation import weighted_f1, macro_f1
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_data
from examples.monolingual.other.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE, TRAIN_N
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

random.seed(transformer_config['manual_seed'])
np.random.seed(transformer_config['manual_seed'])
torch.manual_seed(transformer_config['manual_seed'])

data_config = {
    # "ar_ar": ["dev.ar-ar.data", "dev.ar-ar.gold", "test.ar-ar.data", "test.ar-ar.gold", "test.ar-ar"],
    # "fr_fr": ["dev.fr-fr.data", "dev.fr-fr.gold", "test.fr-fr.data", "test.fr-fr.gold", "test.fr-fr"],
    # "ru_ru": ["dev.ru-ru.data", "dev.ru-ru.gold", "test.ru-ru.data", "test.ru-ru.gold", "test.ru-ru"],
    "zh_zh": ["dev.zh-zh.data", "dev.zh-zh.gold", "test.zh-zh.data", "test.zh-zh.gold", "test.zh-zh"],
}

for key, value in data_config.items():

    train = read_data(os.path.join(DATA_DIRECTORY, value[0]), os.path.join(DATA_DIRECTORY, value[1]),
                      args=transformer_config)
    if TRAIN_N is not None:
        if TRAIN_N > len(train):
            raise ValueError(f"No {TRAIN_N} training instances available for {key}")
        train = train.head(TRAIN_N)

    test = read_data(os.path.join(DATA_DIRECTORY, value[2]), os.path.join(DATA_DIRECTORY, value[3]),
                     args=transformer_config)

    train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
    train = train[['text_a', 'text_b', 'labels']]

    test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()

    train['labels'] = encode(train["labels"])
    test['labels'] = encode(test["labels"])

    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

    # validate  configs
    transformer_config = validate_transformer_config(transformer_config, has_text_b=True)

    if transformer_config["evaluate_during_training"]:
        if transformer_config["n_fold"] > 1:
            test_preds = np.zeros((len(test), transformer_config["n_fold"]))
            for i in range(transformer_config["n_fold"]):
                print(f'Started fold {i}')
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

                test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)

                test_preds[:, i] = test_predictions
                model = None
                del model
                torch.cuda.empty_cache()

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
                                      merge_type=transformer_config['merge_type'],
                                      merge_n=transformer_config['merge_n'])

            test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
            test['predictions'] = test_predictions
            model = None
            del model
            torch.cuda.empty_cache()

    else:
        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                  args=transformer_config, merge_type=transformer_config['merge_type'],
                                  merge_n=transformer_config['merge_n'])
        model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
        test['predictions'] = test_predictions
        model = None
        del model
        torch.cuda.empty_cache()

    print(f'\n Evaluating {key}')
    test['tag'] = decode(test['predictions'])
    test['labels'] = decode(test['labels'])
    print_information(test, "tag", "labels", "pos")

    test = test[['id', 'tag']]
    test.to_json(os.path.join(TEMP_DIRECTORY, value[4]), orient='records')
