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
from examples.monolingual.en_en.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_NAME, MODEL_TYPE
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

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
    train = train.append(dev)

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

                test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
                test_preds[:, i] = test_predictions

            final_test_predictions = []
            for row in test_preds:
                row = row.tolist()
                final_test_predictions.append(int(max(set(row), key=row.count)))
            test['predictions'] = final_test_predictions

        else:
            model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                      args=transformer_config)

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=transformer_config['manual_seed'])

            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              accuracy=sklearn.metrics.accuracy_score)

            model = MonoTransWiCModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=2,
                                      use_cuda=torch.cuda.is_available(), args=transformer_config)

            test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
            test['predictions'] = test_predictions
    else:
        model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                  args=transformer_config,)
        model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)

        test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
        test['predictions'] = test_predictions

    test['tag'] = decode(test['predictions'])
    test['labels'] = decode(test['labels'])
    print_information(test, "tag", "labels", "pos")

    test = test[['id', 'tag']]
    test.to_json(os.path.join(TEMP_DIRECTORY, 'test.en-en'), orient='records')
