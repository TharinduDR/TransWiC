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
    transformer_config, MODEL_NAME, MODEL_TYPE, language_modeling_args, LANGUAGE_FINETUNE
from transwic.algo.transformer.language_modeling_model import LanguageModelingModel
from transwic.algo.transformer.monotranswic import MonoTransWiCModel


if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

train = read_training_file(os.path.join(DATA_DIRECTORY, "training.en-en.data"), os.path.join(DATA_DIRECTORY, "training.en-en.gold"), transformer_config)
dev = read_training_file(os.path.join(DATA_DIRECTORY, "dev.en-en.data"), os.path.join(DATA_DIRECTORY, "dev.en-en.gold"), transformer_config)

train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
train = train[['text_a', 'text_b', 'labels']]

dev = dev.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()

train['labels'] = encode(train["labels"])
dev['labels'] = encode(dev["labels"])

dev_sentence_pairs = list(map(list, zip(dev['text_a'].to_list(), dev['text_b'].to_list())))


if LANGUAGE_FINETUNE:
    train_list = train['text_a'].tolist() + train['text_b'].tolist()
    dev_list = dev['text_a'].tolist() + dev['text_b'].tolist()
    complete_list = train_list + dev_list
    lm_train = complete_list[0: int(len(complete_list)*0.8)]
    lm_test = complete_list[-int(len(complete_list)*0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    MODEL_NAME = language_modeling_args["best_model_dir"]



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

            predictions, raw_outputs = model.predict(dev_sentence_pairs)

            dev_preds[:, i] = predictions

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
        predictions, raw_outputs = model.predict(dev_sentence_pairs)
        dev['predictions'] = predictions

else:
    model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                              args=transformer_config)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(dev_sentence_pairs)
    dev['predictions'] = predictions


dev['predictions'] = decode(dev['predictions'])
dev['labels'] = decode(dev['labels'])

print_information(dev, "predictions", "labels", "pos")