import os

import torch

from examples.common.config_validator import validate_transformer_config
from examples.common.label_converter import decode, encode
from examples.common.print_stat import print_information
from examples.common.reader import read_test_file, read_training_file, read_data
from examples.crosslingual.other.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_TYPE, MODEL_NAME
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)


train = read_training_file(os.path.join("examples/monolingual/en_en/data/", "training.en-en.data"),
                           os.path.join("examples/monolingual/en_en/data/", "training.en-en.gold"), args=transformer_config)

train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
train = train[['text_a', 'text_b', 'labels']]
train['labels'] = encode(train["labels"])

# data_config = {"test.en-ar.data": "test.en-ar",
#                 "test.en-fr.data": "test.en-fr",
#                 "test.en-ru.data": "test.en-ru",
#                 "test.en-zh.data": "test.en-zh",
#                }
data_config = {
    "en_ar": ["test.en-ar.data", "test.en-ar.gold", "test.en-ar"],
    "en_fr": ["test.en-fr.data", "test.en-fr.gold", "test.en-fr"],
    "en_ru": ["test.en-ru.data", "test.en-ru.gold", "test.en-ru"],
    "en_zh": ["test.en-zh.data", "test.en-zh.gold", "test.en-zh"],
}


for key, value in data_config.items():
    test = read_data(os.path.join(DATA_DIRECTORY, value[0]), os.path.join(DATA_DIRECTORY, value[1]),
                     args=transformer_config, cross_Lingual=True)

    test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'tag': 'labels'}).dropna()
    test['labels'] = encode(test["labels"])

    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

    # validate  configs
    transformer_config = validate_transformer_config(transformer_config, has_text_b=True)

    model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2,
                              use_cuda=torch.cuda.is_available(), args=transformer_config,
                              merge_type=transformer_config['merge_type'],
                              merge_n=transformer_config['merge_n'])

    test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
    test['predictions'] = test_predictions
    test['tag'] = decode(test['predictions'])

    print(f'\n Evaluating {key}')
    test['labels'] = decode(test['labels'])
    print_information(test, "tag", "labels", "pos")

    test = test[['id', 'tag']]
    test.to_json(os.path.join(TEMP_DIRECTORY, value[2]), orient='records')

