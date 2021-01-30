import os

import torch

from examples.common.config_validator import validate_transformer_config
from examples.common.label_converter import decode
from examples.common.reader import read_test_file
from examples.monolingual.en_en.transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    transformer_config, MODEL_TYPE, MODEL_NAME
from transwic.algo.transformer.monotranswic import MonoTransWiCModel

model = MonoTransWiCModel(MODEL_TYPE, MODEL_NAME, num_labels=2,
                          use_cuda=torch.cuda.is_available(), args=transformer_config,
                          merge_type=transformer_config['merge_type'],
                          merge_n=transformer_config['merge_n'])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

data_config = {"test.en-ar.data": "test.en-ar",
                "test.en-fr.data": "test.en-fr",
                "test.en-ru.data": "test.en-ru",
                "test.en-zh.data": "test.en-zh",
               }

test = read_test_file(os.path.join(DATA_DIRECTORY, "test.en-en.data"), args=transformer_config)

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

# validate  configs
transformer_config = validate_transformer_config(transformer_config, has_text_b=True)



test_predictions, test_raw_outputs = model.predict(test_sentence_pairs)
test['predictions'] = test_predictions
test['tag'] = decode(test['predictions'])
test = test[['id', 'tag']]
test.to_json(os.path.join(TEMP_DIRECTORY, 'test.en-en'), orient='records')
