import os
from examples.common.reader import read_training_file
from examples.monolingual.en_en.siamese_transformer_config import DATA_PATH

train_df = read_training_file(os.path.join(DATA_PATH, "training.en-en.data"), os.path.join(DATA_PATH, "training.en-en.gold"))
