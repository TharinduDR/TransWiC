import csv
import logging
import math
import os
import shutil

import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from examples.common.reader import read_training_file
from examples.monolingual.en_en.siamese_transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    siamese_transformer_config, MODEL_NAME
from transwic import LoggingHandler
from transwic.algo.siamese_transformer import models, SiameseTransWiC, losses
from transwic.algo.siamese_transformer.datasets import SentencesDataset
from transwic.algo.siamese_transformer.evaluation.binary_classification_evaluator import BinaryClassificationEvaluator
from transwic.algo.siamese_transformer.readers import InputExample

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

train = read_training_file(os.path.join(DATA_DIRECTORY, "training.en-en.data"), os.path.join(DATA_DIRECTORY, "training.en-en.gold"))
dev = read_training_file(os.path.join(DATA_DIRECTORY, "trial.en-en.data"), os.path.join(DATA_DIRECTORY, "trial.en-en.gold"))

if siamese_transformer_config["evaluate_during_training"]:
    if siamese_transformer_config["n_fold"] > 0:
        dev_preds = np.zeros((len(dev), siamese_transformer_config["n_fold"]))
        for i in range(siamese_transformer_config["n_fold"]):

            if os.path.exists(siamese_transformer_config['best_model_dir']) and os.path.isdir(
                    siamese_transformer_config['best_model_dir']):
                shutil.rmtree(siamese_transformer_config['best_model_dir'])

            if os.path.exists(siamese_transformer_config['cache_dir']) and os.path.isdir(
                    siamese_transformer_config['cache_dir']):
                shutil.rmtree(siamese_transformer_config['cache_dir'])

            os.makedirs(siamese_transformer_config['cache_dir'])

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=siamese_transformer_config['manual_seed'] * i)
            train_df.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "train_df.tsv"), header=True, sep='\t',
                            index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
            eval_df.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "eval_df.tsv"), header=True, sep='\t',
                           index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
            dev.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "dev_df.tsv"), header=True, sep='\t',
                       index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

            word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=siamese_transformer_config[
                'max_seq_length'])

            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)

            model = SiameseTransWiC(modules=[word_embedding_model, pooling_model])

            logging.info("Read train dataset")

            label2int = {"F": 0, "T": 1}
            train_samples = []

            train_file_reader = csv.DictReader(open(os.path.join(siamese_transformer_config['cache_dir'], "train_df.tsv")), delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in train_file_reader:
                label_id = label2int[row['tag']]
                train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

            train_dataset = SentencesDataset(train_samples, model=model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=siamese_transformer_config['train_batch_size'])
            train_loss = losses.SoftmaxLoss(model=model,
                                            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                            num_labels=len(label2int))

            eval_samples = []
            eval_file_reader = csv.DictReader(
                open(os.path.join(siamese_transformer_config['cache_dir'], "eval_df.tsv")), delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in eval_file_reader:
                label_id = label2int[row['tag']]
                eval_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

            evaluator = BinaryClassificationEvaluator.from_input_examples(eval_samples, batch_size=siamese_transformer_config['eval_batch_size'],
                                                                             name='eval_result.txt')

            # Configure the training
            num_epochs = siamese_transformer_config['num_train_epochs']

            warmup_steps = math.ceil(
                len(train_dataset) * num_epochs / siamese_transformer_config['train_batch_size'] * 0.1)  # 10% of train data for warm-up
            logging.info("Warmup-steps: {}".format(warmup_steps))

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=siamese_transformer_config['num_train_epochs'],
                      evaluation_steps=siamese_transformer_config["evaluate_during_training_steps"],
                      optimizer_params={'lr': siamese_transformer_config["learning_rate"],
                                        'eps': siamese_transformer_config["adam_epsilon"],
                                        'correct_bias': False},
                      warmup_steps=warmup_steps,
                      output_path=siamese_transformer_config['best_model_dir']
                      )

            model = SiameseTransWiC(siamese_transformer_config['best_model_dir'])

            dev_samples = []
            dev_file_reader = csv.DictReader(
                open(os.path.join(siamese_transformer_config['cache_dir'], "dev_df.tsv")), delimiter='\t',
                quoting=csv.QUOTE_NONE)
            for row in dev_file_reader:
                label_id = label2int[row['tag']]
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

            dev_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples,
                                                                          batch_size=siamese_transformer_config[
                                                                              'eval_batch_size'],
                                                                          name='dev_result.txt')

            model.evaluate(dev_evaluator, output_path=os.path.join(siamese_transformer_config['cache_dir'], "dev_result.txt"))
