import csv
import os
import shutil

from sklearn.model_selection import train_test_split
from torch.testing._internal.common_utils import SEED

from examples.common.reader import read_training_file
from examples.monolingual.en_en.siamese_transformer_config import DATA_DIRECTORY, TEMP_DIRECTORY, \
    siamese_transformer_config, MODEL_NAME
from transwic.algo.siamese_transformer import models

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

train = read_training_file(os.path.join(DATA_DIRECTORY, "training.en-en.data"), os.path.join(DATA_DIRECTORY, "training.en-en.gold"))
dev = read_training_file(os.path.join(DATA_DIRECTORY, "trial.en-en.data"), os.path.join(DATA_DIRECTORY, "trial.en-en.gold"))

if siamese_transformer_config["evaluate_during_training"]:
    if siamese_transformer_config["n_fold"] > 0:
        dev_preds = np.zeros((len(dev), siamese_transformer_config["n_fold"]))
        test_preds = np.zeros((len(test), siamese_transformer_config["n_fold"]))
        for i in range(siamese_transformer_config["n_fold"]):

            if os.path.exists(siamese_transformer_config['best_model_dir']) and os.path.isdir(
                    siamese_transformer_config['best_model_dir']):
                shutil.rmtree(siamese_transformer_config['best_model_dir'])

            if os.path.exists(siamese_transformer_config['cache_dir']) and os.path.isdir(
                    siamese_transformer_config['cache_dir']):
                shutil.rmtree(siamese_transformer_config['cache_dir'])

            os.makedirs(siamese_transformer_config['cache_dir'])

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            train_df.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "train_df.tsv"), header=True, sep='\t',
                            index=False, quoting=csv.QUOTE_NONE)
            eval_df.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "eval_df.tsv"), header=True, sep='\t',
                           index=False, quoting=csv.QUOTE_NONE)
            dev.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "dev_df.tsv"), header=True, sep='\t',
                       index=False, quoting=csv.QUOTE_NONE)


            word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=siamese_transformer_config[
                'max_seq_length'])

            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
        #
        #     model = SiameseTransQuestModel(modules=[word_embedding_model, pooling_model])
        #     train_data = SentencesDataset(sts_reader.get_examples('train.tsv'), model)
        #     train_dataloader = DataLoader(train_data, shuffle=True,
        #                                   batch_size=siamese_transformer_config['train_batch_size'])
        #     train_loss = losses.CosineSimilarityLoss(model=model)
        #
        #     eval_data = SentencesDataset(examples=sts_reader.get_examples('eval_df.tsv'), model=model)
        #     eval_dataloader = DataLoader(eval_data, shuffle=False,
        #                                  batch_size=siamese_transformer_config['train_batch_size'])
        #     evaluator = EmbeddingSimilarityEvaluator(eval_dataloader)
        #
        #     warmup_steps = math.ceil(
        #         len(train_data) * siamese_transformer_config["num_train_epochs"] / siamese_transformer_config[
        #             'train_batch_size'] * 0.1)
        #
        #     model.fit(train_objectives=[(train_dataloader, train_loss)],
        #               evaluator=evaluator,
        #               epochs=siamese_transformer_config['num_train_epochs'],
        #               evaluation_steps=100,
        #               optimizer_params={'lr': siamese_transformer_config["learning_rate"],
        #                                 'eps': siamese_transformer_config["adam_epsilon"],
        #                                 'correct_bias': False},
        #               warmup_steps=warmup_steps,
        #               output_path=siamese_transformer_config['best_model_dir'])
        #
        #     model = SiameseTransQuestModel(siamese_transformer_config['best_model_dir'])
        #
        #     dev_data = SentencesDataset(examples=sts_reader.get_examples("dev.tsv"), model=model)
        #     dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=8)
        #     evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
        #     model.evaluate(evaluator,
        #                    result_path=os.path.join(siamese_transformer_config['cache_dir'], "dev_result.txt"))
        #
        #     test_data = SentencesDataset(examples=sts_reader.get_examples("test.tsv", test_file=True), model=model)
        #     test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
        #     evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
        #     model.evaluate(evaluator,
        #                    result_path=os.path.join(siamese_transformer_config['cache_dir'], "test_result.txt"),
        #                    verbose=False)
        #
        #     with open(os.path.join(siamese_transformer_config['cache_dir'], "dev_result.txt")) as f:
        #         dev_preds[:, i] = list(map(float, f.read().splitlines()))
        #
        #     with open(os.path.join(siamese_transformer_config['cache_dir'], "test_result.txt")) as f:
        #         test_preds[:, i] = list(map(float, f.read().splitlines()))
        #
        # dev['predictions'] = dev_preds.mean(axis=1)
        # test['predictions'] = test_preds.mean(axis=1)