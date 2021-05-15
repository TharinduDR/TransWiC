import os
from multiprocessing import cpu_count

from examples.monolingual.en_en.transformer_config import BASE_PATH_EN

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, "data")
DATA_DIRECTORY_EN = os.path.join(BASE_PATH_EN, "data")

TEMP_DIRECTORY = "temp/data"

MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-large"

LANGUAGES = ["en", "ar", "fr", "ru", "zh"]  # only used by transformer_combined
TRAIN_N = None  # Only used by transformer. If int is given top TRAIN_N training instances will be considered.

transformer_config = {
    'output_dir': 'temp/outputs/',
    "best_model_dir": "temp/outputs/best_model",
    'cache_dir': 'temp/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 120,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 100,
    'save_steps': 100,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'n_fold': 3,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": False,
    'evaluate_during_training_steps': 100,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,


    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "tagging": False,
    "begin_tag": "<begin>",
    "end_tag": "<end>",
    "merge_type": "cls",  # "cls, "concat", "add", "avg", "entity-pool", "entity-first", "entity-last", "cls-*"
    "special_tags": ["<begin>"],  # Should be either begin_tag or end_tag
    # Need to be provided only for the merge_types: concat, add and avg. For others this will be automatically set.

    "manual_seed": 777,

    "config": {},
    "local_rank": -1,
    "encoding": None,
}




