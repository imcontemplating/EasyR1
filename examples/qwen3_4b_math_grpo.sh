#!/bin/bash

set -x

MODEL_PATH=./models/Qwen3-4B-Base  # replace it with your local file path
DATA_PATH=./data/verl_data.json

#python3 -m verl.trainer.main \
#    config=examples/config.yaml \
#    data.max_response_length=4096 \
#    worker.actor.model.model_path=${MODEL_PATH} \
#    trainer.experiment_name=qwen3_4b_math_grpo

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=1024 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_4b_r_few_math \
    data.train_files=${DATA_PATH} \
    data.val_files=${DATA_PATH} \
    reward_model.reward_manager=naive \
    algorithm.adv_estimator=grpo