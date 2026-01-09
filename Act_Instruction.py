ray stop
pkill -9 -f "python3"

export NCCL_P2P_DISABLE=1   
export WANDB_MODE=disabled

python3 -m verl.trainer.main \
    config=examples/config_0.5b.yaml \
    algorithm.adv_estimator=grpo \
    trainer.max_steps=10 \
    trainer.n_gpus_per_node=2 \
    worker.actor.strategy=fsdp \
    worker.critic.strategy=fsdp \
    worker.rollout.name=vllm \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=examples/reward_function/math.py \
    worker.critic.model.model_path=/lambda/nfs/kl-scaling-project-vir/EasyR1/models/Qwen2-0.5B \
    worker.critic.model.tokenizer_path=/lambda/nfs/kl-scaling-project-vir/EasyR1/models/Qwen2-0.5B