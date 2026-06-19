# The Constraint Cliff: How KL Penalty Decay Schedules Govern Training Stability in RLHF

This repository contains the implementation for our paper *The Constraint Cliff: How KL Penalty Decay Schedules Govern Training Stability in RLHF*. We modify the [EasyR1](https://github.com/hiyouga/EasyR1) framework to support **dynamic KL penalty scaling** in GRPO-based reinforcement learning, and provide all code necessary to reproduce our experiments.

---

## Overview

Standard RLHF pipelines apply a constant KL divergence penalty throughout training. We investigate whether time-varying penalty schedules can improve the trade-off between exploration and stability. We evaluate four schedules:

| Schedule | $\alpha(t)$ | Effective penalty at step 500 |
|:---|:---|:---|
| No Decay (baseline) | $1$ | 100% |
| Logarithmic | $1/\ln(t+1)$ | ≈ 16.1% |
| Square Root | $1/\sqrt{t}$ | ≈ 4.5% |
| Linear | $1/t$ | ≈ 0.2% |

**Key findings:**

- **Logarithmic decay** achieves the highest validation accuracy (0.462 vs. 0.452 baseline) while maintaining stable KL dynamics — a zero-cost improvement.
- **Polynomial decays** (Linear and Square Root) both degrade performance below the constant baseline, exhibiting chaotic KL divergence spikes.
- We introduce the **Constraint Cliff** framework to explain this dichotomy: a critical transition beyond which policy divergence exceeds the reward model's reliable region, triggering self-reinforcing collapse.

---

## Implementation

The modification consists of a single function override. No other changes to the EasyR1 codebase are required.

### 1. Configuration

Added `kl_scaling` to `AlgorithmConfig`:

```python
# verl/trainer/config.py
@dataclass
class AlgorithmConfig:
    # ... existing parameters ...
    kl_scaling: str = "none"  # options: none, linear, sqrt, log
```

```yaml
# examples/config.yaml
algorithm:
  kl_scaling: log  # options: none, linear, sqrt, log
```

### 2. Scaling Logic

Modified `apply_kl_penalty` in `verl/trainer/ray_trainer.py` to compute a dynamic scaling factor $\alpha(t)$ at each training step:

```python
import math

# Inside apply_kl_penalty function
if kl_scaling == "none" or global_step <= 1:
    scale = 1.0
elif kl_scaling == "linear":
    scale = 1.0 / global_step
elif kl_scaling == "sqrt":
    scale = 1.0 / math.sqrt(global_step)
elif kl_scaling == "log":
    scale = 1.0 / math.log(global_step + 1)
else:
    scale = 1.0

# Apply scale to the standard KL penalty
data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld * scale
```

### 3. Trainer Integration

Updated the training loop to pass step count and scaling mode:

```python
# Inside trainer/ray_trainer.py (fit function)
batch, kl_metrics = apply_kl_penalty(
    batch,
    self.kl_ctrl,
    self.config.algorithm.kl_penalty,
    global_step=self.global_step,
    kl_scaling=self.config.algorithm.kl_scaling
)
```

---

## Reproducing the Experiments

### Setup

```bash
git clone https://github.com/imcontemplating/EasyR1.git
cd EasyR1
pip install -e .
```

### Training

Run all four schedule variants:

```bash
EXPERIMENTS=("none" "linear" "sqrt" "log")

for MODE in "${EXPERIMENTS[@]}"; do
    python3 -m verl.trainer.main \
        config=examples/config.yaml \
        algorithm.adv_estimator=grpo \
        trainer.max_steps=500 \
        trainer.n_gpus_per_node=2 \
        trainer.logger='["console", "file"]' \
        trainer.save_freq=5 \
        trainer.val_freq=5 \
        worker.actor.strategy=fsdp \
        worker.rollout.name=vllm \
        worker.rollout.n=8 \
        worker.rollout.gpu_memory_utilization=0.5 \
        worker.rollout.tensor_parallel_size=1 \
        worker.rollout.enforce_eager=true \
        worker.reward.reward_function=examples/reward_function/math.py \
        worker.reward.reward_function_name=compute_score \
        worker.actor.model.model_path=Qwen/Qwen2.5-1.5B-Instruct \
        worker.critic.model.model_path=Qwen/Qwen2.5-1.5B-Instruct \
        algorithm.kl_scaling=${MODE} \
        algorithm.use_kl_loss=False \
        trainer.experiment_name=qwen2_5_1_5b_math_grpo_${MODE}
done
```

### Key Hyperparameters

| Parameter | Value |
|:---|:---|
| Base model | Qwen2.5-1.5B-Instruct |
| Training data | Math12K (train split) |
| Evaluation data | Math12K (test split) |
| Advantage estimator | GRPO |
| Rollout samples per prompt | 8 |
| KL penalty coefficient $\beta_0$ | 0.01 |
| KL penalty type | Low-variance KL |
| KL controller | Fixed |
| Learning rate | 1e-6 (AdamW) |
| Max prompt / response length | 2048 / 2048 tokens |
| Rollout batch size | 512 |
| Actor global batch size | 128 |
| Training steps | 500 |
| Hardware | 2× A100 (80GB) |

---

## Results

| Schedule | Final Val Acc | Peak Acc | Peak Step | Post-Peak Decline |
|:---|:---|:---|:---|:---|
| <b>Log</b> (1/ln(<i>t</i>+1)) | <b>0.462</b> | 0.560 | 15 | −0.098 |
| No Decay | 0.452 | 0.548 | 15 | −0.096 |
| Linear ($1/t$) | 0.436 | 0.556 | 15 | −0.120 |
| Sqrt ($1/\sqrt{t}$) | 0.430 | 0.544 | 15 | −0.114 |

For detailed analysis, including the theoretical derivation of KL divergence growth bounds and the Constraint Cliff framework, please refer to the paper.

---

## Citation

```bibtex
@misc{shi2026constraint,
  title  = {The Constraint Cliff: How {KL} Penalty Decay Schedules Govern Training Stability in {RLHF}},
  author = {TODO},
  year   = {2026}
}
```

---

## Acknowledgments

This project builds on [EasyR1](https://github.com/hiyouga/EasyR1) by Zheng et al.