# EasyR1 with Dynamic KL Scaling 

This repository contains a modified version of **EasyR1**, implementing **Dynamic KL Divergence Scaling** strategies for PPO training.

**Goal:**
The primary objective of this project is to **evaluate and compare four distinct KL penalty scaling schedules** (Constant, Linear, Square Root, and Logarithmic). We aim to **identify the optimal decay strategy** that maximizes the model's reasoning capabilities by allowing efficient exploration, while preventing catastrophic forgetting or alignment drift.

---

##  Implementation Details

To enable dynamic scaling, we modified the core trainer logic to adjust the KL penalty based on the current `global_step`.

### Key Modifications

The following changes were applied to the EasyR1 codebase:

#### 1. Configuration Update
Added a `kl_scaling` parameter to `config.yaml` to select the decay strategy.
```yaml
# config/examples/config.yaml
algorithm:
  kl_scaling: none # options: none, linear, sqrt, log
```

#### 2. Scaling Logic
We modified `apply_kl_penalty` to accept `global_step` and calculate a scaling factor $\alpha$.

**The Scaling Formulas:**
* **None (Baseline):** $\alpha = 1.0$
* **Linear:** $\alpha = \frac{1}{t}$
* **Sqrt:** $\alpha = \frac{1}{\sqrt{t}}$
* **Log:** $\alpha = \frac{1}{\ln(t + 1)}$

**Code Implementation:**
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

#### 3. Trainer Integration
Updated the training loop to pass the current step count to the penalty function:
```python
# Inside trainer/ray_trainer.py (fit function)
batch, kl_metrics = apply_kl_penalty(
    batch,
    self.kl_ctrl,
    self.config.algorithm.kl_penalty,
    global_step=self.global_step,          # Added
    kl_scaling=self.config.algorithm.kl_scaling # Added
)
```

---

##  Experiment Methodology

To validate the hypothesis, we conducted a controlled experiment comparing four different scaling strategies.

### Setup
* **Model:** Qwen 2.5 1.5B Instruct
* **Task:** Mathematical Reasoning (Reward based on correct answer format and accuracy)
* **Algorithm:** GRPO / PPO
* **Training Duration:** 500 Steps per experiment
* **Compute:** 2x GPUs
* **Baseline:** `kl_scaling="none"` (Standard constant KL penalty)

### Variables
We tested the following 4 conditions:
1.  **None:** Control group. KL penalty remains constant.
2.  **Linear:** Moderate decay.
3.  **Sqrt:** Fast decay. Penalty approaches 0 very quickly.
4.  **Log:** Slow decay. Retains some constraint for longer.

---

##  Results

We evaluated four different KL penalty scaling schedules: **None (Baseline), Linear, Sqrt, and Log**. Each variable vas conducted over 500 training steps, and we measured both the final validation accuracy and the performance decline relative to Step 0.

### 1. Key Finding: Log Scaling Performs Best
Our results demonstrate that **Logarithmic scaling** is the optimal strategy, outperforming both the baseline and other scaling variants.

| Variant | Val Accuracy | Decline from Step 100 | Performance |
| :--- | :--- | :--- | :--- |
| **log** | **0.462** | **-0.022** (Smallest) |  **Best** |
| none | 0.452 | -0.038 | Baseline |
| linear | 0.436 | -0.058 | Worse than baseline |
| sqrt | 0.430 | -0.066 | Worse than baseline |

*Table 1: Final validation accuracy and performance stability across scaling strategies.*

### 2. Analysis & Interpretation

* **The "Sweet Spot" of Log Scaling:** Log scaling provided the highest final validation accuracy (**0.462**) and the smallest performance decline (**-0.022**). This suggests that a gentle reduction in KL penalty offers the ideal balance—it provides the model enough freedom to explore and learn, while maintaining sufficient constraint to prevent mode collapse or catastrophic forgetting. This aligns with theory, as $log(n)$ grows slowly, resulting in a gradual penalty reduction.

* **Aggressive Scaling Hurts Performance:** Contrary to the hypothesis that *any* decay is better than none, we found that **Linear** and **Sqrt** scaling actually performed **worse than the baseline**. These strategies reduce the penalty too aggressively, allowing for too much drift and causing the model to lose alignment with the reference policy.

### 3. Comparison to State-of-the-Art (R-FEW)
Our findings offer an interesting contrast to methods like **R-FEW**, which utilizes 5% human anchor data to prevent reward hacking and drift.
* **Data-Free Stability:** We show that **Log scaling** can help prevent drift and maintain stability *without* requiring any additional human anchor data.
* **Complementary Approaches:** Since our method is a purely algorithmic change (modifying the penalty scheduler), it is complementary to data-centric approaches like R-FEW. Future work could combine both methods for potentially superior results.

### 4. Conclusion
This experiment identifies a "free improvement" for PPO training. By implementing a simple, one-line code change to use **Logarithmic KL scaling**, we achieved better performance and stability than the standard constant penalty.

---


