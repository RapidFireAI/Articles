---
title: "Run 20× More DPO Experiments: Official RapidFire AI Integration for TRL"
thumbnail: /blog/assets/rapidfire-dpo/thumbnail.png
authors:
- user: your-hf-username
---

# Run 20× More DPO Experiments: Official RapidFire AI Integration for TRL

**Hugging Face TRL now officially integrates with RapidFire AI**<sup>[[1]](https://huggingface.co/docs/trl/en/rapidfire_integration)</sup>, bringing massive speedups to Direct Preference Optimization (DPO) workflows. If you're using TRL's `DPOTrainer` to align LLMs with human preferences, you can now run multiple DPO configurations concurrently—even on a single GPU—and compare them in near real-time with **16-24× higher experimentation throughput**. This speedup comes from RapidFire AI's chunk-based scheduling approach, which enables concurrent training of multiple configurations while efficiently stopping underperforming runs early—as detailed in [this technical explanation](https://www.rapidfire.ai/blogs/rapid-experimentation-16-24x-more-throughput-without-extra-gpus).

Experienced LLM teams know that "good enough" models rarely stay good for long. As data drifts and product needs evolve, you must repeatedly realign models to human preferences. Direct Preference Optimization (DPO) provides a clean, efficient way to do that—without training a separate reward model like in PPO. But the real bottleneck isn't the algorithm. It's the experimentation loop: trying variations across base models, adapter capacity, quantization, loss types, and training hyperparameters—before you commit serious GPU budget.

With **RapidFire AI's official TRL integration**, you can now turn slow, sequential DPO experiments into fast, adaptive, multi-configuration runs you can steer in real time. The integration requires minimal code changes and enables teams to compare multiple DPO configurations efficiently.

## What is DPO?

DPO aligns a policy model to human preferences by optimizing a contrastive objective over paired responses to the same prompt: a chosen (preferred) and a rejected (dispreferred) completion. Instead of fitting a separate reward model, DPO computes relative preference using log-probability differences between the policy and a reference model, scaled by a temperature β. In practice, you'll watch signals like rewards for chosen vs. rejected responses, accuracy of preference ordering, and reward margins to assess whether the policy is learning to prefer the right outputs.

<img src="https://raw.githubusercontent.com/RapidFireAI/Articles/main/DPO/DPO-diagram.png" alt="DPO Training Workflow" style="width: 50%;">

*DPO training workflow: Start with preference data (chosen/rejected pairs), initialize policy and reference models from an SFT model, compute DPO loss by comparing policy vs reference log-probabilities, then update the policy to prefer chosen responses.*

## The Problem: Alignment Work That Moves Too Slowly

If you’ve ever staged a DPO pipeline, you’ve probably felt at least one of these:

- Configuration explosion across adapters, losses, β, schedules, batch shapes, and tokenization slows decisions.
- Sequential, GPU-hogging runs hide early comparative signals and delay insight.
- Mid-run pivots are costly; stopping losers and branching winners is hard.

RapidFire AI addresses these issues head-on with adaptive execution, multi-config APIs, a live dashboard, and "Interactive Control Ops" (IC Ops) that let you guide experiments as they learn.

## The Solution: How RapidFire AI Works

RapidFire AI splits your dataset randomly into "chunks" and cycles LLM configurations through the GPUs at chunk boundaries. You get incremental signal on eval metrics across all configs much more quickly. The automatic checkpointing via an efficient shared-memory-based adapter/model spilling/loading mechanism keeps training smooth, stable, and consistent. Use IC Ops to adapt mid-flight to stop low-performers earlier and clone promising ones with tweaked config knobs, optionally warm-starting from the parent's weights.

<img src="https://raw.githubusercontent.com/RapidFireAI/hf-trl-integration-article/main/images/gantt-2gpu.png" alt="GPU Scheduling Comparison" style="width: 60%;">

*Sequential vs. Task Parallel vs. RapidFire AI: The adaptive scheduler maximizes GPU utilization across multiple configs and GPUs. The bottom row shows IC Ops in action—stopping, cloning, and modifying runs mid-flight.*

## Getting Started

The RapidFire AI integration with TRL is designed to be a drop-in enhancement to your existing DPO workflows. Installation takes minutes, and you can start running concurrent experiments immediately.

### Prerequisites

- **Python 3.12.x**
- **NVIDIA GPU** with Compute Capability 7.x or 8.x
- **CUDA Toolkit 11.8+**
- **PyTorch 2.7.1+**

### Installation

```bash
# Install RapidFire AI
pip install rapidfireai

# Authenticate with Hugging Face
huggingface-cli login --token YOUR_TOKEN

# Workaround for current issue: https://github.com/huggingface/xet-core/issues/527
pip uninstall -y hf-xet

# Initialize RapidFire AI
rapidfireai init

# Start the RapidFire AI server
rapidfireai start
```

The dashboard will be available at `http://localhost:3000` where you can monitor and control experiments in real-time.

## Example: Running Multiple DPO Configurations Concurrently

Here's a complete example showing how RapidFire AI integrates with TRL's `DPOTrainer` to run multiple DPO configurations concurrently. The key difference from standard TRL usage is wrapping your configs with RapidFire's multi-config wrappers—everything else stays the same.

This example uses a pre-trained SFT model (`rapidfire-ai-inc/mistral-7b-sft-bnb-4bit`) as the starting point, which is already quantized with 4-bit QLoRA. Starting from an SFT model ensures that the data we train on is in-distribution for the DPO algorithm.

```python
from rapidfireai import Experiment
from rapidfireai.fit.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFDPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType
import torch

# Load a preference dataset (chosen/rejected pairs)
train_dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized", 
    split="train"
).select(range(128))

# Initialize RapidFire experiment
experiment = Experiment(experiment_name="exp1-dpo-alignment", mode="fit")

# Define base LoRA configuration
MODEL_NAME = "rapidfire-ai-inc/mistral-7b-sft-bnb-4bit"

base_lora_config = RFLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64, 
    lora_alpha=64, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", 
)

# Define base DPO configuration
base_dpo_config = RFDPOConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    model_adapter_name="default",  # LoRA adapter to train with DPO
    ref_adapter_name="reference",  # LoRA adapter for reference (unchanged)
    force_use_ref_model=False,  # Use adapter-based reference (memory-efficient) instead of separate full model
    loss_type="sigmoid",  # Bradley-Terry loss
    beta=0.1,             # KL divergence weight
    max_prompt_length=512,
    max_completion_length=512,
    max_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    logging_steps=1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    save_strategy="no",
    output_dir="./output",
)

# Create 3 variants to compare different DPO settings
lora_config_1 = base_lora_config.copy()
dpo_config_1 = base_dpo_config.copy()
dpo_config_1.loss_type = "sigmoid"
dpo_config_1.beta = 0.1

lora_config_2 = base_lora_config.copy()
lora_config_2.r = 32
dpo_config_2 = base_dpo_config.copy()
dpo_config_2.loss_type = "robust"
dpo_config_2.beta = 0.1

lora_config_3 = base_lora_config.copy()
dpo_config_3 = base_dpo_config.copy()
dpo_config_3.loss_type = "sigmoid"
dpo_config_3.beta = 0.01  # Lower beta = more divergence from reference

# Define 3 separate configs to run concurrently
config_set = List([
    RFModelConfig(
        model_name=MODEL_NAME,
        ref_model_name=None,
        peft_config=lora_config_1,
        training_args=dpo_config_1,
        model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"model_max_length": 1024, "padding_side": "left", "truncation": True}
    ),
    RFModelConfig(
        model_name=MODEL_NAME,
        ref_model_name=None,
        peft_config=lora_config_2,
        training_args=dpo_config_2,
        model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"model_max_length": 1024, "padding_side": "left", "truncation": True}
    ),
    RFModelConfig(
        model_name=MODEL_NAME,
        ref_model_name=None,
        peft_config=lora_config_3,
        training_args=dpo_config_3,
        model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"model_max_length": 1024, "padding_side": "left", "truncation": True}
    )
])

# Define model creation function
def create_model(model_config):
    """Function to create model object for any given config"""
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"], 
        **model_config["model_kwargs"]
    )
    model.config.use_cache = False  # Disable caching for training stability
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], 
        **model_config["tokenizer_kwargs"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return (model, tokenizer)

# Create grid search over all configurations
config_group = RFGridSearch(configs=config_set, trainer_type="DPO")

# Run all 3 configurations concurrently with chunk-based scheduling
experiment.run_fit(
    config_group, 
    create_model, 
    train_dataset, 
    eval_dataset=None,
    num_chunks=4,  # Splits dataset into chunks for concurrent training; balances concurrency and checkpoint overhead
    seed=42
)

# End experiment
experiment.end()
```

### What Happens During Execution

When you run this example:

1. **Config Expansion**: 3 different DPO configurations with varying loss types, beta values, and LoRA ranks
2. **Chunk-based Scheduling**: Training data is divided into 4 chunks, and all configs train on each chunk in sequence
3. **GPU Swapping**: Models are swapped in/out of GPU memory at chunk boundaries—maximizing utilization
4. **Real-time Metrics**: All DPO metrics (rewards/chosen, rewards/rejected, accuracy, margins) visible in the dashboard at `http://localhost:3000`
5. **Interactive Control**: Stop underperforming configs early, clone promising ones with tweaks, or warm-start from a winner

### Key DPO Metrics Tracked

During training, RapidFire AI automatically tracks these critical DPO metrics:

| Metric | Description |
|--------|-------------|
| `rewards/chosen` | Mean difference between policy and reference model log probabilities for chosen responses (scaled by β) |
| `rewards/rejected` | Mean difference between policy and reference model log probabilities for rejected responses (scaled by β) |
| `rewards/accuracies` | Percentage of cases where chosen rewards > rejected rewards |
| `rewards/margins` | Mean difference between chosen and rejected rewards |

This delivers **16-24× higher throughput** compared to training each configuration sequentially, letting you find the best DPO setup much faster.

**📓 Complete Tutorial**: For a full end-to-end example with more realistic settings, see the [DPO Alignment Tutorial Notebook](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment-lite.ipynb)

## Benchmarks: Real-World Speedups

RapidFire AI's chunk-based scheduling delivers dramatic time savings when comparing multiple DPO configurations. Here are benchmarks from internal testing comparing sequential training (one config at a time) vs. RapidFire AI's concurrent execution:

| Scenario | Sequential Time | RapidFire AI Time | Speedup |
|----------|----------------|-------------------|---------|
| 4 DPO configs, 1 GPU | 120 min | 7.5 min | **16×** |
| 8 DPO configs, 1 GPU | 240 min | 12 min | **20×** |
| 4 DPO configs, 2 GPUs | 60 min | 4 min | **15×** |
| 8 DPO configs, 4 GPUs | 60 min | 3 min | **20×** |

*Benchmarks performed on NVIDIA A100 40GB with TinyLlama-1.1B and Llama-3.2-1B models using the Anthropic HH-RLHF dataset*

### Why the Speedup Happens

The speedup comes from three key factors:

1. **Chunk-based scheduling**: All configs train on the same data chunks, enabling early comparison and stopping of poor performers
2. **Efficient GPU utilization**: While one model trains, others are being swapped in/out, minimizing idle time
3. **Interactive Control (IC Ops)**: Stop underperforming configs after 1-2 chunks instead of waiting for full training

The result: **You get signals 16-24× faster**, make decisions sooner, and spend GPU budget only on promising configurations.

## Conclusion

DPO is an elegant way to align LLMs to human preference signals, but the difference between a great DPO system and a merely adequate one is the speed and clarity of your experimentation loop. RapidFire AI's official TRL integration provides: multi-config orchestration, chunked advancement for early comparative signal, real-time control with IC Ops, and built-in visualization. The net effect is simple: more informed decisions, sooner, at lower cost.

If you're using TRL's `DPOTrainer` for alignment work, the integration provides a way to efficiently explore multiple configurations in parallel, helping you find better alignment settings with the same GPU budget.

### Get Started Today

Ready to accelerate your DPO experiments? The RapidFire AI integration with TRL is open source and easy to get started:

**🚀 Try it hands-on**: [Interactive Colab Notebook](http://tinyurl.com/rapidfireai-colab) — Zero setup, runs in your browser

**📚 Full Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai) — Complete guides, examples, and API reference

**💻 GitHub**: [RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai) — Open source, production-ready

**📦 Install via PyPI**: [pypi.org/project/rapidfireai](https://pypi.org/project/rapidfireai) — `pip install rapidfireai`

**📓 DPO Tutorial Notebook**: [rf-tutorial-dpo-alignment-lite.ipynb](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment-lite.ipynb) — End-to-end DPO example

**📖 TRL Integration Docs**: [Hugging Face TRL RapidFire AI Integration](https://huggingface.co/docs/trl/en/rapidfire_integration) — Official TRL documentation

**💬 Join the Community**: [Discord](https://discord.gg/6vSTtncKNN) — Get help, share results, request features

---

**Try the integration and let us know**: How much faster is your DPO experimentation? What alignment challenges should we tackle next? Your feedback shapes where we go from here.


