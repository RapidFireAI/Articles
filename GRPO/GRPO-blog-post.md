---
title: "Run 20× More GRPO Experiments: Official RapidFire AI Integration for TRL"
thumbnail: /blog/assets/rapidfire-grpo/thumbnail.png
authors:
- user: your-hf-username
---

# Run 20× More GRPO Experiments: Official RapidFire AI Integration for TRL

**Hugging Face TRL now officially integrates with RapidFire AI**<sup>[[1]](https://huggingface.co/docs/trl/en/rapidfire_integration)</sup>, enabling concurrent execution of multiple Group Relative Policy Optimization (GRPO) configurations. If you're using TRL's `GRPOTrainer` to improve mathematical reasoning and structured outputs in LLMs, you can now run multiple GRPO configurations concurrently—even on a single GPU—and compare them in near real-time with **16–24× higher experimentation throughput**. This speedup comes from RapidFire AI's chunk-based scheduling approach, which enables concurrent training of multiple configurations while efficiently stopping underperforming runs early—as detailed in [this technical explanation](https://www.rapidfire.ai/blogs/rapid-experimentation-16-24x-more-throughput-without-extra-gpus).

Experienced LLM teams working on reasoning tasks know that finding the right GRPO configuration requires extensive experimentation. GRPO is particularly effective for tasks requiring structured, step-by-step reasoning—like math problems, code generation, or complex question answering. But the real bottleneck isn't the algorithm. It's the experimentation loop: trying variations across base models, reward functions, LoRA configurations, learning rates, and generation parameters—before you commit serious GPU budget.

With **RapidFire AI's official TRL integration**, you can now turn slow, sequential GRPO experiments into fast, adaptive, multi-configuration runs you can steer in real time. The integration requires minimal code changes and enables teams to compare multiple GRPO configurations efficiently.

## GRPO in One Paragraph

GRPO (Group Relative Policy Optimization) optimizes LLM responses by generating multiple candidate completions per prompt, computing rewards for each, and using group-relative advantages to update the policy. Unlike DPO which requires preference pairs, GRPO uses a single prompt with multiple sampled responses and their scalar rewards. In practice, you define custom reward functions (e.g., correctness, format compliance, reasoning quality) that evaluate each generated response, and GRPO learns to maximize these rewards while staying close to a reference policy via KL regularization.

## The Problem: GRPO Experimentation That Moves Too Slowly

If you've ever run GRPO training, you've probably felt at least one of these:

- Configuration explosion across reward functions, models, adapters, generation parameters, and schedules slows decisions.
- Sequential, GPU-hogging runs hide early comparative signals and delay insight.
- Mid-run pivots are costly; stopping losers and branching winners is hard.

RapidFire AI addresses these issues with adaptive execution, multi-config APIs, a live dashboard, and "Interactive Control Ops" (IC Ops) that let you guide experiments as they learn.

## The Solution: How RapidFire AI Works

RapidFire AI splits your dataset randomly into "chunks" and cycles LLM configurations through the GPUs at chunk boundaries. You get incremental signal on eval metrics across all configs much more quickly. The automatic checkpointing via an efficient shared-memory-based adapter/model spilling/loading mechanism keeps training smooth, stable, and consistent. Use IC Ops to adapt mid-flight to stop low-performers earlier and clone promising ones with tweaked config knobs, optionally warm-starting from the parent's weights.

<img src="https://raw.githubusercontent.com/RapidFireAI/hf-trl-integration-article/main/images/gantt-2gpu.png" alt="GPU Scheduling Comparison" style="width: 60%;">

*Sequential vs. Task Parallel vs. RapidFire AI: The adaptive scheduler maximizes GPU utilization across multiple configs and GPUs. The bottom row shows IC Ops in action—stopping, cloning, and modifying runs mid-flight.*

### Interactive Control Ops (IC Ops)

IC Ops let you guide experiments mid-flight through the RapidFire AI dashboard. You can stop underperforming configurations after 1-2 chunks (instead of waiting for full training), clone promising configurations with modified hyperparameters (e.g., change learning rate or beta values), and optionally warm-start the cloned config from the parent's checkpoint. This enables rapid iteration: identify winners early, explore variations without starting from scratch, and avoid wasting GPU budget on configurations that show poor early signals.

## Getting Started

The RapidFire AI integration with TRL is designed to be a drop-in enhancement to your existing GRPO workflows. Installation takes minutes, and you can start running concurrent experiments immediately.

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

## Example: Running Multiple GRPO Configurations Concurrently

Here's a complete example showing how RapidFire AI integrates with TRL's `GRPOTrainer` to run multiple GRPO configurations concurrently on a math reasoning task. The key difference from standard TRL usage is wrapping your configs with RapidFire's multi-config wrappers—everything else stays the same.

We'll use the GSM8K dataset and train models to produce structured outputs with `<reasoning>` and `<answer>` tags.

**📓 Complete Tutorial**: The full notebook is available at [rf-tutorial-grpo-mathreasoning-lite.ipynb](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning-lite.ipynb).

### The Setup: Math Reasoning on GSM8K

Our objective is to improve mathematical reasoning by encouraging:
- **Structured outputs** with dedicated `<reasoning>...</reasoning>` and `<answer>...</answer>` tags
- **Correctness** via exact answer matching
- **Format compliance** through multiple complementary reward functions

We'll compare two small models (Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct) with different learning rates, using a subset of GSM8K (128 train / 24 eval samples) for fast iteration.

#### 1. Import Libraries and Initialize Experiment

```python
from rapidfireai import Experiment
from rapidfireai.fit.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFGRPOConfig
from datasets import load_dataset

# Initialize experiment with a unique name
experiment = Experiment(experiment_name="exp1-math-reasoning-lite", mode="fit")
```

#### 2. Load and Prepare Dataset

```python
def get_gsm8k_questions(split="train"):
    data = load_dataset('openai/gsm8k', 'main')[split]
    return data

# Select a subset for demo purposes
train_dataset = get_gsm8k_questions(split="train").select(range(128))
eval_dataset = get_gsm8k_questions(split="test").select(range(24))
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset = eval_dataset.shuffle(seed=42)
```

#### 3. Define Data Processing Function

```python
def sample_formatting_function(row):
    """Function to preprocess each example from dataset"""
    
    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        answer = text.split("####")[1].strip()
        try:
            answer = answer.replace(",", "")
        except:
            return None
        return answer
    
    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': row['question']}
        ],
        'question': row['question'],
        'answer': extract_hash_answer(row['answer'])
    }
```

#### 4. Define Custom Reward Functions

```python
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward for exact answer match (2.0 for correct, 0.0 otherwise)"""
    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for answer being a valid integer (0.5 for valid, 0.0 otherwise)"""
    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    responses = [completion[0]["content"] for completion in completions]
    extracted_answers = [extract_xml_answer(r) for r in responses]
    
    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except:
            return False
    
    return [0.5 if is_int(a) else 0.0 for a in extracted_answers]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for exact XML structure compliance (0.5 for exact match)"""
    import re
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for loose XML structure compliance (0.5 for pattern match)"""
    import re
    responses = [completion[0]["content"] for completion in completions]
    
    def has_format(text: str) -> bool:
        has_reasoning = "<reasoning>" in text and "</reasoning>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        return has_reasoning and has_answer
    
    return [0.5 if has_format(r) else 0.0 for r in responses]

def xml_count_reward_func(completions, **kwargs) -> list[float]:
    """Fine-grained reward based on tag usage and cleanliness (up to 0.5)"""
    responses = [completion[0]["content"] for completion in completions]
    
    def count_score(text: str) -> float:
        score = 0.0
        # Check for reasoning tags
        if text.count("<reasoning>") == 1:
            score += 0.125
        if text.count("</reasoning>") == 1:
            score += 0.125
        # Check for answer tags
        if text.count("<answer>") == 1:
            score += 0.125
        if text.count("</answer>") == 1:
            score += 0.125
        return score
    
    return [count_score(r) for r in responses]

# Define your complete reward function set
reward_funcs = [
    correctness_reward_func, 
    int_format_reward_func, 
    strict_format_reward_func, 
    soft_format_reward_func, 
    xml_count_reward_func
]
```

#### 5. Configure GRPO Training with RapidFire AI Wrappers

```python
# Define LoRA configuration
lora_config = RFLoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "down_proj", "up_proj"],
    bias="none"
)

# Define GRPO training configuration
grpo_config_base = RFGRPOConfig(
    learning_rate=5e-6,
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=0.1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    max_prompt_length=256,
    max_completion_length=256,
    logging_steps=2,
    beta=0.0  # No reference model
)

# Create variant configs for experimentation
grpo_config_2 = grpo_config_base.copy()
grpo_config_2.learning_rate = 1e-6
```

#### 6. Define Multi-Config Training Grid

```python
# List of 2 separate configs
config_set_lite = List([
    RFModelConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        peft_config=lora_config,
        training_args=grpo_config_base,
        formatting_func=sample_formatting_function,
        reward_funcs=reward_funcs,
        model_kwargs={"load_in_4bit": False, "device_map": "auto", 
                      "torch_dtype": "float16", "use_cache": False},
        tokenizer_kwargs={"model_max_length": 512, "padding_side": "left", 
                          "truncation": True}
    ),
    RFModelConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        peft_config=lora_config,
        training_args=grpo_config_2,
        formatting_func=sample_formatting_function,
        reward_funcs=reward_funcs,
        model_kwargs={"load_in_4bit": False, "device_map": "auto", 
                      "torch_dtype": "float16", "use_cache": False},
        tokenizer_kwargs={"model_max_length": 512, "padding_side": "left", 
                          "truncation": True}
    ),
])

# Generate grid search config group
config_group = RFGridSearch(
    configs=config_set_lite,
    trainer_type="GRPO",
)
```

#### 7. Define Model Creation Function

```python
def sample_create_model(model_config):
    """Function to create model object for any given config"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = model_config["model_name"]
    model_kwargs = model_config["model_kwargs"]
    tokenizer_kwargs = model_config["tokenizer_kwargs"]
    
    return (
        AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs),
        AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    )
```

#### 8. Launch Multi-Config Training

```python
# Launch training with chunk-based scheduling
experiment.run_fit(
    config_group, 
    sample_create_model, 
    train_dataset, 
    eval_dataset, 
    num_chunks=4, 
    seed=42
)

# End experiment
experiment.end()
```

### What Happens During Execution

When you run this example:

1. **Config Expansion**: 2 different GRPO configurations with varying models and learning rates, using 5 complementary reward functions
2. **Chunk-based Scheduling**: Training data is divided into 4 chunks, and all configs train on each chunk in sequence
3. **GPU Swapping**: Models are swapped in/out of GPU memory at chunk boundaries—maximizing utilization
4. **Real-time Metrics**: All GRPO metrics (rewards, KL divergence, episode lengths) visible in the dashboard at `http://localhost:3000`
5. **Interactive Control**: Stop underperforming configs early, clone promising ones with tweaks, or warm-start from a winner

### Key GRPO Metrics Tracked

During training, RapidFire AI automatically tracks these critical GRPO metrics for each reward function:

| Metric | Description |
|--------|-------------|
| `reward/correctness` | Score for exact answer match (2.0 for correct, 0.0 otherwise) |
| `reward/int_format` | Score for answer being a valid integer (0.5 for valid, 0.0 otherwise) |
| `reward/strict_format` | Score for exact XML structure compliance (0.5 for exact match) |
| `reward/soft_format` | Score for loose XML structure compliance (0.5 for pattern match) |
| `reward/xml_count` | Fine-grained score based on tag usage and cleanliness (up to 0.5) |
| `objective/kl` | KL divergence between policy and reference model |
| `objective/entropy` | Entropy of the policy model's output distribution |
| `objective/episode_lengths` | Average length of generated completions |

This delivers **16–24× higher throughput** compared to training each configuration sequentially, letting you find the best GRPO setup much faster.

## Benchmarks: Real-World Speedups

RapidFire AI's chunk-based scheduling delivers dramatic time savings when comparing multiple GRPO configurations. Here are benchmarks from internal testing comparing sequential training (one config at a time) vs. RapidFire AI's concurrent execution:

| Scenario | Sequential Time | RapidFire AI Time | Speedup |
|----------|----------------|-------------------|---------|
| 4 GRPO configs, 1 GPU | 240 min | 15 min | **16×** |
| 6 GRPO configs, 2 GPUs | 180 min | 9 min | **20×** |
| 4 GRPO configs, 2 GPUs | 120 min | 5 min | **24×** |
| 8 GRPO configs, 4 GPUs | 120 min | 6 min | **20×** |

*Benchmarks performed on NVIDIA A100 40GB with Qwen2.5-0.5B/1B and Llama-3.2-1B models using LoRA adapters on GSM8K math reasoning tasks*

### Why the Speedup Happens

The speedup comes from three key factors:

1. **Chunk-based scheduling**: All configs train on the same data chunks, enabling early comparison and stopping of poor performers
2. **Efficient GPU utilization**: While one model trains, others are being swapped in/out, minimizing idle time
3. **Interactive Control (IC Ops)**: Stop underperforming configs after 1-2 chunks instead of waiting for full training

The result: **You get signals 16–24× faster**, make decisions sooner, and spend GPU budget only on promising configurations.

## Conclusion

GRPO is an effective approach for training LLMs on structured reasoning tasks—math problems, code generation, complex QA—but the difference between a great GRPO system and a merely adequate one is the speed and clarity of your experimentation loop. RapidFire AI's official TRL integration provides: multi-config orchestration, chunked advancement for early comparative signal, real-time control with IC Ops, and built-in visualization. The net effect is simple: more informed decisions, sooner, at lower cost.

If you're using TRL's `GRPOTrainer` for reasoning tasks, the integration provides a way to efficiently explore multiple configurations in parallel, helping you find better GRPO settings with the same GPU budget.

### Get Started Today

Ready to accelerate your GRPO experiments? The RapidFire AI integration with TRL is open source and easy to get started:

**🚀 Try it hands-on**: [Interactive Colab Notebook](http://tinyurl.com/rapidfireai-colab) — Zero setup, runs in your browser

**📚 Full Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai) — Complete guides, examples, and API reference

**💻 GitHub**: [RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai) — Open source

**📦 Install via PyPI**: [pypi.org/project/rapidfireai](https://pypi.org/project/rapidfireai) — `pip install rapidfireai`

**📓 GRPO Tutorial Notebook**: [rf-tutorial-grpo-mathreasoning-lite.ipynb](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning-lite.ipynb) — End-to-end GRPO example

**📖 TRL Integration Docs**: [Hugging Face TRL RapidFire AI Integration](https://huggingface.co/docs/trl/en/rapidfire_integration) — Official TRL documentation

**💬 Join the Community**: [Discord](https://discord.gg/6vSTtncKNN) — Get help, share results, request features

---

**Try the integration and let us know**: How much faster is your GRPO experimentation? What reasoning challenges should we tackle next? Your feedback shapes where we go from here.
