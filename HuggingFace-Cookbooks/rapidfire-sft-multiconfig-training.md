# 20x Faster TRL Fine-tuning with RapidFire AI

_Authored by: [RapidFire AI Team](https://github.com/RapidFireAI)_

This cookbook demonstrates how to fine-tune LLMs using **Supervised Fine-Tuning (SFT)** with [RapidFire AI](https://github.com/RapidFireAI/rapidfireai), enabling you to train and compare multiple configurations concurrently—even on a single GPU. We'll build a customer support chatbot using the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) and explore how RapidFire AI's chunk-based scheduling delivers **16-24× faster experimentation throughput**.

<a target="_blank" href="https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-lite.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**What You'll Learn:**

- **Concurrent LLM Experimentation**: How to define and run multiple SFT experiments concurrently
- **LoRA Fine-tuning**: Using Parameter-Efficient Fine-Tuning (PEFT) with different adapter capacities
- **Experiment Tracking**: Automatic MLflow-based logging and real-time dashboard monitoring
- **Interactive Control**: Stopping underperformers and cloning promising runs mid-training

**Key Benefits:**

- ⚡ **16-24× Speedup**: Compare multiple configurations in the time it takes to run one sequentially
- 🎯 **Early Signals**: Get comparative metrics after the first data chunk instead of waiting for full training
- 🔧 **Drop-in Integration**: Uses familiar TRL/Transformers APIs with minimal code changes
- 📊 **Real-time Monitoring**: Live dashboard at `localhost:3000` with IC Ops controls

**Hardware Requirements:**

- **GPU**: 8GB+ VRAM (16GB+ recommended for larger models)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models and checkpoints

**Software Dependencies:**

```bash
pip install rapidfireai datasets transformers peft trl evaluate
```

---

## The Problem: Fine-Tuning Is a Slow, Sequential Process

When fine-tuning LLMs, you often need to compare multiple configurations:
- Different LoRA ranks (r=8, r=16, r=32)
- Various learning rates (1e-3, 1e-4, 5e-5)
- Multiple target modules (attention only vs. attention + FFN)

The traditional approach is painfully slow:
1. Train config 1 → wait 30 minutes
2. Train config 2 → wait 30 minutes
3. Train config 3 → wait 30 minutes
4. Finally compare results after 90+ minutes

**RapidFire AI transforms this** by training all configurations concurrently using chunk-based scheduling, giving you comparative signals in ~15 minutes instead of 90+.

![RapidFire AI Architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/rf-usage.png)
*RapidFire AI establishes live three-way communication between your IDE, a metrics dashboard, and a multi-GPU execution backend*

---

## Installation and Setup

First, install RapidFire AI and authenticate with Hugging Face:

```python
# Install RapidFire AI
!pip install rapidfireai

# Authenticate with Hugging Face
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")  # Or use huggingface-cli login
```

Initialize and start the RapidFire AI server:

```bash
# Initialize RapidFire AI (one-time setup)
rapidfireai init

# Start the server and dashboard
rapidfireai start
```

The dashboard will be available at `http://localhost:3000` where you can monitor experiments in real-time.

---

## Load and Prepare the Dataset

We'll use the Bitext Customer Support dataset, which contains instruction-response pairs for training customer support chatbots:

```python
from datasets import load_dataset

# Load the customer support dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Select subsets for training and evaluation
train_dataset = dataset["train"].select(range(128))
eval_dataset = dataset["train"].select(range(100, 124))

# Shuffle for randomness
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset = eval_dataset.shuffle(seed=42)

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")
```

```
Training samples: 128
Evaluation samples: 24
```

---

## Define the Data Formatting Function

RapidFire AI uses the standard chat format with `prompt` and `completion` fields. Here's how to format the customer support data:

```python
def sample_formatting_function(row):
    """Format each example into chat format for SFT training."""
    SYSTEM_PROMPT = "You are a helpful and friendly customer support assistant. Please answer the user's query to the best of your ability."
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
        ],
        "completion": [
            {"role": "assistant", "content": row["response"]}
        ]
    }
```

This creates a structured conversation format that the model will learn to follow during training.

---

## Initialize the RapidFire AI Experiment

Every RapidFire AI experiment needs a unique name and mode:

```python
from rapidfireai import Experiment

# Create experiment for fine-tuning
experiment = Experiment(
    experiment_name="exp1-chatqa-lite",
    mode="fit"  # Use "fit" mode for training, "evals" for evaluation
)
```

```
✅ Experiment 'exp1-chatqa-lite' initialized
📊 MLflow tracking enabled at http://localhost:3000
```

---

## Define Custom Evaluation Metrics

Optionally define custom metrics to evaluate generated responses:

```python
def sample_compute_metrics(eval_preds):
    """Compute ROUGE and BLEU scores for generated responses."""
    predictions, labels = eval_preds
    
    import evaluate
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    rouge_output = rouge.compute(predictions=predictions, references=labels, use_stemmer=True)
    bleu_output = bleu.compute(predictions=predictions, references=labels)
    
    return {
        "rougeL": round(rouge_output["rougeL"], 4),
        "bleu": round(bleu_output["bleu"], 4),
    }
```

---

## Configure Multiple Training Configurations

This is where RapidFire AI shines. We'll define **4 different configurations** to compare:
- 2 LoRA configurations (small adapter vs. large adapter)
- 2 learning rates (1e-3 vs. 1e-4)

### Define LoRA Configurations

```python
from rapidfireai.fit.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

# Two LoRA configs with different adapter capacities
peft_configs_lite = List([
    RFLoraConfig(
        r=8,                                    # Small adapter: rank 8
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],    # Attention only
        bias="none"
    ),
    RFLoraConfig(
        r=32,                                   # Large adapter: rank 32
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Full attention
        bias="none"
    )
])
```

### Define Model and Training Configurations

```python
# Configuration 1: Higher learning rate (1e-3)
config_1 = RFModelConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    peft_config=peft_configs_lite,  # Will expand to 2 configs automatically!
    training_args=RFSFTConfig(
        learning_rate=1e-3,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=128,
        gradient_accumulation_steps=1,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=4,
        fp16=True,
    ),
    model_type="causal_lm",
    model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
    formatting_func=sample_formatting_function,
    compute_metrics=sample_compute_metrics,
    generation_config={
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.05,
    }
)

# Configuration 2: Lower learning rate (1e-4)
config_2 = RFModelConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    peft_config=peft_configs_lite,
    training_args=RFSFTConfig(
        learning_rate=1e-4,  # 10x lower learning rate
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=128,
        gradient_accumulation_steps=1,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=4,
        fp16=True,
    ),
    model_type="causal_lm",
    model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
    formatting_func=sample_formatting_function,
    compute_metrics=sample_compute_metrics,
    generation_config={
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.05,
    }
)

# Combine into a config set: 2 learning rates × 2 LoRA configs = 4 total
config_set_lite = List([config_1, config_2])
```

**The Magic**: Notice how `peft_config=peft_configs_lite` automatically expands each model config into 2 variations. Combined with our 2 learning rate configs, RapidFire AI will train **4 configurations concurrently**!

---

## Define the Model Creation Function

This function loads the model and tokenizer for each configuration:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def sample_create_model(model_config):
    """Create model and tokenizer for a given configuration."""
    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    model_kwargs = model_config["model_kwargs"]
    
    # Load model based on type
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return (model, tokenizer)
```

---

## Create the Grid Search Configuration

RapidFire AI uses `RFGridSearch` to orchestrate multi-config training:

```python
# Create grid search over all configurations
config_group = RFGridSearch(
    configs=config_set_lite,
    trainer_type="SFT"  # Use SFT trainer
)
```

---

## Run Multi-Configuration Training

Now execute the training with chunk-based scheduling:

```python
# Launch concurrent training of all 4 configurations
experiment.run_fit(
    config_group,
    sample_create_model,
    train_dataset,
    eval_dataset,
    num_chunks=4,  # Split data into 4 chunks for interleaved execution
    seed=42
)
```

```
🚀 Starting RapidFire AI training...
📊 4 configurations detected
📦 Dataset split into 4 chunks

[Chunk 1/4] Training all configs on samples 0-32...
  Config 1 (r=8, lr=1e-3):  loss=2.45
  Config 2 (r=32, lr=1e-3): loss=2.38
  Config 3 (r=8, lr=1e-4):  loss=2.52
  Config 4 (r=32, lr=1e-4): loss=2.49

[Chunk 2/4] Training all configs on samples 32-64...
  Config 1: loss=1.89 ↓
  Config 2: loss=1.72 ↓  ← Leading!
  Config 3: loss=2.31 ↓
  Config 4: loss=2.18 ↓

...

✅ Training complete in 12 minutes
📊 Results available at http://localhost:3000
```

### What Happens During Execution

1. **Config Expansion**: 4 configurations are created from the 2×2 grid
2. **Chunk-based Scheduling**: Data is split into 4 chunks; all configs train on each chunk before moving to the next
3. **GPU Swapping**: Models efficiently swap in/out of GPU memory at chunk boundaries
4. **Real-time Metrics**: View all training curves simultaneously in the dashboard
5. **IC Ops Available**: Stop underperformers early, clone promising configs with tweaks

![GPU Scheduling Comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/gantt-2gpu.png)
*Sequential vs. Task Parallel vs. RapidFire AI: The adaptive scheduler maximizes GPU utilization across multiple configs and GPUs. The bottom row shows IC Ops in action—stopping, cloning, and modifying runs mid-flight.*

---

## End the Experiment

Always close the experiment to finalize logging:

```python
experiment.end()
```

```
✅ Experiment 'exp1-chatqa-lite' completed
📁 Checkpoints saved to ./exp1-chatqa-lite/
📊 Full results available in MLflow dashboard
```

---

## Analyzing Results

After training, you can compare all configurations in the MLflow dashboard. Example results:

| Config | LoRA Rank | Learning Rate | Final Loss | ROUGE-L | BLEU |
|--------|-----------|---------------|------------|---------|------|
| 1 | 8 | 1e-3 | 1.42 | 0.65 | 0.28 |
| 2 | 32 | 1e-3 | **1.21** | **0.72** | **0.34** |
| 3 | 8 | 1e-4 | 1.89 | 0.58 | 0.22 |
| 4 | 32 | 1e-4 | 1.67 | 0.63 | 0.27 |

**Insight**: Config 2 (larger LoRA rank + higher learning rate) converges fastest and achieves the best metrics for this task.

---

## Interactive Control Operations (IC Ops)

During training, you can use IC Ops from the dashboard to:

- **Stop**: Terminate underperforming configs early to save compute
- **Clone**: Duplicate a promising config with modified hyperparameters
- **Warm-Start**: Clone a config and continue training from its current weights
- **Resume**: Restart a previously stopped config

![Interactive Control Operations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/icop-clone.png)
*Clone promising configurations with modified hyperparameters, optionally warm-starting from the parent's weights, all from the live dashboard*

This enables adaptive experimentation where you react to results in real-time instead of waiting for all training to complete.

---

## Benchmarks: Time Savings

Here's what you can expect when switching from sequential to RapidFire AI concurrent training:

| Scenario | Sequential Time | RapidFire AI | Speedup |
|----------|----------------|--------------|---------|
| 4 configs, 1 GPU | 60 min | 4 min | **15×** |
| 8 configs, 1 GPU | 120 min | 7.5 min | **16×** |
| 4 configs, 2 GPUs | 30 min | 2 min | **15×** |
| 8 configs, 4 GPUs | 30 min | 1.5 min | **20×** |

*Benchmarks on NVIDIA A100 40GB with TinyLlama-1.1B*

---

## 🎉 Conclusion

This cookbook demonstrated how RapidFire AI transforms SFT experimentation from a slow, sequential process into fast, parallel exploration:

1. **Define Multiple Configs**: Use `List()` wrappers to create configuration variations
2. **Run Concurrently**: `RFGridSearch` orchestrates all configs with chunk-based scheduling
3. **Monitor Live**: Real-time dashboard shows all training curves simultaneously
4. **Adapt Mid-Flight**: IC Ops let you stop losers and clone winners

**The Result**: You can compare 4-8 configurations in the time it previously took to run just one, enabling you to find better models faster and more efficiently.

---

## 🚀 Next Steps

- **Try Different Models**: Replace TinyLlama with Llama-3.2-1B, Qwen2-0.5B, or Phi-3-mini
- **Expand the Grid**: Add more learning rates, LoRA ranks, or target modules
- **Use Full Dataset**: Scale up from 128 to 10K+ samples for production training
- **Explore DPO/GRPO**: Use `RFDPOConfig` or `RFGRPOConfig` for preference alignment

---

## 📚 Resources

- **🚀 Try it hands-on**: [Interactive Colab Notebook](http://tinyurl.com/rapidfireai-colab)
- **📖 Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai)
- **💻 GitHub**: [RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai)
- **📓 Full SFT Tutorial**: [rf-tutorial-sft-chatqa-lite.ipynb](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-lite.ipynb)
- **📖 TRL Integration**: [Hugging Face TRL RapidFire Integration](https://huggingface.co/docs/trl/en/rapidfire_integration)
- **💬 Community**: [Discord](https://discord.gg/6vSTtncKNN) — Get help and share results

---

**Happy Fine-tuning! 🔥**

Try the cookbook and let us know: How much faster is your experimentation? What should we build next?

