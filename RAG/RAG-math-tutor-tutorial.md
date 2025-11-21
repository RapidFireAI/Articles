# Optimizing RAG Pipelines with RapidFire AI: A GSM8K Few-Shot Tutorial

Retrieval-Augmented Generation (RAG) pipelines are complex systems with many moving parts. Optimizing them often involves tedious trial-and-error with different prompts, models, and retrieval strategies. **RapidFire AI** changes this by enabling hyperparallelized experimentation, allowing you to test multiple configurations simultaneously and find the best setup faster.

In this tutorial, we'll walk through a practical example using the GSM8K dataset (grade school math problems) to demonstrate how RapidFire AI can streamline your RAG optimization workflow. We'll explore few-shot prompting strategies and compare different model configurations side-by-side.

## What Are We Building?

Unlike traditional chatbots or AI agents that perform actions, this RAG application is designed as a **mathematical reasoning system** that solves grade school math word problems. The GSM8K dataset contains 8,500 linguistically diverse math word problems that require multi-step reasoning—problems like "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. How much does she make selling the rest?"

The "retrieval" component in this RAG pipeline works differently than you might expect:

- **Instead of searching external documents**, it retrieves semantically similar example problems from a curated knowledge base of solved math problems
- **The retrieved examples serve as few-shot prompts** that guide the LLM on how to approach and structure its solution
- **Semantic similarity matching** ensures the most relevant examples are selected for each new problem

This approach is particularly valuable for:
- **Educational platforms** that provide step-by-step math tutoring
- **Homework assistance tools** that explain problem-solving approaches
- **Mathematical reasoning benchmarks** for evaluating LLM capabilities
- **Adaptive learning systems** that select appropriate example complexity

By testing different configurations (number of examples, embedding models, LLM choice), we can optimize the system's accuracy and find the sweet spot between context length and performance.

## Prerequisites

Before we begin, ensure you have RapidFire AI installed and your environment set up:

```bash
pip install rapidfireai
rapidfireai init --evals
```

You'll also need an OpenAI API key for this tutorial.

## Step 1: Setup and Data Loading

First, let's import the necessary libraries and load our dataset. We'll use the GSM8K dataset from Hugging Face.

```python
from rapidfireai import Experiment
from rapidfireai.evals.automl import List, RFOpenAIAPIModelConfig, RFPromptManager, RFGridSearch
from datasets import load_dataset
import pandas as pd

# Load and shuffle the dataset
gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train")
gsm8k_dataset = gsm8k_dataset.shuffle(seed=42)
```

## Step 2: Initialize the Experiment

Create a RapidFire `Experiment`. This object manages your runs and tracks results.

```python
experiment = Experiment(experiment_name="exp1-gsm8k-fewshot", mode="evals")
```

## Step 3: Dynamic Few-Shot Prompting

One of the most powerful ways to improve LLM performance on reasoning tasks is few-shot prompting. RapidFire's `RFPromptManager` makes it easy to manage and select examples dynamically.

We'll define a set of math problem examples and use `SemanticSimilarityExampleSelector` to pick the most relevant ones for each query.

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

INSTRUCTIONS = "You are a helpful assistant that is good at solving math problems. You think step by step and ALWAYS output the final answer after '####'."

# ... (Define your list of 'examples' here, see notebook for full list) ...

fewshot_prompt_manager = RFPromptManager(
    instructions=INSTRUCTIONS,
    examples=examples,
    embedding_cls=HuggingFaceEmbeddings,
    embedding_kwargs={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 128},
    },
    example_selector_cls=SemanticSimilarityExampleSelector,
    example_prompt_template=PromptTemplate(
        input_variables=["question", "answer"],
        template="Question: {question}\nAnswer: {answer}",
    ),
    k=List([3, 5]),  # Key Feature: We will test both 3 and 5 examples!
)
```

Notice `k=List([3, 5])`. This is RapidFire's magic at work. We are defining a search space right inside our object definition. RapidFire will automatically create configurations to test both 3-shot and 5-shot prompting.

## Step 4: Define Pipeline Logic

We need to define how data flows through our pipeline:
1.  **Preprocessing**: Formats the input prompts.
2.  **Postprocessing**: Extracts the answer from the model's output.
3.  **Metrics**: Calculates accuracy.

```python
import re

def sample_preprocess_fn(batch, rag, prompt_manager):
    return {
        "prompts": [
            [
                {"role": "system", "content": prompt_manager.get_instructions()},
                {
                    "role": "user",
                    "content": f"Here are some examples: \n{examples}. \nNow answer the following question:\n{question}",
                },
            ]
            for question, examples in zip(
                batch["question"],
                prompt_manager.get_fewshot_examples(user_queries=batch["question"]),
            )
        ],
        **batch,
    }

def extract_solution(answer):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", answer)
    return solution.group(0).split("#### ")[1].replace(",", "") if solution else "0"

def sample_postprocess_fn(batch):
    batch["model_answer"] = [extract_solution(ans) for ans in batch["generated_text"]]
    batch["ground_truth"] = [extract_solution(ans) for ans in batch["answer"]]
    return batch

def sample_compute_metrics_fn(batch):
    correct = sum(1 for p, g in zip(batch["model_answer"], batch["ground_truth"]) if p == g)
    return {"Correct": {"value": correct}, "Total": {"value": len(batch["model_answer"])}}
```

## Step 5: Configure Models and Search Space

Now we define the models we want to test. We'll compare `gpt-4o-mini` and `gpt-4o`, and for each, we'll test different "reasoning effort" levels (if applicable, or just treat them as different configs).

```python
OPENAI_API_KEY = "your-api-key" # Or use input()

# Config 1: GPT-4o-mini
openai_config1 = RFOpenAIAPIModelConfig(
    client_config={"api_key": OPENAI_API_KEY},
    model_config={
        "model": "gpt-4o-mini",
        "max_completion_tokens": 1024,
    },
    prompt_manager=fewshot_prompt_manager, # Links to our k=[3,5] search
)

# Config 2: GPT-4o
openai_config2 = RFOpenAIAPIModelConfig(
    client_config={"api_key": OPENAI_API_KEY},
    model_config={
        "model": "gpt-4o",
        "max_completion_tokens": 1024,
    },
    prompt_manager=fewshot_prompt_manager,
)

config_set = {
    "openai_config": List([openai_config1, openai_config2]),
    "batch_size": 128,
    "preprocess_fn": sample_preprocess_fn,
    "postprocess_fn": sample_postprocess_fn,
    "compute_metrics_fn": sample_compute_metrics_fn,
    # ... add accumulate_metrics_fn if needed
}

# Create the grid search object
config_group = RFGridSearch(config_set)
```

This setup creates a grid of experiments:
(GPT-4o-mini vs GPT-4o) × (3-shot vs 5-shot) = **4 distinct configurations**.

## Step 6: Run the Evaluation

Launch the experiment! RapidFire will handle the scheduling, ensuring efficient execution across your available resources (or API rate limits).

```python
results = experiment.run_evals(
    config_group=config_group,
    dataset=gsm8k_dataset,
    num_actors=2, # Parallel workers
    num_shards=4,
    seed=42,
)
```

## Step 7: Analyze Results

Once finished, you can easily view the results as a pandas DataFrame to compare performance.

```python
results_df = pd.DataFrame([
    {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in {**metrics_dict, 'run_id': run_id}.items()}
    for run_id, (_, metrics_dict) in results.items()
])

print(results_df)
```

## Conclusion

In this tutorial, we saw how RapidFire AI simplifies the process of testing multiple RAG strategies. By defining search spaces for parameters like `k` (number of few-shot examples) and model types, we could automatically generate and execute a comprehensive evaluation matrix.

This allows you to make data-driven decisions about your RAG pipeline's configuration without writing complex boilerplate code for loop management or parallelization.

**Ready to try it yourself?** Check out the [RapidFire AI Documentation](https://oss-docs.rapidfire.ai) for more advanced features like Interactive Control Ops and custom local models.
