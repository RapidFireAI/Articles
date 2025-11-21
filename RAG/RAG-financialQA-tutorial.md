# Optimizing RAG Pipelines with RapidFireAI: A Financial Q&A Use Case

Retrieval-Augmented Generation (RAG) has become the standard for building knowledgeable AI assistants. However, building a *good* RAG pipeline involves making dozens of choices: Which chunk size? Which embedding model? Should I rerank? Which LLM works best for my budget?

In this tutorial, we'll walk through how to use **RapidFireAI** to systematically optimize a RAG pipeline for a Financial Opinion Q&A chatbot. We'll use the FiQA dataset and explore how to tune both the retrieval (LangChain) and generation (vLLM) components.

## The Application: Financial Opinion Q&A Chatbot

We're building a **chatbot that answers opinion-based financial questions** by retrieving relevant information from a corpus of financial documents and generating contextually-aware responses. Unlike simple factual lookup systems, this RAG application handles subjective queries such as:

- *"Should I invest in index funds or individual stocks?"*
- *"What's the best way to save for retirement in my 30s?"*
- *"Is it worth refinancing my mortgage right now?"*

The [FiQA dataset](https://huggingface.co/datasets/explodinggradients/fiqa) (Financial Opinion Mining and Question Answering) is specifically designed for this task. It's part of the BEIR benchmark and contains:

- **Queries**: Real-world financial questions from forums and Q&A sites
- **Corpus**: A collection of financial documents, posts, and expert answers
- **Relevance judgments (qrels)**: Ground truth labels indicating which documents are relevant to each query

Our RAG system works by:
1. **Retrieval**: When a user asks a question, we search the financial corpus for the most relevant documents using embeddings and optional reranking
2. **Generation**: The retrieved context is fed into an LLM along with the question to generate a coherent, contextually-grounded answer
3. **Evaluation**: We measure retrieval quality (Precision, Recall, NDCG, MRR) to ensure our pipeline is finding and using the right information

This makes it ideal for customer support chatbots, financial advisory tools, or educational platforms that need to provide nuanced, context-aware financial guidance.

## Prerequisites

Before we begin, make sure you have RapidFireAI installed. You can follow the [Install and Get Started](https://oss-docs.rapidfire.ai/en/latest/walkthrough.html) guide in our documentation.

## 1. Setup and Initialization

First, let's import the necessary modules and initialize a RapidFireAI experiment.

```python
from rapidfireai import Experiment
from rapidfireai.evals.automl import List, RFLangChainRagSpec, RFvLLMModelConfig, RFPromptManager, RFGridSearch
import pandas as pd
from pathlib import Path

# Initialize the experiment
experiment = Experiment(experiment_name="exp1-fiqa-rag", mode="evals")
```

## 2. Loading the Data

For this tutorial, we'll use the FiQA (Financial Opinion QA) dataset. We'll load the queries and the corpus.

```python
from datasets import load_dataset

# Assuming dataset is in 'datasets/fiqa'
dataset_dir = Path("datasets")

# Load queries
fiqa_dataset = load_dataset("json", data_files=str(dataset_dir / "fiqa" / "queries.jsonl"), split="train")
fiqa_dataset = fiqa_dataset.rename_columns({"text": "query", "_id": "query_id"})

# Load relevance judgments (qrels)
qrels = pd.read_csv(str(dataset_dir / "fiqa" / "qrels.tsv"), sep="\t")
qrels = qrels.rename(columns={"query-id": "query_id", "corpus-id": "corpus_id", "score": "relevance"})
```

## 3. Defining the RAG Search Space

This is where RapidFireAI shines. Instead of hardcoding a single RAG configuration, we define a **search space** using `RFLangChainRagSpec`.

We will test:
*   **2 Chunking Strategies**: Different chunk sizes (256 vs 128).
*   **2 Reranking Strategies**: Different `top_n` values (2 vs 5).

This gives us 4 combinations to evaluate for the retrieval part.

```python
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

batch_size = 128

rag_gpu = RFLangChainRagSpec(
    document_loader=DirectoryLoader(
        path=str(dataset_dir / "fiqa"),
        glob="corpus.jsonl",
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",
            "content_key": "text",
            "metadata_func": lambda record, metadata: {"corpus_id": int(record.get("_id"))},
            "json_lines": True,
            "text_content": False,
        },
        sample_seed=42,
    ),
    # Define multiple text splitters to test
    text_splitter=List([
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="gpt2", chunk_size=256, chunk_overlap=32),
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="gpt2", chunk_size=128, chunk_overlap=32),
        ]
    ),
    embedding_cls=HuggingFaceEmbeddings,
    embedding_kwargs={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": batch_size},
    },
    vector_store=None,  # Uses FAISS by default
    search_type="similarity",
    search_kwargs={"k": 15},
    # Define multiple reranking configurations
    reranker_cls=CrossEncoderReranker,
    reranker_kwargs={
        "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda:0"},
        "top_n": List([2, 5]),
    },
    enable_gpu_search=True,
)
```

## 4. Configuring the Generator (vLLM)

Next, we define the generation component using `RFvLLMModelConfig`. We'll test two different models:
1.  `Qwen/Qwen2.5-0.5B-Instruct` (Small, fast)
2.  `Qwen/Qwen2.5-3B-Instruct` (Larger, potentially more accurate)

```python
# Config 1: 0.5B Model
vllm_config1 = RFvLLMModelConfig(
    model_config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "half",
        "gpu_memory_utilization": 0.7,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "max_model_len": 2048,
        "disable_log_stats": True,
    },
    sampling_params={"temperature": 0.8, "top_p": 0.95, "max_tokens": 512},
    rag=rag_gpu,
)

# Config 2: 3B Model
vllm_config2 = RFvLLMModelConfig(
    model_config={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dtype": "half",
        "gpu_memory_utilization": 0.7,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "max_model_len": 2048,
        "disable_log_stats": True,
    },
    sampling_params={"temperature": 0.8, "top_p": 0.95, "max_tokens": 512},
    rag=rag_gpu,
)
```

## 5. Processing and Metrics

We need to define how to process the data and how to evaluate the results.

*   **`preprocess_fn`**: Constructs the prompt with retrieved context. This function takes the user's query and the retrieved financial documents, formats them into a conversation-style prompt, and passes them to the LLM for generation.
*   **`compute_metrics_fn`**: Calculates retrieval quality metrics to ensure our chatbot is pulling in the right information:
    - **Precision**: What fraction of retrieved documents are actually relevant?
    - **Recall**: What fraction of all relevant documents did we retrieve?
    - **F1**: Harmonic mean of Precision and Recall
    - **NDCG@5**: Normalized Discounted Cumulative Gain - rewards ranking relevant documents higher
    - **MRR**: Mean Reciprocal Rank - measures how quickly we find the first relevant document

These metrics are crucial for a financial Q&A chatbot because retrieving the wrong context could lead to incorrect or misleading financial advice. By optimizing these metrics, we ensure the chatbot provides accurate, well-grounded responses.

*(Note: Full code for these functions is available in the [tutorial notebook](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-rag-fiqa.ipynb), but they essentially handle prompt formatting and standard IR metric calculations.)*

## 6. Running the Grid Search

Now we combine everything into a `RFGridSearch`. We have:
*   4 RAG configurations
*   2 vLLM configurations

This results in **8 total experiments** to run.

```python
config_set = {
    "vllm_config": List([vllm_config1, vllm_config2]),
    "batch_size": batch_size,
    "preprocess_fn": sample_preprocess_fn, # Defined in notebook
    "postprocess_fn": sample_postprocess_fn, # Defined in notebook
    "compute_metrics_fn": sample_compute_metrics_fn, # Defined in notebook
    "accumulate_metrics_fn": sample_accumulate_metrics_fn, # Defined in notebook
}

config_group = RFGridSearch(config_set)

# Run the evaluation
results = experiment.run_evals(
    config_group=config_group,
    dataset=fiqa_dataset,
    num_actors=2, # Parallelize across GPUs if available
    num_shards=4,
    seed=42,
)
```

## 7. Analyzing Results

Finally, we can view the results as a DataFrame to compare performance across all configurations.

```python
results_df = pd.DataFrame([
    {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in {**metrics_dict, 'run_id': run_id}.items()}
    for run_id, (_, metrics_dict) in results.items()
])

print(results_df)
```

This DataFrame will show you the trade-offs between model size, chunk size, and reranking depth, allowing you to pick the optimal configuration for your specific requirements (latency vs. accuracy).

## Conclusion

RapidFireAI makes it easy to treat your RAG pipeline as a hyperparameter optimization problem. By defining search spaces for both retrieval and generation, you can systematically find the best setup for your data without writing complex boilerplate code.

In this financial Q&A chatbot example, we explored 8 different configurations across chunking strategies, reranking approaches, and model sizes. This systematic approach helps you find the sweet spot between accuracy and cost - perhaps the smaller 0.5B model with aggressive reranking performs just as well as the 3B model for your use case, saving you significant inference costs.

For more hands-on practice, check out the [complete tutorial notebook](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-rag-fiqa.ipynb) and join our [Discord community](https://discord.gg/6vSTtncKNN) for support!
