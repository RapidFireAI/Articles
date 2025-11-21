# Optimizing RAG Pipelines with RapidFireAI: A SciFact Case Study

Retrieval-Augmented Generation (RAG) systems are powerful, but optimizing them for specific domains is often a trial-and-error process involving numerous parameters. Which embedding model works best? What's the optimal chunk size? Should you use a reranker?

In this tutorial, we'll show you how to use **RapidFireAI** to systematically evaluate and optimize a RAG pipeline. We'll use the **SciFact** dataset—a benchmark for verifying scientific claims—to demonstrate how to build, test, and refine a RAG system for high-precision tasks.

## The Real-World Application: Scientific Fact Verification

**This is NOT a chatbot.** Instead, this RAG system is a **scientific claim verification tool**—a specialized fact-checking system designed for scientific literature. Think of it as an automated research assistant that can assess the validity of scientific statements.

### How It Works in Practice

When you input a scientific claim like *"Vitamin D supplementation reduces the risk of cancer,"* the system:

1. **Searches** through thousands of scientific abstracts to find relevant research
2. **Ranks** the evidence by relevance using semantic understanding
3. **Reasons** through the evidence using an LLM to determine if it supports or contradicts the claim
4. **Returns** a verdict: SUPPORT, CONTRADICT, or NOINFO (insufficient evidence)

### Real-World Use Cases

This type of RAG system could be deployed for:

*   **Medical Fact-Checking**: Verifying clinical claims in health articles or social media
*   **Research Literature Review**: Helping researchers quickly validate or challenge hypotheses against existing literature
*   **Science Journalism**: Enabling journalists to verify scientific statements before publication
*   **Misinformation Detection**: Identifying false or misleading scientific claims in public discourse
*   **Evidence-Based Medicine**: Supporting healthcare professionals in validating treatment claims

Unlike conversational AI systems that generate free-form responses, this application requires high precision and explainability—it must cite specific evidence and reason transparently about the relationship between claims and evidence.

## The RAG Pipeline for Claim Verification

To solve this, we build a RAG pipeline with three key stages:

1.  **Retrieval**: We search the corpus for abstracts that are semantically similar to the claim. We'll test different search methods (Similarity vs. MMR) to see which yields better coverage.
2.  **Reranking**: Raw retrieval often returns noisy results. We use a Cross-Encoder model to re-score the top results, ensuring the most relevant evidence is passed to the LLM.
3.  **Generation (Reasoning)**: Finally, we pass the claim and the top-ranked evidence to an LLM (like GPT-4o). The LLM acts as a judge, reasoning through the evidence to output a final verdict.

## Prerequisites

Before we begin, ensure you have:
1.  **RapidFireAI** installed in your environment.
2.  An **OpenAI API key** (or access to another supported LLM provider).
3.  The **SciFact dataset** (queries, corpus, and qrels) available locally.

## Step 1: Setting up the Experiment

First, we initialize a RapidFireAI `Experiment`. This object tracks our configurations, runs, and results, making it easy to compare different approaches.

```python
from rapidfireai import Experiment

experiment = Experiment(experiment_name="exp1-scifact-full-evaluation", mode="evals")
```

## Step 2: Preparing the Data

We need to load our dataset. For SciFact, we have queries (scientific claims), a corpus of abstracts, and relevance judgments (qrels).

We'll load the queries and format them into a Hugging Face `Dataset`. We also handle some metadata cleanup to extract labels (SUPPORT, CONTRADICT, NOINFO).

```python
import json
import pandas as pd
from datasets import Dataset

# Load queries
data = []
with open("datasets/scifact/queries.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Process labels
for d in data:
    if d["metadata"]:
        for info in d["metadata"].values():
            tags = set([meta["label"] for meta in info])
            d["label"] = tags.pop()
    else:
        d["label"] = "NOINFO"

# Create Dataset
scifact_dataset = Dataset.from_dict({
    "query": [d["text"] for d in data],
    "query_id": [d["_id"] for d in data],
    "label": [d["label"] for d in data],
})

# Load Qrels (Ground Truth)
qrels = pd.read_csv("datasets/scifact/qrels.tsv", sep="\t")
qrels = qrels.rename(columns={"query-id": "query_id", "corpus-id": "corpus_id", "score": "relevance"})
```

## Step 3: Configuring the RAG Pipeline

This is where RapidFireAI shines. We can define a complex RAG pipeline using `RFLangChainRagSpec`. In this example, we're setting up a GPU-accelerated pipeline with:

*   **Loader**: `DirectoryLoader` with `JSONLoader` to ingest the corpus.
*   **Splitter**: `RecursiveCharacterTextSplitter` (chunk size 512).
*   **Embeddings**: `OpenAIEmbeddings` (text-embedding-3-small).
*   **Retriever**: A hybrid search (similarity + MMR) with `k=15`.
*   **Reranker**: `CrossEncoderReranker` (ms-marco-MiniLM-L6-v2) to refine the top 5 results.

```python
from rapidfireai.evals.automl import RFLangChainRagSpec, List
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = input("Enter your OpenAI API key: ")

def metadata_func(record, metadata):
    metadata["corpus_id"] = int(record.get("_id"))
    metadata["title"] = record.get("title")
    return metadata

def custom_template(doc) -> str:
    return f"{doc.metadata['title']}: {doc.page_content}"

rag_gpu = RFLangChainRagSpec(
    document_loader=DirectoryLoader(
        path="datasets/scifact/",
        glob="corpus.jsonl",
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",
            "content_key": "text",
            "metadata_func": metadata_func,
            "json_lines": True,
            "text_content": False,
        },
        sample_seed=1337,
    ),
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2", chunk_size=512, chunk_overlap=32
    ),
    embedding_cls=OpenAIEmbeddings,
    embedding_kwargs={"model": "text-embedding-3-small", "api_key": OPENAI_API_KEY},
    search_type=List(["similarity", "mmr"]),  # Testing 2 search types automatically!
    search_kwargs={"k": 15},
    reranker_cls=CrossEncoderReranker,
    reranker_kwargs={
        "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda:0"},
        "top_n": 5,
    },
    enable_gpu_search=True,
    document_template=custom_template,
)
```

Notice `search_type=List(["similarity", "mmr"])`. RapidFireAI will automatically treat this as a hyperparameter and evaluate both options!


## Step 3.5: Defining Data Processing Functions

We need a few helper functions to format the inputs for the LLM and process the outputs.

`sample_preprocess_fn` constructs the prompt by combining the query and the retrieved context. `sample_postprocess_fn` extracts the answer (SUPPORT/CONTRADICT/NOINFO) from the model's output.

```python
import re

INSTRUCTIONS = """
You are a helpful assistant that can verify scientific claims.
... (Full instructions in notebook) ...
Response: The evidence suggests ... #### SUPPORT
"""

def sample_preprocess_fn(batch, rag, prompt_manager):
    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)
    serialized_context = rag.serialize_documents(all_context)
    
    return {
        "prompts": [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Claim:\n{q}.\nEvidence:\n{c}.\nYour response:"}
            ]
            for q, c in zip(batch["query"], serialized_context)
        ],
        "retrieved_documents": [[d.metadata["corpus_id"] for d in docs] for docs in all_context],
        **batch,
    }

def extract_solution(answer):
    solution = re.search(r"####\s*(SUPPORT|CONTRADICT|NOINFO)", answer, re.IGNORECASE)
    return solution.group(1).upper() if solution else "INVALID"

def sample_postprocess_fn(batch):
    batch["ground_truth_documents"] = [
        qrels[qrels["query_id"] == qid]["corpus_id"].tolist() for qid in batch["query_id"]
    ]
    batch["answer"] = [extract_solution(ans) for ans in batch["generated_text"]]
    return batch
```

## Step 4: Defining Custom Metrics

To measure success, we need metrics. For SciFact, we care about both retrieval quality (NDCG, MRR) and generation quality (Precision, Recall, F1 of the final answer).

We define a `sample_compute_metrics_fn` that calculates these based on the retrieved documents and the generated answer.

```python
import math

def compute_ndcg_at_k(retrieved_docs, expected_docs, k=5):
    # ... implementation of NDCG ...
    pass

def compute_rr(retrieved_docs, expected_docs):
    # ... implementation of MRR ...
    pass

def sample_compute_metrics_fn(batch):
    # Calculate Precision, Recall, F1, NDCG, MRR
    # ...
    return {
        "Precision": {"value": ...},
        "Recall": {"value": ...},
        "F1 Score": {"value": ...},
        "NDCG@5": {"value": ...},
        "MRR": {"value": ...},
    }
```

## Step 5: Running Multi-Config Evaluations

Now for the "AutoML" part. We want to compare different LLMs for the generation step. We define two `RFOpenAIAPIModelConfig` objects: one using `gpt-4o-mini` (faster, cheaper) and one using `gpt-4o` (more powerful).

We combine these into a `RFGridSearch` configuration set.

```python
from rapidfireai.evals.automl import RFOpenAIAPIModelConfig, RFGridSearch

# Config 1: GPT-4o-mini
openai_config1 = RFOpenAIAPIModelConfig(
    client_config={"api_key": OPENAI_API_KEY},
    model_config={"model": "gpt-4o-mini", "reasoning_effort": "high"},
    rag=rag_gpu,
)

# Config 2: GPT-4o
openai_config2 = RFOpenAIAPIModelConfig(
    client_config={"api_key": OPENAI_API_KEY},
    model_config={"model": "gpt-4o", "reasoning_effort": "medium"},
    rag=rag_gpu,
)

config_set = {
    "openai_config": List([openai_config1, openai_config2]),
    "batch_size": 32,
    "preprocess_fn": sample_preprocess_fn,
    "postprocess_fn": sample_postprocess_fn,
    "compute_metrics_fn": sample_compute_metrics_fn,
    # ... other params
}

config_group = RFGridSearch(config_set)
```

Finally, we run the evaluation!

```python
results = experiment.run_evals(
    config_group=config_group,
    dataset=scifact_dataset,
    num_actors=2,
    num_shards=4,
    seed=42,
)
```

## Step 6: Analyzing Results

RapidFireAI returns the results as a dictionary, which we can easily convert to a Pandas DataFrame for analysis.

```python
results_df = pd.DataFrame([
    {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in {**metrics_dict, 'run_id': run_id}.items()}
    for run_id, (_, metrics_dict) in results.items()
])

print(results_df)
```

This DataFrame will show you the performance metrics (Precision, Recall, NDCG, etc.) for every combination of parameters (e.g., `gpt-4o` + `similarity` search vs. `gpt-4o-mini` + `mmr` search).

## Conclusion

By using RapidFireAI, we turned a complex optimization problem into a structured experiment. We defined our search space (models, search types), set up our metrics, and let the system handle the execution and tracking. This allows ML engineers to focus on interpreting results and iterating on their RAG strategies rather than writing boilerplate evaluation code.

**Key Takeaway**: This tutorial demonstrates how RAG systems extend beyond conversational AI. By applying RAG to scientific fact verification, we've built a specialized tool that combines information retrieval with reasoning capabilities—a pattern that can be adapted for legal document analysis, medical diagnostics, financial due diligence, and other domains requiring evidence-based decision making.

## Next Steps

To explore this further:
*   Experiment with different reranking models to improve precision
*   Test other LLM providers (Anthropic, Cohere) to compare reasoning quality
*   Try domain-specific embedding models (e.g., BioBERT for medical claims)
*   Extend the system to provide evidence snippets alongside verdicts
*   Scale up to the full SciFact dataset for production-ready evaluation
