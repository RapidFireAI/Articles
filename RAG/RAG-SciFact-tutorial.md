# Optimizing RAG Pipelines with RapidFireAI: A SciFact Case Study

Retrieval-Augmented Generation (RAG) systems are powerful, but optimizing them for specific domains is often a trial-and-error process involving numerous parameters. Which embedding model works best? What's the optimal chunk size? Should you use a reranker? How do different LLMs compare for your specific use case?

In this tutorial, we'll walk through a complete example of using **[RapidFireAI](https://github.com/RapidFireAI/rapidfireai)** to systematically evaluate and optimize a RAG pipeline. We'll use the **SciFact** dataset—a benchmark for verifying scientific claims—to demonstrate how to compare multiple configurations in parallel and identify the best setup for high-precision tasks.

## The Application: Scientific Claim Verification

In this tutorial, we'll optimize a RAG pipeline for **scientific claim verification**—a specialized fact-checking system for scientific literature. Unlike conversational AI chatbots that generate free-form responses, this system acts as an automated research assistant that assesses the validity of scientific statements with high precision and explainability.

### The SciFact Benchmark

To demonstrate RAG optimization, we use [SciFact](https://github.com/allenai/scifact), a dataset created by the Allen Institute for AI for scientific fact-checking. Unlike simple question-answering tasks, SciFact requires systems to understand the nuanced relationship between scientific claims and evidence.

**The Task**: Given a scientific claim like *"High cardiopulmonary fitness causes increased mortality rate"*, the system must:
1. Retrieve relevant research abstracts from a corpus of 5,000+ scientific documents
2. Analyze the relationship between the claim and evidence
3. Return a verdict: **SUPPORT**, **CONTRADICT**, or **NOINFO** (insufficient evidence)

**The Challenge**: This isn't about keyword matching—it requires semantic understanding and reasoning about scientific relationships.

### Real-World Applications

This pattern extends beyond academic benchmarks. Similar RAG systems can power:
- Medical fact-checking platforms verifying clinical claims
- Research literature review tools for hypothesis validation
- Science journalism fact-checking pipelines
- Misinformation detection systems for public health
- Legal document analysis for evidence-based decision making

Unlike conversational AI, this application demands **high precision and explainability**—every verdict must be backed by retrievable evidence.

## Why Use RapidFireAI for RAG Optimization?

The traditional approach to RAG optimization is painful:
- Run one configuration, wait for results
- Change a parameter, run again
- Manually track results across spreadsheets
- Spend days testing just a handful of combinations

**RapidFireAI transforms this process** by enabling:
- ⚡ **Parallel execution**: Test 4+ configurations simultaneously on the same hardware
- 📊 **Automatic experiment tracking**: All metrics logged to MLflow automatically
- 🎯 **Smart resource management**: Handles GPU scheduling and API rate limits intelligently
- 🚀 **16-24x speedup**: Complete comprehensive evaluations in hours instead of days

For this SciFact example, we'll compare **4 different configurations** (2 LLM models × 2 search strategies) running concurrently on 256 examples—something that would take hours sequentially but completes in ~20 minutes with RapidFireAI.

## The RAG Pipeline Architecture

The claim verification RAG pipeline we'll optimize has three key stages:

1.  **Retrieval**: Search the corpus for abstracts semantically similar to the claim. We'll compare `similarity` search (cosine similarity) vs. `MMR` (Maximal Marginal Relevance for diversity) to see which yields better coverage.

2.  **Reranking**: Raw retrieval often returns noisy results. A `CrossEncoderReranker` model re-scores the top candidates, ensuring the most relevant evidence reaches the LLM.

3.  **Generation (Reasoning)**: The LLM acts as a judge, analyzing the claim against the evidence to output a verdict (SUPPORT/CONTRADICT/NOINFO).

## Notebook Walkthrough

You can find the complete code for this tutorial in the [RapidFireAI GitHub repository](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-scifact-full-evaluation.ipynb). If you encounter any issues or need guidance while running it, join our [Discord community](https://discord.gg/6vSTtncKNN)—we're here to help!

Let's break down what the notebook does step by step:

### 1. Load and Prepare the Dataset

The SciFact dataset consists of:
- **queries.jsonl**: Scientific claims to verify
- **corpus.jsonl**: 5,000+ research abstracts (the knowledge base)
- **qrels.tsv**: Ground truth relevance judgments

The notebook loads these into a Hugging Face `Dataset` format and extracts the ground truth labels (SUPPORT/CONTRADICT/NOINFO) for evaluation:

```python
# Load queries and process labels
data = []
with open("datasets/scifact/queries.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
# ... label processing logic ...

# Create Hugging Face Dataset
scifact_dataset = Dataset.from_dict({
    "query": [d["text"] for d in data],
    "query_id": [d["_id"] for d in data],
    "label": [d["label"] for d in data],
})

# Load Ground Truth (Qrels)
qrels = pd.read_csv("datasets/scifact/qrels.tsv", sep="\t")
```

### 2. Initialize the Experiment

Every RapidFireAI evaluation starts by creating an `Experiment` object that tracks all configurations, runs, and results:

```python
from rapidfireai import Experiment

experiment = Experiment(
    experiment_name="exp1-scifact-full-evaluation", 
    mode="evals"  # Use "evals" mode for RAG/context engineering
)
```

This automatically sets up MLflow tracking and experiment logging.

### 3. Configure the RAG Pipeline

Here's where RapidFireAI's power becomes apparent. Using `RFLangChainRagSpec`, you define your RAG pipeline with familiar LangChain components:

```python
from rapidfireai.evals.automl import RFLangChainRagSpec, List

rag_config = RFLangChainRagSpec(
    document_loader=DirectoryLoader(...),  # Load corpus.jsonl
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=32
    ),
    embedding_cls=OpenAIEmbeddings,
    embedding_kwargs={"model": "text-embedding-3-small"},
    search_type=List(["similarity", "mmr"]),  # 🎯 Test BOTH automatically!
    search_kwargs={"k": 15},  # Retrieve top 15 candidates
    reranker_cls=CrossEncoderReranker,
    reranker_kwargs={
        "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "top_n": 5  # Rerank to top 5
    },
)
```

**The Magic**: Notice `search_type=List(["similarity", "mmr"])`. By wrapping options in `List()`, RapidFireAI automatically creates separate configurations to test both approaches in parallel!


### 4. Define Data Processing & Metrics

RapidFireAI lets you customize how data flows through your pipeline using three callback functions:

```python
def sample_preprocess_fn(batch, rag, prompt_manager):
    # 1. Retrieve context
    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)
    serialized_context = rag.serialize_documents(all_context)
    
    # 2. Construct prompts
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

def sample_postprocess_fn(batch):
    # Extract verdict (SUPPORT/CONTRADICT) from LLM output using Regex
    batch["answer"] = [extract_solution(ans) for ans in batch["generated_text"]]
    return batch

def sample_compute_metrics_fn(batch):
    # Calculate NDCG, Precision, F1, etc.
    return {
        "Precision": {"value": precision_score(...)},
        "NDCG@5": {"value": ndcg_score(...)},
        # ... other metrics
    }
```

**Preprocessing** (`sample_preprocess_fn`): Retrieves context and formats the prompt.
**Postprocessing** (`sample_postprocess_fn`): Extracts the structured verdict from the LLM's text response.
**Metrics** (`sample_compute_metrics_fn`): Computes retrieval (NDCG) and generation (F1) scores per batch. These are automatically aggregated.

### 5. Define the Configuration Grid

Now we set up the "AutoML" search space. We want to compare:
- **2 LLM models**: `gpt-4o-mini` (fast/cheap) vs. `gpt-4o` (powerful)
- **2 search strategies**: `similarity` vs. `mmr` (from step 3)

This creates **4 total configurations** (2 × 2):

```python
from rapidfireai.evals.automl import RFOpenAIAPIModelConfig, RFGridSearch

openai_config1 = RFOpenAIAPIModelConfig(
    model_config={"model": "gpt-4o-mini"},
    rag=rag_config,  # Includes both search types!
)

openai_config2 = RFOpenAIAPIModelConfig(
    model_config={"model": "gpt-4o"},
    rag=rag_config,
)

config_group = RFGridSearch({
    "openai_config": List([openai_config1, openai_config2]),
    "preprocess_fn": sample_preprocess_fn,
    "postprocess_fn": sample_postprocess_fn,
    "compute_metrics_fn": sample_compute_metrics_fn,
    "batch_size": 32,
})
```

### 6. Run the Evaluation

With one command, RapidFireAI orchestrates everything:

```python
results = experiment.run_evals(
    config_group=config_group,
    dataset=scifact_dataset,
    num_actors=2,      # Parallel workers
    num_shards=4,      # Data chunks for interleaved execution
    seed=42,
)
```

**What happens**: RapidFireAI automatically:
- Divides the dataset into 4 shards
- Runs all 4 configurations in parallel, swapping between them shard-by-shard
- Manages API rate limits and cost optimization
- Tracks all metrics in real-time
- Logs everything to MLflow

**Cost & Time**: ~$5 for 256 examples, completing in ~20 minutes with parallel execution.

### 7. Analyze Results

Results are returned as a structured dictionary, easily converted to a DataFrame:

```python
results_df = pd.DataFrame([
    {**metrics_dict, 'run_id': run_id}
    for run_id, (_, metrics_dict) in results.items()
])
```

**Example output**:

| run_id | Model | Search | NDCG@5 | F1 Score | Precision | Recall |
|--------|-------|--------|--------|----------|-----------|--------|
| 1 | gpt-4o-mini | similarity | 0.72 | 0.81 | 0.84 | 0.78 |
| 2 | gpt-4o-mini | mmr | 0.75 | 0.79 | 0.82 | 0.76 |
| 3 | gpt-4o | similarity | 0.73 | 0.86 | 0.88 | 0.84 |
| 4 | gpt-4o | mmr | 0.76 | 0.87 | 0.89 | 0.85 |

**Interpretation**: In this example, `gpt-4o` + `mmr` gives the best overall performance—better retrieval (NDCG) and more accurate reasoning (F1).

## Key Takeaways

This tutorial demonstrates how RapidFireAI transforms RAG optimization from a tedious manual process into a systematic, scalable experiment:

1. **Parallel Evaluation**: Test multiple configurations simultaneously instead of sequentially
2. **Automatic Tracking**: All metrics and artifacts logged to MLflow automatically
3. **Declarative Configuration**: Define your search space with simple Python objects
4. **Domain Flexibility**: The same pattern applies to medical, legal, financial, and other specialized RAG systems

**Key Insight**: This example demonstrates how RAG systems excel at evidence-based decision making—fact verification, legal analysis, medical diagnostics—where precision and explainability are paramount.

## Try It Yourself

### Getting Started

1. **Install RapidFireAI**:
   ```bash
   pip install rapidfireai
   rapidfireai init --evals
   ```
   Full installation instructions are in the [GitHub README](https://github.com/RapidFireAI/rapidfireai#install-and-get-started).

2. **Run the Notebook**:
   - Clone the repo: `git clone https://github.com/RapidFireAI/rapidfireai.git`
   - Navigate to: `tutorial_notebooks/rag-contexteng/`
   - Open: `rf-tutorial-scifact-full-evaluation.ipynb`

3. **Start Experimenting**:
   - Try different embedding models (`text-embedding-3-large`, BioBERT)
   - Test more search strategies or rerankers
   - Expand to the full dataset (1,109 examples)
   - Adapt the pattern to your own domain

### Resources

- 📖 **Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai)
- 💬 **Discord Community**: [Join for help & discussions](https://discord.gg/6vSTtncKNN)
- 🐛 **Issues & Feature Requests**: [GitHub Issues](https://github.com/RapidFireAI/rapidfireai/issues)
- ⭐ **Star the repo** if you find this useful!

### Adapting to Your Use Case

To apply this to your own RAG system:
1. Replace the SciFact dataset with your documents and queries
2. Customize the prompt instructions for your domain
3. Define metrics relevant to your task (accuracy, recall, custom business metrics)
4. Expand the configuration grid with your hyperparameters

Example configuration grid for a legal document RAG:
```python
config_set = {
    "embedding_model": List(["text-embedding-3-small", "text-embedding-3-large"]),
    "chunk_size": List([256, 512, 1024]),
    "model": List(["gpt-4o", "claude-3-opus"]),
}
# This creates 2 × 3 × 2 = 12 configurations to test in parallel!
```

Happy optimizing! 🚀
