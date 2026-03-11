# Show HN: RapidFire AI – Run 100+ RAG experiments in parallel, even on a single GPU

Hey HN,

We built **[RapidFire AI](https://github.com/RapidFireAI/rapidfireai)**, an open-source framework that lets you compare dozens (or hundreds) of RAG and context engineering configurations in parallel — without needing a GPU cluster.

---

**The problem:** Tuning a RAG pipeline means experimenting with chunk sizes, embedding models, retrieval strategies, reranking thresholds, prompt schemes, generator models, and more. With traditional tools, you run these sequentially, wait for each to finish on the full dataset, and then compare. That's painfully slow and wastes tokens/compute on configs you'd have killed after seeing the first 10% of results.

**What we do differently:** RapidFire AI shards your eval dataset and schedules all configs one shard at a time, cycling through them with efficient swapping. You get running metric estimates with confidence intervals in real time (based on online aggregation from the database systems literature). Spot a bad config early? Stop it. See a promising one? Clone it and tweak knobs on the fly — all without restarting anything.

On a beefy machine you can comfortably run 100+ configs in a single experiment. Want to see it in action without installing anything? We have a **[Google Colab tutorial](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb)** that runs 4 RAG retrieval configs in parallel on a free Colab GPU — zero local setup, under 2 minutes to get started. It builds a financial Q&A pipeline on the FiQA dataset, grid-searches over chunk sizes and reranker settings, and shows live metrics with confidence intervals as the configs run. If you're only calling OpenAI or other closed APIs, you don't even need a GPU at all.

---

### Key features

- `pip install rapidfireai` — pure Python, works on CPU-only, single-GPU, or multi-GPU
- Wraps **LangChain** for retrieval/reranking and supports **vLLM + OpenAI** for generation
- **Interactive Control (IC) Ops:** stop, resume, clone-modify configs mid-run from a dashboard or notebook
- **Online aggregation** with confidence intervals so you can make statistically informed early-stopping decisions
- **Grid search and random search** over any knob (chunk size, top-k, reranker model, prompt template, generator params, etc.)
- **MLflow-based** metrics dashboard with real-time plots

---

### Example speedup

16 RAG configs × 400 queries at ~10s/query takes **~18 hours** sequentially. With RapidFire AI + IC Ops (stop poor performers early, clone winners), we got it down to **~4 hours** on the same machine — a **4.7× improvement** while exploring more configs and reaching better final metrics.

---

### Try it

- **Docs:** https://oss-docs.rapidfire.ai
- **Tutorial notebooks** (FiQA financial QA, GSM8K math reasoning, SciFact claim verification): https://github.com/RapidFireAI/rapidfireai/tree/main/tutorial_notebooks/rag-contexteng
- **Google Colab** (zero setup, under 2 min): https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb

---

We'd love feedback on what knobs/integrations matter most to you. Gemini and Claude API support for the generator is coming soon. Happy to answer questions here.
