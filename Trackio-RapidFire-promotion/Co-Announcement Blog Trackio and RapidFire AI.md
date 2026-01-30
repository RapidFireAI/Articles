# **Announcing New Trackio Integration: Moving from LLM Observability to Real-time Control and Rapid Experimentation**

## **TL;DR**

Large language model (LLM) development increasingly depends on fast, systematic experimentation across retrieval, training, and evaluation choices. Yet many teams still rely on workflows where experiments are run one at a time and evaluation tools only surface results after the fact.

Today, we’re excited to share a new integration between Trackio and RapidFire AI that changes this dynamic. With this integration, Trackio provides a lightweight, real-time visualization layer for your experiments. AI engineers can now define, launch, and compare multiple RAG and fine-tuning configurations in parallel—while directly analyzing tradeoffs across accuracy, latency, and cost.

This post is co-authored by Hugging Face contributors and the RapidFire AI team, and reflects a shared goal: making LLM experimentation more systematic, reproducible, and aligned with how models are actually built and deployed.

## **The Problem with Today’s LLM Experimentation Workflows**

Most LLM experimentation today is still fragmented and sequential. Engineers typically modify one parameter at a time, rerun pipelines, and then inspect results after the fact. Evaluation tools surface metrics, but they sit downstream of the actual experimentation logic.

As a result:

* Experimentation cycles are slow and manual  
* Difficulty comparing configurations consistently  
* Fragmented experimentation logic spread across scripts and notebooks  
* Optimization efforts that focus on one metric at a time

As LLM systems grow more complex—especially with RAG and post-training workflows—this approach becomes increasingly limiting.

## **Trackio for Real-Time Experiment Visualization**

With the Trackio–RapidFire AI integration, Trackio provides a free, open-source and lightweight dashboard for visualizing your experiments as they run. Instead of waiting for experiments to complete, engineers can monitor progress in real-time.

From Trackio's dashboard, users can:

* View multiple RAG or fine-tuning configurations simultaneously  
* Compare metrics side-by-side across all parallel runs  
* Track training loss, evaluation metrics, and hyperparameters in real-time

## **RapidFire AI: Executing Experiments at Scale**

RapidFire AI provides the execution layer that makes interactive experimentation possible. Once experiment configurations are defined, RapidFire AI orchestrates their execution in parallel, ensuring that results are both reproducible and directly comparable.

This includes:

* Hyper-parallel execution of RAG and fine-tuning experiments  
* Automated handling of experiment variants and parameters  
* Structured result collection aligned with evaluation metrics

This integration enables Trackio to focus on insight and comparison while RapidFire AI handles scale, speed, and experimental rigor.

## **How RapidFire AI Integrates with Trackio**

RapidFire AI is an experiment execution framework for LLM fine-tuning and post-training that enables hyper-parallel training across multiple configurations. This is where Trackio becomes invaluable. RapidFire AI has built native Trackio support into its metric logging system:

* Automatic Metric Capture: Training metrics (loss, learning rate, gradient norms) are automatically logged to Trackio during training  
* Evaluation Metrics: Custom evaluation metrics (ROUGE, BLEU, accuracy) are captured at each evaluation step  
* Hyperparameter Tracking: Each run's configuration is logged, making it easy to understand what parameters produced which results  
* Real-Time Dashboard: View all your parallel runs in Trackio's dashboard as they train

## **Practical Workflows Enabled by the Integration**

**RAG Experimentation**: Engineers can explore multiple RAG configurations in parallel, including variations in:

* Chunk size  
* Embedding models  
* Retrieval and reranking approaches  
* Prompts

Results can be compared side by side across retrieval metrics, task accuracy, latency, and inference cost—making tradeoffs immediately visible.

**Fine-Tuning and Post-Training:** For fine-tuning workflows, the integration supports parallel sweeps across:

* LoRA rank and adapter settings  
* Learning rates and training schedules  
* Dataset variants and preprocessing choices

Each resulting model is evaluated using consistent metrics, enabling clear and reproducible comparisons.

![Trackio dashboard showing RapidFire AI fine-tuning metrics](./trackio-screenshot-sft.png)

*Trackio dashboard comparing 4 fine-tuning runs with different hyperparameters. The plots show training loss, validation loss, learning rate schedules, and ROUGE-L scores—making it easy to identify which configuration (Run 4, in orange) achieves the lowest loss and best generation quality.*

## **Multi-Objective Optimization in Practice**

Production LLM systems are inherently multi-objective. Accuracy improvements that dramatically increase latency or cost are often unacceptable, while lower-cost configurations may fail quality thresholds.

By combining Trackio’s interface with RapidFire AI’s parallel execution, teams can monitor and reason about these tradeoffs directly. This enables informed engineering decisions based on real data rather than intuition or trial-and-error.

## **Why This Matters for LLM Teams**

For individual engineers, the integration shortens feedback loops and reduces reliance on brittle, ad-hoc experimentation scripts. For teams, it creates shared, reproducible artifacts that make collaboration and review easier. For organizations, it reduces experimentation cost while increasing confidence in production deployments.

By aligning experimentation tools with real-world workflows, Trackio and RapidFire AI aim to make rigorous LLM development more accessible to the community.

## **Get Started**

Ready to try the integration? Here's how to get started:

**1. Install RapidFire AI:**

```bash
pip install rapidfireai
```

Trackio is included as a dependency of RapidFire AI, so no separate installation is needed.

**2. Try the tutorial notebooks:**

We've created hands-on tutorials that walk through complete experiments with Trackio tracking. Each tutorial demonstrates configuring Trackio, running parallel experiments, and viewing results in the dashboard.

**Fine-Tuning (SFT) Tutorials:**

* [SFT with Trackio \- Colab Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/trackio/rf-colab-tutorial-sft-trackio.ipynb) \- Run directly in Google Colab, no local installation required
* [SFT with Trackio \- Local Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/trackio/rf-tutorial-sft-trackio.ipynb) \- For local development with full RapidFire AI features

**RAG Optimization Tutorials:**

* [RAG FiQA with Trackio \- Colab Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/trackio/rf-colab-tutorial-rag-fiqa-trackio.ipynb) \- Run directly in Google Colab, no local installation required
* [RAG FiQA with Trackio \- Local Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/trackio/rf-tutorial-rag-fiqa-trackio.ipynb) \- For local development with full RapidFire AI features

**3. Learn more:**

* [RapidFire AI Integration Guide](https://github.com/gradio-app/trackio/blob/main/docs/source/rapidfireai_integration.md) \- Official integration documentation with setup instructions and examples
* [Trackio GitHub Repository](https://github.com/gradio-app/trackio) \- Full documentation and examples  
* [Trackio Documentation](https://huggingface.co/docs/trackio/index) \- API reference and guides  
* [RapidFire AI Documentation](https://oss-docs.rapidfire.ai/) \- Getting started with RapidFire AI  
* [RapidFire AI GitHub](https://github.com/RapidFireAI/rapidfireai) \- Source code and more tutorials

## **Conclusion**

By integrating Trackio and RapidFire AI, we've combined hyper-parallel experiment execution with free, open-source experiment tracking. ML engineers can now run many configurations simultaneously while maintaining full visibility into every run's progress.

We believe experiment tracking should be accessible to everyone—not locked behind pricing tiers or requiring complex infrastructure. Trackio embodies this philosophy, and we're excited to bring it to the RapidFire AI community.

We invite you to try the integration, share feedback, and help shape the future of LLM experimentation tooling.