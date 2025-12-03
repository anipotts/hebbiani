# Consensus Evaluation Experiments

This directory contains implementations of **Hebbia's consensus-based evaluation framework** from the paper ["Who Evaluates the Evaluator: Reaching Autonomous Consensus on Agentic Outputs"](https://arxiv.org/abs/2406.16602).

## 🎯 What This Does

This framework evaluates LLM agents by:

1. **Generating responses** from multiple agents at different temperatures
2. **Scoring responses** using an evaluator LLM with logprob-based uncertainty quantification
3. **Statistical analysis** via permutation tests to determine significance
4. **Comparing agents** on multiple criteria (insightfulness, accuracy, etc.)

## 📁 File Structure

```
experiments/
├── README.md                          # This file
├── config.py                          # experiment config
├── consensus_core.py                  # Shared core logic (don't modify)
├── consensus_eval_sync_refactored.py  # Synchronous version
├── consensus_eval_async_refactored.py # Async version (recommended)
└── examples/
    └── config_example.py              # Example configurations
```

## 🚀 Quick Start

### 1. Set Your API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2. Customize Your Experiment

Edit `config.py` to define:

- **Agents** you want to compare (`get_agents()`)
- **Questions** to evaluate (`get_questions()`)
- **Experiment parameters** (`get_experiment_config()`)

### 3. Run the Experiment

**For small experiments (< 100 calls):**

```bash
python consensus_eval_sync_refactored.py
```

**For larger experiments (recommended):**

```bash
python consensus_eval_async_refactored.py
```

### 4. View Results

Results are saved to `experiment_logs/`:

- `*_run_records.jsonl` - All evaluation records
- `*_significance.json` - Statistical significance tests
- `async_leaderboard.json` - Agent comparison leaderboard (async only)

## ⚙️ Configuration Guide

### Experiment Parameters

In `config.py`, customize:

```python
def get_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        # Models
        api_model_agent="gpt-4o",      # Model for generating responses
        api_model_eval="gpt-4o",       # Model for evaluation

        # Experiment size
        n_bootstrap_runs=10,           # Start small (10) for testing
        temperatures=[0.1, 0.4, 0.8],  # Temperature values to test
        perm_test_iterations=5_000,     # Statistical test iterations

        # Concurrency (async only)
        max_concurrent_requests=8,      # Parallel API calls

        # Document context (optional)
        document_context=None,          # For grounded Q&A experiments
    )
```

### Defining Agents

```python
def get_agents():
    from consensus_core import AgentConfig

    agent_baseline = AgentConfig(
        name="baseline_H0",
        model="gpt-4o",
        prompt="You are the baseline agent. Provide concise, factual answers.",
    )

    agent_candidate = AgentConfig(
        name="candidate_HA",
        model="gpt-4o",
        prompt="You are the improved candidate agent. Provide highly detailed, analytic answers.",
    )

    return agent_baseline, agent_candidate
```

### Defining Questions

```python
def get_questions():
    from consensus_core import Question

    return [
        Question(
            id="q1",
            text="Summarize this quarter's revenue drivers.",
            criterion="insightfulness",  # What you're evaluating
        ),
        # Add more questions...
    ]
```

## 📊 Understanding Results

### Expected Score

Each response gets an **expected score** (1-5) calculated from logprobs:

- Extracts probability distribution over tokens ["1", "2", "3", "4", "5"]
- Computes weighted average: `E[score] = Σ(score × probability)`

### Significance Tests

Permutation tests compare agents:

- **p-value < 0.05**: Statistically significant difference
- **p-value ≈ 1.0**: No significant difference (agents perform similarly)

### Leaderboard (Async Only)

Ranks agents by mean expected score:

```
candidate_HA | mean=4.123 | std=0.456 | n=90
baseline_H0  | mean=3.789 | std=0.512 | n=90
```

## 💰 Cost Estimation

For **N bootstrap runs × Q questions × 2 agents × T temperatures × 2 API calls**:

- **Small (N=10, Q=3, T=3)**: ~360 calls ≈ **$0.50-1.00**
- **Medium (N=50, Q=3, T=3)**: ~1,800 calls ≈ **$3-4**
- **Large (N=100, Q=5, T=3)**: ~6,000 calls ≈ **$10-15**

Using GPT-4o pricing: $3/million input tokens, $10/million output tokens.

## 🔧 Sync vs Async

| Feature         | Sync                         | Async                     |
| --------------- | ---------------------------- | ------------------------- |
| **Speed**       | Sequential (slow)            | Parallel (10x+ faster)    |
| **Use Case**    | Small experiments, debugging | Production runs           |
| **Concurrency** | N/A                          | Configurable (default: 8) |
| **Leaderboard** | ❌                           | ✅                        |

**Recommendation**: Use async for all experiments unless debugging.

## 🧠 Needle-in-the-Transcript: NVDA Guidance Extraction under Distractors

This specialized experiment measures how reliably different LLM agents can extract **exact NVIDIA guidance numbers** when transcripts contain heavy numerical distractors (YoY deltas, prior-quarter comps, ranges, etc.).

- **Corpus**: `NVDA_Q2_FY2026` + `NVDA_Q3_FY2026` transcripts in `data/`.
- **Agents**:
  - `baseline_H0`: terse extractor that outputs only the value or `UNKNOWN`.
  - `candidate_HA`: reasoning-heavy agent that explains in 1–3 sentences and ends with `FINAL: ...`.
- **Questions per transcript**:
  1. Next-quarter revenue guidance
  2. Next-quarter EPS guidance range
  3. Reported data-center revenue this quarter
- **Context wiring**: Each agent receives the full transcript plus metadata header (quarter, fiscal year, call date, doc_id).
- **Scoring**: Evaluator LLM focuses on verbatim numeric precision (correct quarter, metric, units, range bounds). Scores 1–5 with logprob-derived expected scores.
- **Optional ground truth**: `ground_truth.py` ships with placeholders so you can plug in official guidance numbers and compare `exact_match` / `normalized_match` flags alongside LLM scores.

### Run It

```bash
# sync (debug / small-scale)
python experiments/consensus_eval_sync_refactored.py

# async (recommended for N ≥ 50 bootstrap runs)
python experiments/consensus_eval_async_refactored.py
```

Both variants emit:

- `experiment_logs/sync_run_records.jsonl` or `async_run_records.jsonl`: per-sample logs containing `doc_id`, `question_text`, `agent_model`, `temperature`, `raw_response`, `final_answer`, evaluator score, and optional ground-truth matches.
- `experiment_logs/sync_significance.json` or `async_significance.json`: bootstrap means + permutation-test p-values per question.
- Async path additionally saves `experiment_logs/async_leaderboard.json`.

### Smoke-Test a Subset

For quick validation (single transcript or question), use the environment filters configured in `config.py`:

```bash
export EXPERIMENT_DOC_IDS="NVDA_Q2_FY2026"
export EXPERIMENT_QUESTION_SUFFIXES="q1_revenue_guidance"
python experiments/consensus_eval_sync_refactored.py
```

Unset those variables (e.g., `unset EXPERIMENT_DOC_IDS EXPERIMENT_QUESTION_SUFFIXES`) before running the full experiment.

## 🕸️ RabbitMQ Experiment Runner

Phase 1 upgrades the evaluation loop into a proper queue-driven backend. The goal is to mirror how Hebbia-style infra fans out long-running tasks, adds retries, and isolates poison messages in a DLQ.

### 1. Start RabbitMQ

```bash
pip install -r requirements.txt
docker-compose up -d rabbitmq
```

The management UI lives at http://localhost:15672 (guest/guest).

### 2. Enqueue tasks

Use the producer to enumerate `(run, doc, question, agent, temperature)` combos:

```bash
# optional smoke-test overrides
export EXPERIMENT_DOC_IDS="NVDA_Q2_FY2026"
export EXPERIMENT_QUESTION_SUFFIXES="q1_revenue_guidance"
export N_BOOTSTRAPS=2
export TEMPERATURES_JSON='[0.1]'

python experiments/queue_producer.py
```

Messages land on the durable `extraction_tasks` queue with a dead-letter exchange configured automatically.

### 3. Run one or more workers

```bash
export OPENAI_API_KEY=...
python experiments/queue_worker.py
```

Each worker:

- Pulls tasks with `prefetch=1`
- Calls the same pipeline helpers as the sync/async scripts
- Retries up to 3 times on transient errors (with exponential backoff baked into `pipeline.py`)
- Logs records to `experiment_logs/run_records_rabbitmq.jsonl`
- Publishes exhausted tasks (plus error metadata) to the DLQ `failed_tasks_dlq`

### 4. Inspect the DLQ

```bash
python experiments/dlq_monitor.py
```

This script peeks at DLQ messages (without draining them) and reports the top error types, agents, and questions so you can debug ingestion pathologies quickly.

### Handy overrides

- `EXPERIMENT_DOC_IDS`, `EXPERIMENT_QUESTION_SUFFIXES`
- `N_BOOTSTRAPS`
- `TEMPERATURES_JSON` (JSON array, e.g. `[0.1,0.4,0.8]`)
- `MAX_CONCURRENT_REQUESTS` (drives async runner concurrency; workers process serially)
- `RABBITMQ_RUN_LOG` (custom log path)

Reminder: gpt-4o has a 30K TPM ceiling, so keep `MAX_CONCURRENT_REQUESTS` low during smoke tests and prefer multiple slow workers over one aggressive one.

### Why It Matters

Financial analysts, retrieval engineers, and Hebbia-style matrix builders care about:

- **Exact numeric fidelity**: mixing up reported vs guidance numbers is unacceptable.
- **Distractor robustness**: transcripts place multiple numerics in close proximity; reasoning agents can help—or hallucinate.
- **Production signals**: Bootstrapping + permutation tests quantify whether reasoning prompts materially outperform terse baselines.

Use this framework to audition new prompts, swap models, or extend to other tickers without rewriting the orchestration logic.

## 📝 Example: Financial Q&A Evaluation

```python
# In config.py

def get_questions():
    return [
        Question(
            id="revenue_drivers",
            text="What were the main revenue drivers this quarter?",
            criterion="factual_accuracy",
        ),
        Question(
            id="guidance",
            text="What is management's revenue guidance for next quarter?",
            criterion="verbatim_precision",
        ),
    ]

def get_experiment_config():
    return ExperimentConfig(
        # ... other config ...
        document_context=_load_document("data/nvidia_q2_2025_call.txt"),
    )
```

## 🐛 Troubleshooting

### "No logprobs returned"

- Ensure you're using a model that supports logprobs (GPT-4o, GPT-4 Turbo)
- Check API key permissions

### Rate limit errors

- Reduce `max_concurrent_requests` in config
- Add retry logic (future enhancement)

### High costs

- Reduce `n_bootstrap_runs` for testing
- Use fewer temperatures
- Test with smaller models first

## 📚 References

- **Paper**: ["Who Evaluates the Evaluator: Reaching Autonomous Consensus on Agentic Outputs"](https://arxiv.org/abs/2406.16602)
- **Hebbia**: [Company website](https://hebbia.ai)
- **OpenAI API**: [Documentation](https://platform.openai.com/docs)

## 🤝 Contributing

To extend this framework:

1. Add new evaluation criteria in `get_questions()`
2. Implement custom agents in `get_agents()`
3. Modify statistical tests in `consensus_core.py`
4. Add visualization tools (future enhancement)

---

**Questions?** Open an issue or check the main [README.md](../README.md) for project context.
