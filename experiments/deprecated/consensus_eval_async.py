"""
consensus_eval_async.py

Async + batched version of the consensus evaluation experiment.

Key differences from consensus_eval.py:
- Uses AsyncOpenAI + asyncio for concurrency
- Limits concurrent requests via a semaphore (to avoid hammering the API)
- Default N is smaller (10) for faster prototyping
- Builds a simple leaderboard of agents by mean expected score
"""

import os
import json
import random
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI

# --------------------------------------
# USER CONFIG
# --------------------------------------

API_MODEL_AGENT = "gpt-4o"  # model used to produce answers
API_MODEL_EVAL = "gpt-4o"  # evaluator model
LIKERT_TOKENS = ["1", "2", "3", "4", "5"]

# Experiment size
J = 3  # number of temperatures
TEMPERATURES = [0.1, 0.4, 0.8]  # length J
N = 10  # bootstrap replications (keep small for speed)
PERM_TEST_ITER = 5_000  # lower for speed while prototyping

# Concurrency
MAX_CONCURRENT_REQUESTS = 8  # tune based on your rate limits + comfort

# --------------------------------------
# CLIENT INIT
# --------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it by running: export OPENAI_API_KEY='your-api-key-here'"
    )

client = AsyncOpenAI(api_key=api_key)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# --------------------------------------
# DATA STRUCTURES
# --------------------------------------


@dataclass
class Question:
    id: str
    text: str
    criterion: str  # e.g., "logical_coherence", "insightfulness"


@dataclass
class AgentConfig:
    name: str
    model: str
    prompt: str


@dataclass
class EvalRecord:
    run_id: int
    question_id: str
    criterion: str
    agent_name: str
    response_text: str
    expected_score: float
    score_distribution: Dict[str, float]


# --------------------------------------
# ASYNC LLM CALL
# --------------------------------------


async def llm_call_async(model: str, prompt: str, temperature: float = 0.1):
    """Run an OpenAI API call with logprobs under concurrency control."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                logprobs=True,
                top_logprobs=5,
            )
            return resp
        except Exception as e:
            print(f"\n❌ API Error in llm_call_async: {e}")
            print(f"   Model: {model}, Temperature: {temperature}")
            raise


# --------------------------------------
# STEP 1: GENERATE AGENT OUTPUT (WITH CONTEXT SLOT)
# --------------------------------------


async def run_agent_async(
    agent: AgentConfig, question: Question, temperature: float, context: str
) -> str:
    """
    Ask the agent a question grounded in a document context.
    Change `context` to include real NVIDIA reports or other docs.
    """
    composed_prompt = f"""
{agent.prompt}

You are answering questions based ONLY on the following document.
The document is an earnings-related filing/transcript for a public company.
Do NOT invent numbers or facts that are not supported by the document.
If information is missing, say so explicitly.

DOCUMENT:
<<<
{context}
>>>

QUESTION (criterion = {question.criterion}):
{question.text}
"""
    resp = await llm_call_async(agent.model, composed_prompt, temperature)
    return resp.choices[0].message.content


# --------------------------------------
# STEP 2–5: EVALUATOR MODEL SCORING WITH LOGPROBS (ASYNC)
# --------------------------------------


async def evaluate_response_async(
    evaluator_model: str, question: Question, agent_output: str
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluator model assigns a Likert score 1–5.

    We extract token logprobs, convert to linear probs, normalize,
    compute expected score.
    """
    eval_prompt = f"""
You are scoring an LLM response on a single criterion: {question.criterion}.

Give ONLY a single integer 1, 2, 3, 4, or 5.
1 = very poor
5 = excellent

Response to evaluate:
\"\"\"{agent_output}\"\"\"
"""

    resp = await llm_call_async(evaluator_model, eval_prompt, temperature=0.0)
    msg = resp.choices[0]

    if not msg.logprobs or not msg.logprobs.content:
        raise ValueError(
            "No logprobs returned from API. Check if model supports logprobs."
        )

    logprob_dict = msg.logprobs.content[0].top_logprobs

    # Map "1"–"5" tokens → linear probabilities
    unnormalized: Dict[str, float] = {}
    for lk in LIKERT_TOKENS:
        if lk in logprob_dict:
            unnormalized[lk] = np.exp(logprob_dict[lk])
        else:
            unnormalized[lk] = 1e-12

    total = sum(unnormalized.values())
    probs = {k: v / total for k, v in unnormalized.items()}

    expected = sum(int(k) * probs[k] for k in LIKERT_TOKENS)
    return expected, probs


# --------------------------------------
# ONE UNIT OF WORK: (run, question, agent, temp)
# --------------------------------------


async def process_one(
    run_id: int,
    question: Question,
    agent: AgentConfig,
    temperature: float,
    context: str,
) -> EvalRecord:
    """Process a single (run, question, agent, temp) unit: answer + evaluation."""
    agent_output = await run_agent_async(agent, question, temperature, context)
    expected_score, dist = await evaluate_response_async(
        API_MODEL_EVAL, question, agent_output
    )

    return EvalRecord(
        run_id=run_id,
        question_id=question.id,
        criterion=question.criterion,
        agent_name=agent.name,
        response_text=agent_output,
        expected_score=expected_score,
        score_distribution=dist,
    )


# --------------------------------------
# STEP 7: ENTIRE EXPERIMENT LOOP (BOOTSTRAP N TIMES, ASYNC)
# --------------------------------------


async def run_experiment_async(
    agent_A: AgentConfig,
    agent_B: AgentConfig,
    questions: List[Question],
    context: str,
) -> List[EvalRecord]:
    """
    Create async tasks for all (run, question, agent, temp) combos
    and execute them with concurrency.
    """
    tasks = []
    for run_id in range(N):
        for question in questions:
            for agent in [agent_A, agent_B]:
                for t in TEMPERATURES:
                    tasks.append(process_one(run_id, question, agent, t, context))

    all_records: List[EvalRecord] = []
    # tqdm over async tasks
    for coro_chunk in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Overall async progress",
        unit="unit",
    ):
        rec = await coro_chunk
        all_records.append(rec)

    return all_records


# --------------------------------------
# STEP 8: PERMUTATION TEST (SAME AS BEFORE)
# --------------------------------------


def permutation_test(
    scores_A: List[float], scores_B: List[float], iterations: int = PERM_TEST_ITER
) -> float:
    """
    Two-sided permutation test.
    H0: mean(A) == mean(B)
    Returns empirical p-value.
    """
    observed = abs(np.mean(scores_A) - np.mean(scores_B))
    combined = scores_A + scores_B
    nA = len(scores_A)

    count = 0
    for _ in range(iterations):
        random.shuffle(combined)
        permA = combined[:nA]
        permB = combined[nA:]
        perm_diff = abs(np.mean(permA) - np.mean(permB))
        if perm_diff >= observed:
            count += 1

    return count / iterations


def compute_significance(records: List[EvalRecord], agent_A: str, agent_B: str):
    """
    Compute significance per question × criterion.
    """
    print("\n=== SIGNIFICANCE TESTS ===")
    results = []
    df: Dict[Tuple[str, str, str], List[float]] = {}

    for r in records:
        key = (r.question_id, r.criterion, r.agent_name)
        df.setdefault(key, []).append(r.expected_score)

    question_ids = set(r.question_id for r in records)

    for qid in question_ids:
        crits = {r.criterion for r in records if r.question_id == qid}
        for crit in crits:
            scores_A = df.get((qid, crit, agent_A), [])
            scores_B = df.get((qid, crit, agent_B), [])
            if len(scores_A) == 0 or len(scores_B) == 0:
                continue

            p = permutation_test(scores_A, scores_B)
            results.append({"question": qid, "criterion": crit, "p_value": p})
            print(f"[Q={qid}] [Criterion={crit}]  p={p:.4f}")

    return results


# --------------------------------------
# LEADERBOARD BUILDER
# --------------------------------------


def build_leaderboard(records: List[EvalRecord]):
    """
    Aggregate expected scores per agent and print a leaderboard.
    """
    by_agent: Dict[str, List[float]] = {}
    for r in records:
        by_agent.setdefault(r.agent_name, []).append(r.expected_score)

    leaderboard = []
    for agent_name, scores in by_agent.items():
        arr = np.array(scores, dtype=float)
        leaderboard.append(
            {
                "agent": agent_name,
                "mean_expected_score": float(arr.mean()),
                "std_expected_score": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n_samples": int(len(arr)),
            }
        )

    leaderboard.sort(key=lambda x: x["mean_expected_score"], reverse=True)

    print("\n=== MODEL LEADERBOARD (by mean expected score) ===")
    for row in leaderboard:
        print(
            f"{row['agent']:>15} | mean={row['mean_expected_score']:.3f} "
            f"| std={row['std_expected_score']:.3f} | n={row['n_samples']}"
        )

    return leaderboard


# --------------------------------------
# MAIN
# --------------------------------------


async def main():
    # Define two agents (AgentH0 and AgentHA)
    agent_HO = AgentConfig(
        name="baseline_H0",
        model=API_MODEL_AGENT,
        prompt="You are the baseline agent. Provide concise, factual answers.",
    )

    agent_HA = AgentConfig(
        name="candidate_HA",
        model=API_MODEL_AGENT,
        prompt="You are the improved candidate agent. Provide highly detailed, analytic answers.",
    )

    # Sample questions
    questions = [
        Question(
            id="q1",
            text="Summarize this quarter's revenue drivers.",
            criterion="insightfulness",
        ),
        Question(
            id="q2",
            text="What was management's view on macroeconomic trends?",
            criterion="logical_coherence",
        ),
        Question(
            id="q3",
            text="Extract the exact revenue guidance figure.",
            criterion="verbatim_precision",
        ),
    ]

    # TODO: load real NVIDIA context from a local file.
    # For now, use a placeholder string, then replace with:
    # with open('data/nvidia_q2_2025_call.txt', 'r') as f:
    #     doc_text = f.read()
    doc_text = (
        "PLACEHOLDER: paste or load your NVIDIA earnings transcript or 10-Q text here."
    )

    records = await run_experiment_async(agent_HO, agent_HA, questions, doc_text)

    os.makedirs("experiment_logs", exist_ok=True)
    with open("experiment_logs/run_records_async.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")

    sig_results = compute_significance(records, agent_HO.name, agent_HA.name)
    with open("experiment_logs/significance_async.json", "w") as f:
        json.dump(sig_results, f, indent=2)

    leaderboard = build_leaderboard(records)
    with open("experiment_logs/leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

    print("\nAsync experiment complete.")


if __name__ == "__main__":
    asyncio.run(main())
