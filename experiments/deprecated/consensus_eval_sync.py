"""
consensus_eval.py

A fully faithful reproduction of Hebbia’s consensus-based evaluation framework,
as described in "Who Evaluates the Evaluator: Reaching Autonomous Consensus on Agentic Outputs".

This script runs:
- J temperature replications per question per agent
- N bootstrap replications of the *entire experiment*
- Independent evaluator LLM scoring each response on a Likert 1–5 scale
- Logprob capture of score tokens
- Conversion to linear probabilities → normalized distributions → expected scores
- Storage of all data for later statistical tests
- Permutation tests (10,000 iterations) for significance

You can use this as a base experiment for Hebbiani.
"""

import os
import json
import random
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import numpy as np
from tqdm import tqdm

# --------------------------------------
# USER CONFIG
# --------------------------------------

API_MODEL_AGENT = "gpt-4o"  # model used to produce answers (AgentHO or AgentHA) - changed from gpt-4.1
API_MODEL_EVAL = "gpt-4o"  # evaluator model (AgentEval) - changed from gpt-4.1
J = 3  # temperature replications per question
N = 50  # bootstrap replications
TEMPERATURES = [0.1, 0.4, 0.8]  # length J
PERM_TEST_ITER = 10_000

# Score tokens for Likert scale
LIKERT_TOKENS = ["1", "2", "3", "4", "5"]

# Initialize OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it by running: export OPENAI_API_KEY='your-api-key-here'"
    )
client = OpenAI(api_key=api_key)

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
    score_distribution: Dict[str, float]  # token → probability


# --------------------------------------
# UTILITY: LLM CALL
# --------------------------------------


def llm_call(model: str, prompt: str, temperature: float = 0.1) -> dict:
    """Run an OpenAI API call and return raw response with logprobs."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )
    except Exception as e:
        print(f"\n❌ API Error in llm_call: {e}")
        print(f"   Model: {model}, Temperature: {temperature}")
        raise


# --------------------------------------
# STEP 1: GENERATE AGENT OUTPUT
# --------------------------------------


def run_agent(agent: AgentConfig, question: Question, temperature: float) -> str:
    """Ask the agent a question and return its generated response text."""
    composed_prompt = f"{agent.prompt}\n\nQuestion:\n{question.text}"
    resp = llm_call(agent.model, composed_prompt, temperature)
    return resp.choices[0].message.content


# --------------------------------------
# STEP 2–5: EVALUATOR MODEL SCORING WITH LOGPROBS
# --------------------------------------


def evaluate_response(
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

    resp = llm_call(evaluator_model, eval_prompt, temperature=0.0)
    msg = resp.choices[0]

    # Extract top_logprobs from the first token of the output
    if not msg.logprobs or not msg.logprobs.content:
        raise ValueError(
            "No logprobs returned from API. Check if model supports logprobs."
        )

    logprob_dict = msg.logprobs.content[0].top_logprobs

    # Map "1"–"5" tokens → linear probabilities
    unnormalized = {}
    for lk in LIKERT_TOKENS:
        if lk in logprob_dict:
            unnormalized[lk] = np.exp(logprob_dict[lk])
        else:
            # If missing, assign tiny epsilon (model didn’t surface the token)
            unnormalized[lk] = 1e-12

    # Normalize
    total = sum(unnormalized.values())
    probs = {k: v / total for k, v in unnormalized.items()}

    # Expected score
    expected = sum(int(k) * probs[k] for k in LIKERT_TOKENS)

    return expected, probs


# --------------------------------------
# STEP 7: ENTIRE EXPERIMENT LOOP (BOOTSTRAP N TIMES)
# --------------------------------------


def run_experiment(
    agent_A: AgentConfig, agent_B: AgentConfig, questions: List[Question]
) -> List[EvalRecord]:
    """
    Produce N bootstrap samples; for each:
    - For each question:
        * Run J evaluations at multiple temperatures
        * Evaluate each with AgentEval
    - Store all EvalRecords
    """
    all_records = []

    # Calculate total iterations for progress tracking
    total_iterations = (
        N * len(questions) * 2 * len(TEMPERATURES)
    )  # runs × questions × agents × temps
    pbar = tqdm(total=total_iterations, desc="Overall Progress", unit="call")

    for run_id in range(N):
        print(f"\n=== Bootstrap Run {run_id+1}/{N} ===")
        for question in questions:
            for agent in [agent_A, agent_B]:
                for t in TEMPERATURES:
                    try:
                        pbar.set_description(
                            f"Run {run_id+1}/{N} | Q: {question.id} | Agent: {agent.name} | T: {t}"
                        )
                        agent_output = run_agent(agent, question, temperature=t)
                        pbar.update(1)

                        expected_score, dist = evaluate_response(
                            API_MODEL_EVAL, question, agent_output
                        )
                        pbar.update(1)

                        record = EvalRecord(
                            run_id=run_id,
                            question_id=question.id,
                            criterion=question.criterion,
                            agent_name=agent.name,
                            response_text=agent_output,
                            expected_score=expected_score,
                            score_distribution=dist,
                        )
                        all_records.append(record)
                    except Exception as e:
                        pbar.close()
                        print(
                            f"\n❌ Error processing: Run {run_id+1}, Question {question.id}, Agent {agent.name}, Temp {t}"
                        )
                        print(f"   Error: {e}")
                        raise

    pbar.close()
    return all_records


# --------------------------------------
# STEP 8 + Stats: PERMUTATION TEST
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
    df = {}

    # Group by (question, criterion, agent)
    for r in records:
        key = (r.question_id, r.criterion, r.agent_name)
        df.setdefault(key, []).append(r.expected_score)

    # Compare A vs B per question × criterion
    question_ids = set([r.question_id for r in records])

    for qid in question_ids:
        crits = set(r.criterion for r in records if r.question_id == qid)
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
# MAIN RUNNER
# --------------------------------------

if __name__ == "__main__":

    # --------------------------------------
    # Define two agents (AgentHO and AgentHA)
    # --------------------------------------
    agent_HO = AgentConfig(
        name="baseline_H0",
        model="gpt-4o",  # changed from gpt-4.1
        prompt="You are the baseline agent. Provide concise, factual answers.",
    )

    agent_HA = AgentConfig(
        name="candidate_HA",
        model="gpt-4o",  # changed from gpt-4.1
        prompt="You are the improved candidate agent. Provide highly detailed, analytic answers.",
    )

    # --------------------------------------
    # Sample questions (you can insert financial, legal, or synthetic)
    # --------------------------------------
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

    # RUN THE EXPERIMENT
    records = run_experiment(agent_HO, agent_HA, questions)

    # Save full JSONL
    os.makedirs("experiment_logs", exist_ok=True)
    with open("experiment_logs/run_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")

    # Compute significance
    sig_results = compute_significance(records, agent_HO.name, agent_HA.name)

    with open("experiment_logs/significance.json", "w") as f:
        json.dump(sig_results, f, indent=2)

    print("\nExperiment complete.")
