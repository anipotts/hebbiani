"""
Core Consensus Evaluation Framework

Shared data structures, normalization utilities, and statistical helpers that
power both the synchronous and asynchronous experiment runners.
"""

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Score tokens for Likert scale
LIKERT_TOKENS = ["1", "2", "3", "4", "5"]


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Document:
    """Normalized earnings call transcript."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class Question:
    """An evaluation question tied to a specific document."""

    id: str
    text: str
    criterion: str  # e.g., "verbatim_precision"
    doc_id: str


@dataclass
class AgentConfig:
    """Configuration for an agent being evaluated."""

    name: str
    model: str
    prompt: str


@dataclass
class EvalRecord:
    """A single evaluation record from the experiment."""

    run_id: int
    doc_id: str
    question_id: str
    question_text: str
    criterion: str
    agent_name: str
    agent_model: str
    temperature: float
    raw_response: str
    final_answer: str
    expected_score: float
    score_distribution: Dict[str, float]  # token → probability
    exact_match: Optional[bool] = None
    normalized_match: Optional[bool] = None
    # Phase 2: Calibration fields
    confidence_score: Optional[float] = None
    calibrated_correct: Optional[bool] = None
    raw_reasoning: Optional[str] = None


@dataclass
class ParsedAgentOutput:
    """Structure for phase 2 agent outputs."""

    reasoning: Optional[str]
    answer_value: str
    confidence_score: Optional[float]
    final_answer_for_eval: str


# ============================================================================
# NORMALIZATION UTILITIES
# ============================================================================


def normalize_transcript_text(raw_text: str) -> str:
    """Normalize transcript whitespace while preserving speaker names."""

    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        lower = stripped.lower()
        if any(
            token in lower
            for token in (
                "copyright",
                "all rights reserved",
                "privacy policy",
                "terms of use",
            )
        ):
            # Strip site chrome / boilerplate if it appears
            continue

        cleaned_lines.append(stripped)

    collapsed = "\n".join(cleaned_lines)
    collapsed = re.sub(r"[ \t]{2,}", " ", collapsed)
    return collapsed.strip()


def parse_final_answer(agent_name: str, raw_output: str) -> str:
    """
    Extract the final answer from an agent response.

    Candidate agents emit reasoning followed by 'FINAL: ...'.
    Baseline agents just emit the answer.
    """

    if not raw_output:
        return ""

    final_lines = [
        line
        for line in raw_output.splitlines()
        if line.strip().upper().startswith("FINAL:")
    ]
    if final_lines:
        return final_lines[-1].split(":", 1)[-1].strip()

    return raw_output.strip()


def parse_agent_json_output(agent_name: str, raw_output: str) -> ParsedAgentOutput:
    """
    Parse structured JSON output from agents (Phase 2).
    Falls back to parse_final_answer if JSON is invalid.
    """
    clean_output = raw_output.strip()
    # Try to strip markdown code blocks if present
    if clean_output.startswith("```json"):
        clean_output = clean_output[7:]
    if clean_output.startswith("```"):
        clean_output = clean_output[3:]
    if clean_output.endswith("```"):
        clean_output = clean_output[:-3]
    clean_output = clean_output.strip()

    try:
        data = json.loads(clean_output)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")

        answer_value = str(data.get("answer_value", "")).strip()
        confidence_score = data.get("confidence_score")
        reasoning = data.get("reasoning")

        # Validate confidence is float or None
        if confidence_score is not None:
            try:
                confidence_score = float(confidence_score)
                if not (0.0 <= confidence_score <= 1.0):
                    # Warn or clamp? Requirements say "validated", let's invalidate if OOB
                    # treat as None if out of bounds? Or just clamp. Let's treat as None.
                    confidence_score = None
            except (ValueError, TypeError):
                confidence_score = None

        # If answer value is empty, fall back to previous logic?
        # The prompt requires "answer_value". If empty, maybe the model failed.
        # Let's rely on what we got.

        return ParsedAgentOutput(
            reasoning=str(reasoning) if reasoning else None,
            answer_value=answer_value,
            confidence_score=confidence_score,
            final_answer_for_eval=answer_value,
        )

    except (json.JSONDecodeError, ValueError):
        # Fallback to legacy parsing
        legacy_answer = parse_final_answer(agent_name, raw_output)
        return ParsedAgentOutput(
            reasoning=None,
            answer_value=legacy_answer,
            confidence_score=None,
            final_answer_for_eval=legacy_answer,
        )


# ============================================================================
# PROMPT BUILDERS
# ============================================================================


def _document_header(document: Document) -> str:
    """Build a short metadata header for the transcript."""

    meta = document.metadata
    company = meta.get("symbol", "Unknown")
    quarter = meta.get("fiscal_quarter", "?")
    fiscal_year = meta.get("fiscal_year", "?")
    call_date = meta.get("call_date", "Unknown date")

    return (
        f"You are analyzing the following earnings call transcript.\n"
        f"Company: {company}\n"
        f"Quarter: {quarter}\n"
        f"Fiscal year: {fiscal_year}\n"
        f"Call date: {call_date}\n"
        f"Document ID: {document.doc_id}\n"
    )


def build_agent_prompt(
    agent: AgentConfig, question: Question, document: Document
) -> str:
    """
    Build the prompt for an agent to answer a question with full transcript context.
    """

    header = _document_header(document)
    return (
        f"{agent.prompt}\n\n"
        f"{header}\n"
        f"Task:\n{question.text}\n\n"
        f'Here is the full transcript:\n"""{document.text}"""'
    )


def build_eval_prompt(question: Question, agent_answer: str) -> str:
    """
    Build the evaluation prompt for scoring a response under verbatim precision.
    """

    criterion_description = (
        "5 = Exact numeric match (correct metric, quarter, units, and any range bounds).\n"
        "3 = Partially correct (minor formatting/rounding issues or missing units).\n"
        "1 = Incorrect metric/quarter, hallucinated values, or 'UNKNOWN' when guidance exists."
    )

    return f"""
You are rating an answer for the criterion: {question.criterion}.
Determine how faithfully it captures the requested numeric value from the transcript.

Scoring guide:
{criterion_description}

Question:
{question.text}

Answer under review:
\"\"\"{agent_answer}\"\"\"

Respond with ONLY a single integer score between 1 and 5.
"""


# ============================================================================
# EVALUATION LOGIC
# ============================================================================


def extract_score_from_logprobs(logprob_dict: Dict) -> Tuple[float, Dict[str, float]]:
    """
    Extract expected score from logprobs dictionary.
    """

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


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def permutation_test(
    scores_A: List[float], scores_B: List[float], iterations: int = 10_000
) -> float:
    """
    Two-sided permutation test.
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


def compute_significance(
    records: List[EvalRecord], agent_A: str, agent_B: str, perm_iterations: int = 10_000
) -> List[Dict[str, Any]]:
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

            p = permutation_test(scores_A, scores_B, iterations=perm_iterations)
            results.append({"question": qid, "criterion": crit, "p_value": p})
            print(f"[Q={qid}] [Criterion={crit}]  p={p:.4f}")

    return results


def build_leaderboard(records: List[EvalRecord]) -> List[Dict[str, Any]]:
    """
    Aggregate expected scores per agent and build a leaderboard.
    """

    by_agent: Dict[str, List[float]] = {}
    for r in records:
        by_agent.setdefault(r.agent_name, []).append(r.expected_score)

    leaderboard: List[Dict[str, Any]] = []
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


# ============================================================================
# FILE I/O
# ============================================================================


def save_results(
    records: List[EvalRecord],
    sig_results: List[Dict[str, Any]],
    output_dir: str = "experiment_logs",
    prefix: str = "",
):
    """
    Save experiment results to files.
    """

    os.makedirs(output_dir, exist_ok=True)

    records_file = os.path.join(output_dir, f"{prefix}run_records.jsonl")
    with open(records_file, "w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")

    sig_file = os.path.join(output_dir, f"{prefix}significance.json")
    with open(sig_file, "w") as f:
        json.dump(sig_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - Records: {records_file}")
    print(f"  - Significance: {sig_file}")
