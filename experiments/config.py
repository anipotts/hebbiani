"""
Experiment configuration for the needle-in-the-transcript evaluation.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from consensus_core import AgentConfig, Document, Question
from document_loader import load_transcripts


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""

    experiment_name: str = "nvda_guidance_needle_in_transcript"
    api_model_agent: str = "gpt-4o"
    api_model_eval: str = "gpt-4o"
    n_bootstrap_runs: int = 50
    temperatures: List[float] = None
    perm_test_iterations: int = 10_000
    max_concurrent_requests: int = 8

    # Phase 2: Calibration threshold
    calibration_confidence_threshold: float = 0.7

    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.1, 0.4, 0.8]


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANSCRIPT_PATHS = [
    DATA_DIR / "NVDA_Q2_FY2026_earnings_call_transcript_2025_08_27.json",
    DATA_DIR / "NVDA_Q3_FY2026_earnings_call_transcript_2025_11_19.json",
]

QUESTION_TEMPLATES = [
    (
        "q1_revenue_guidance",
        "Based on the management commentary and guidance section of this earnings call, "
        "extract the exact revenue guidance figure for the next quarter (include currency and any ranges).",
    ),
    (
        "q2_eps_guidance",
        "Based on the management commentary and guidance section of this earnings call, "
        "extract the exact EPS guidance range for the next quarter, including units and low/high values.",
    ),
    (
        "q3_data_center_revenue_reported",
        "From this quarter's results discussed in the call, extract the exact data center segment revenue reported "
        "for this quarter, including currency and units (e.g., 'USD 51 billion').",
    ),
]


def _parse_filter(env_var: str) -> List[str]:
    raw = os.environ.get(env_var, "")
    items = [token.strip() for token in raw.split(",") if token.strip()]
    return items


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_list_env(name: str, default: List[float]) -> List[float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        return [float(val) for val in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        parts = [token.strip() for token in raw.split(",") if token.strip()]
        if not parts:
            return default
        return [float(token) for token in parts]


DOC_ID_FILTER = set(_parse_filter("EXPERIMENT_DOC_IDS"))
QUESTION_SUFFIX_FILTER = set(_parse_filter("EXPERIMENT_QUESTION_SUFFIXES"))

OUTPUT_SCHEMA_INSTRUCTION = """
You must output a valid JSON object with exactly these fields:
{
  "reasoning": "<string> (your chain-of-thought, or empty string if none)",
  "answer_value": "<string> (the extracted answer)",
  "confidence_score": <float> (your confidence between 0.0 and 1.0)
}
Output ONLY the JSON object. Do not add any other text, markdown formatting, or explanation outside the JSON.
"""

def get_experiment_config() -> ExperimentConfig:
    """Return the core experiment configuration."""

    default_temps = [0.1, 0.4, 0.8]
    return ExperimentConfig(
        experiment_name="nvda_guidance_needle_in_transcript",
        api_model_agent="gpt-4o",
        api_model_eval="gpt-4o",
        n_bootstrap_runs=_int_env("N_BOOTSTRAPS", 50),
        temperatures=_float_list_env("TEMPERATURES_JSON", default_temps),
        perm_test_iterations=10_000,
        max_concurrent_requests=_int_env("MAX_CONCURRENT_REQUESTS", 8),
    )


# ============================================================================
# DOCUMENTS, AGENTS, AND QUESTIONS
# ============================================================================


def get_documents() -> List[Document]:
    """Load and normalize NVDA transcripts."""

    documents = load_transcripts(TRANSCRIPT_PATHS)
    if DOC_ID_FILTER:
        documents = [doc for doc in documents if doc.doc_id in DOC_ID_FILTER]
    return documents


def get_agents() -> Tuple[AgentConfig, AgentConfig]:
    """Define the paired agents for the experiment."""

    baseline_prompt = (
        "You answer earnings-call extraction questions as concisely as possible.\n\n"
        "Rules:\n"
        "* Output the requested value(s) in 'answer_value'.\n"
        "* Set 'reasoning' to an empty string \"\".\n"
        "* Set 'confidence_score' to your subjective confidence (0.0-1.0).\n"
        "* If a currency is present, include it (e.g., 'USD 65 billion').\n"
        "* If a range is requested, output 'low – high' using the units management used.\n"
        "* If you are not sure, output 'UNKNOWN' in 'answer_value'.\n\n"
        f"{OUTPUT_SCHEMA_INSTRUCTION}"
    )

    reasoning_prompt = (
        "You are an earnings-call extraction expert.\n"
        "Your job is to read noisy, distractor-heavy transcripts and extract EXACT numeric guidance or reported figures.\n\n"
        "You must:\n"
        "* Explain your reasoning in 1–3 short sentences in the 'reasoning' field.\n"
        "* Quote or paraphrase specific evidence.\n"
        "* Output the final extracted value in 'answer_value'.\n"
        "* Set 'confidence_score' to your subjective confidence (0.0-1.0).\n\n"
        "If you are not sure, explain why in 'reasoning' and put 'UNKNOWN' in 'answer_value'.\n\n"
        f"{OUTPUT_SCHEMA_INSTRUCTION}"
    )

    baseline = AgentConfig(name="baseline_H0", model="gpt-4o", prompt=baseline_prompt)
    candidate = AgentConfig(
        name="candidate_HA", model="gpt-4o", prompt=reasoning_prompt
    )
    return baseline, candidate


def get_questions(documents: List[Document]) -> List[Question]:
    """Generate question objects for every transcript."""

    questions: List[Question] = []
    for doc in documents:
        for suffix, template in QUESTION_TEMPLATES:
            if QUESTION_SUFFIX_FILTER and suffix not in QUESTION_SUFFIX_FILTER:
                continue
            qid = f"{doc.doc_id}_{suffix}"
            questions.append(
                Question(
                    id=qid,
                    text=template,
                    criterion="verbatim_precision",
                    doc_id=doc.doc_id,
                )
            )
    return questions
