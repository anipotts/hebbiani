"""
Ground-truth hooks for NVDA guidance extraction experiments.

These mappings intentionally ship with placeholder values that can be filled in
manually by the researcher once the authoritative numbers have been verified.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


# Question → canonical answer string (fill in as authoritative data becomes available)
GROUND_TRUTH: Dict[str, Optional[str]] = {
    # Q2 FY2026 (next quarter is Q3 FY2026)
    "NVDA_Q2_FY2026_q1_revenue_guidance": "$54.0 billion",
    "NVDA_Q2_FY2026_q2_eps_guidance": None,  # no public guidance found
    "NVDA_Q2_FY2026_q3_data_center_revenue_reported": "$41.1 billion",
    # Q3 FY2026 (next quarter is Q4 FY2026)
    "NVDA_Q3_FY2026_q1_revenue_guidance": "$65.0 billion",
    "NVDA_Q3_FY2026_q2_eps_guidance": None,  # no public guidance found
    "NVDA_Q3_FY2026_q3_data_center_revenue_reported": "$51.2 billion",
}


def get_ground_truth_map() -> Dict[str, Optional[str]]:
    """Expose the mutable ground-truth mapping."""

    return GROUND_TRUTH


def _normalize_answer(answer: str) -> str:
    """Normalize numeric strings for approximate matching."""

    if answer is None:
        return ""

    normalized = answer.lower().strip()
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def compare_to_ground_truth(
    question_id: str, answer: str
) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Compare an answer to the ground truth if available.

    Returns:
        Tuple of (exact_match, normalized_match), each possibly None when the
        ground truth is not yet populated.
    """

    truth = GROUND_TRUTH.get(question_id)
    if not truth:
        return None, None

    exact = answer.strip() == truth.strip()
    normalized = _normalize_answer(answer) == _normalize_answer(truth)
    return exact, normalized
