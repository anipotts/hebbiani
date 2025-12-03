from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TaskPayload(BaseModel):
    run_id: int
    doc_id: str
    question_id: str
    agent_name: str
    agent_model: str
    temperature: float
    question_text: Optional[str] = None
    criterion: Optional[str] = None
    context_variant: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    retry_count: int = 0


class ResultPayload(BaseModel):
    run_id: int
    doc_id: str
    question_id: str
    agent_name: str
    agent_model: str
    temperature: float
    final_answer: str
    confidence_score: Optional[float] = None
    calibrated_correct: Optional[bool] = None
    raw_reasoning: Optional[str] = None
    expected_score: float
    exact_match: Optional[bool] = None
    normalized_match: Optional[bool] = None
    error: Optional[str] = None
