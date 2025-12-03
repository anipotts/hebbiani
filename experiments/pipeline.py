"""
Shared pipeline utilities for agent generation + evaluator scoring.

These helpers centralize retry/backoff behavior and ensure both the queue
workers and the legacy experiment runners use identical prompting logic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict, Tuple

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from consensus_core import (
    AgentConfig,
    Document,
    EvalRecord,
    ParsedAgentOutput,
    Question,
    build_agent_prompt,
    build_eval_prompt,
    extract_score_from_logprobs,
    parse_agent_json_output,
)
from ground_truth import compare_to_ground_truth
from config import ExperimentConfig  # To access calibration threshold if needed, or pass it in.


RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


async def _async_retry(call: Callable[[], asyncio.Future], max_attempts: int = 5, base_delay: float = 2.0):
    for attempt in range(max_attempts):
        try:
            return await call()
        except RETRYABLE_ERRORS as err:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2**attempt))


def _sync_retry(call: Callable[[], any], max_attempts: int = 5, base_delay: float = 2.0):
    for attempt in range(max_attempts):
        try:
            return call()
        except RETRYABLE_ERRORS as err:
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))


async def run_agent_async(
    client: AsyncOpenAI,
    agent: AgentConfig,
    question: Question,
    document: Document,
    temperature: float,
) -> str:
    prompt = build_agent_prompt(agent, question, document)

    async def _invoke():
        return await client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )

    resp = await _async_retry(_invoke)
    return resp.choices[0].message.content


def run_agent_sync(
    client: OpenAI,
    agent: AgentConfig,
    question: Question,
    document: Document,
    temperature: float,
) -> str:
    prompt = build_agent_prompt(agent, question, document)

    def _invoke():
        return client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )

    resp = _sync_retry(_invoke)
    return resp.choices[0].message.content


async def evaluate_answer_async(
    client: AsyncOpenAI,
    eval_model: str,
    question: Question,
    final_answer: str,
) -> Tuple[float, Dict[str, float]]:
    prompt = build_eval_prompt(question, final_answer)

    async def _invoke():
        return await client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

    resp = await _async_retry(_invoke)
    msg = resp.choices[0]
    if not msg.logprobs or not msg.logprobs.content:
        raise ValueError("No logprobs returned from evaluator model.")
    logprob_dict = {item.token: item.logprob for item in msg.logprobs.content[0].top_logprobs}
    return extract_score_from_logprobs(logprob_dict)


def evaluate_answer_sync(
    client: OpenAI,
    eval_model: str,
    question: Question,
    final_answer: str,
) -> Tuple[float, Dict[str, float]]:
    prompt = build_eval_prompt(question, final_answer)

    def _invoke():
        return client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

    resp = _sync_retry(_invoke)
    msg = resp.choices[0]
    if not msg.logprobs or not msg.logprobs.content:
        raise ValueError("No logprobs returned from evaluator model.")
    logprob_dict = {item.token: item.logprob for item in msg.logprobs.content[0].top_logprobs}
    return extract_score_from_logprobs(logprob_dict)


def build_eval_record(
    run_id: int,
    question: Question,
    document: Document,
    agent: AgentConfig,
    temperature: float,
    raw_response: str,
    final_answer: str,
    expected_score: float,
    score_distribution: Dict[str, float],
    parsed_output: ParsedAgentOutput = None,
) -> EvalRecord:
    exact_match, normalized_match = compare_to_ground_truth(question.id, final_answer)
    
    # Phase 2: Calibration
    from config import get_experiment_config # lazy import to avoid circularity if any
    conf_threshold = get_experiment_config().calibration_confidence_threshold
    
    confidence_score = parsed_output.confidence_score if parsed_output else None
    raw_reasoning = parsed_output.reasoning if parsed_output else None
    
    calibrated_correct = None
    if confidence_score is not None:
        calibrated_correct = (
            confidence_score >= conf_threshold
            and bool(normalized_match)
        )

    return EvalRecord(
        run_id=run_id,
        doc_id=document.doc_id,
        question_id=question.id,
        question_text=question.text,
        criterion=question.criterion,
        agent_name=agent.name,
        agent_model=agent.model,
        temperature=temperature,
        raw_response=raw_response,
        final_answer=final_answer,
        expected_score=expected_score,
        score_distribution=score_distribution,
        exact_match=exact_match,
        normalized_match=normalized_match,
        confidence_score=confidence_score,
        calibrated_correct=calibrated_correct,
        raw_reasoning=raw_reasoning,
    )


def process_single_turn_sync(
    client: OpenAI,
    agent: AgentConfig,
    question: Question,
    document: Document,
    temperature: float,
    eval_model: str,
    run_id: int,
) -> EvalRecord:
    """
    Process a single experiment turn synchronously.
    Runs agent -> parses -> runs evaluator -> builds record.
    """
    raw_response = run_agent_sync(client, agent, question, document, temperature)
    parsed = parse_agent_json_output(agent.name, raw_response)
    final_answer = parsed.final_answer_for_eval
    
    expected_score, dist = evaluate_answer_sync(client, eval_model, question, final_answer)
    
    return build_eval_record(
        run_id=run_id,
        question=question,
        document=document,
        agent=agent,
        temperature=temperature,
        raw_response=raw_response,
        final_answer=final_answer,
        expected_score=expected_score,
        score_distribution=dist,
        parsed_output=parsed
    )


async def process_single_turn_async(
    client: AsyncOpenAI,
    agent: AgentConfig,
    question: Question,
    document: Document,
    temperature: float,
    eval_model: str,
    run_id: int,
    semaphore: asyncio.Semaphore = None,
) -> EvalRecord:
    """
    Process a single experiment turn asynchronously with optional semaphore.
    Runs agent -> parses -> runs evaluator -> builds record.
    """
    if semaphore:
        async with semaphore:
            raw_response = await run_agent_async(client, agent, question, document, temperature)
    else:
        raw_response = await run_agent_async(client, agent, question, document, temperature)
        
    parsed = parse_agent_json_output(agent.name, raw_response)
    final_answer = parsed.final_answer_for_eval
    
    if semaphore:
        async with semaphore:
            expected_score, dist = await evaluate_answer_async(client, eval_model, question, final_answer)
    else:
        expected_score, dist = await evaluate_answer_async(client, eval_model, question, final_answer)
        
    return build_eval_record(
        run_id=run_id,
        question=question,
        document=document,
        agent=agent,
        temperature=temperature,
        raw_response=raw_response,
        final_answer=final_answer,
        expected_score=expected_score,
        score_distribution=dist,
        parsed_output=parsed
    )
