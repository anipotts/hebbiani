"""
Consensus Evaluation - Async Version

Async + batched version of the consensus evaluation experiment.

Key advantages:
- Uses AsyncOpenAI + asyncio for concurrency
- Limits concurrent requests via semaphore (respects API rate limits)
- Much faster for large experiments (10x+ speedup)
- Includes leaderboard generation

Use this for:
- Large experiments (> 100 API calls)
- Production runs
- When speed matters

For small experiments or debugging, use consensus_eval_sync.py instead.
"""

import asyncio
import os
from typing import Dict, List

from openai import AsyncOpenAI
from tqdm import tqdm

from consensus_core import (
    AgentConfig,
    Document,
    EvalRecord,
    Question,
    build_leaderboard,
    compute_significance,
    parse_final_answer,
    save_results,
)
from config import get_documents, get_experiment_config, get_agents, get_questions
from pipeline import process_single_turn_async

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it by running: export OPENAI_API_KEY='your-api-key-here'"
    )
client = AsyncOpenAI(api_key=api_key)


async def process_one(
    run_id: int,
    question: Question,
    document: Document,
    agent: AgentConfig,
    temperature: float,
    config,
    semaphore: asyncio.Semaphore,
) -> EvalRecord:
    """Process a single (run, question, agent, temp) unit: answer + evaluation."""
    return await process_single_turn_async(
        client,
        agent,
        question,
        document,
        temperature,
        config.api_model_eval,
        run_id,
        semaphore,
    )


async def run_experiment_async(
    agent_A: AgentConfig,
    agent_B: AgentConfig,
    questions: List[Question],
    documents: List[Document],
    config,
) -> List[EvalRecord]:
    """
    Run the full experiment asynchronously with concurrency control.

    Creates async tasks for all (run, question, agent, temp) combos
    and executes them with controlled concurrency.
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    doc_map: Dict[str, Document] = {doc.doc_id: doc for doc in documents}

    tasks = []
    for run_id in range(config.n_bootstrap_runs):
        for question in questions:
            document = doc_map.get(question.doc_id)
            if document is None:
                raise ValueError(f"No document loaded for question {question.id}")

            for agent in [agent_A, agent_B]:
                for t in config.temperatures:
                    tasks.append(
                        process_one(
                            run_id,
                            question,
                            document,
                            agent,
                            t,
                            config,
                            semaphore,
                        )
                    )

    all_records: List[EvalRecord] = []
    # Process with progress bar
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Async progress",
        unit="unit",
    ):
        rec = await coro
        all_records.append(rec)

    return all_records


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Main execution function."""
    config = get_experiment_config()
    documents = get_documents()
    agent_A, agent_B = get_agents()
    questions = get_questions(documents)

    print("=" * 70)
    print("CONSENSUS EVALUATION EXPERIMENT (Async)")
    print("=" * 70)
    print(f"Models: Agent={config.api_model_agent}, Eval={config.api_model_eval}")
    print(f"Bootstrap runs: {config.n_bootstrap_runs}")
    print(f"Temperatures: {config.temperatures}")
    print(f"Transcripts: {len(documents)}")
    print(f"Questions: {len(questions)}")
    print(f"Agents: {agent_A.name} vs {agent_B.name}")
    print(f"Max concurrent requests: {config.max_concurrent_requests}")
    print("=" * 70)

    # Run experiment
    records = await run_experiment_async(agent_A, agent_B, questions, documents, config)

    # Compute significance
    sig_results = compute_significance(
        records, agent_A.name, agent_B.name, config.perm_test_iterations
    )

    # Build leaderboard
    leaderboard = build_leaderboard(records)

    # Save results
    save_results(records, sig_results, prefix="async_")

    os.makedirs("experiment_logs", exist_ok=True)
    with open("experiment_logs/async_leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

    print("\n✅ Async experiment complete.")


if __name__ == "__main__":
    asyncio.run(main())
