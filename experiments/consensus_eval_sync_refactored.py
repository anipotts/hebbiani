"""
Consensus Evaluation - Synchronous Version

A faithful reproduction of Hebbia's consensus-based evaluation framework,
as described in "Who Evaluates the Evaluator: Reaching Autonomous Consensus on Agentic Outputs".

This synchronous version processes API calls sequentially. Use this for:
- Small experiments (< 100 API calls)
- When you need deterministic execution order
- Debugging individual API calls

For larger experiments, use consensus_eval_async.py instead.
"""

import os
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

from consensus_core import (
    AgentConfig,
    Document,
    EvalRecord,
    Question,
    compute_significance,
    parse_final_answer,
    save_results,
)
from config import get_documents, get_experiment_config, get_agents, get_questions
from pipeline import process_single_turn_sync

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it by running: export OPENAI_API_KEY='your-api-key-here'"
    )
client = OpenAI(api_key=api_key)


def run_experiment(
    agent_A: AgentConfig,
    agent_B: AgentConfig,
    questions: List[Question],
    documents: List[Document],
    config,
) -> List[EvalRecord]:
    """
    Run the full experiment synchronously.

    Produces N bootstrap samples; for each:
    - For each question:
        * Run evaluations at multiple temperatures
        * Evaluate each with AgentEval
    - Store all EvalRecords
    """
    all_records: List[EvalRecord] = []
    doc_map: Dict[str, Document] = {doc.doc_id: doc for doc in documents}

    # Calculate total iterations for progress tracking
    total_iterations = (
        config.n_bootstrap_runs * len(questions) * 2 * len(config.temperatures)
    )
    pbar = tqdm(total=total_iterations, desc="Overall Progress", unit="call")

    for run_id in range(config.n_bootstrap_runs):
        print(f"\n=== Bootstrap Run {run_id+1}/{config.n_bootstrap_runs} ===")
        for question in questions:
            document = doc_map.get(question.doc_id)
            if document is None:
                raise ValueError(f"No document loaded for question {question.id}")

            for agent in [agent_A, agent_B]:
                for t in config.temperatures:
                    try:
                        pbar.set_description(
                            f"Run {run_id+1}/{config.n_bootstrap_runs} | "
                            f"{question.id} | Agent: {agent.name} | T: {t}"
                        )
                        record = process_single_turn_sync(
                            client,
                            agent,
                            question,
                            document,
                            t,
                            config.api_model_eval,
                            run_id,
                        )
                        pbar.update(2)  # Both agent and eval calls done inside
                        all_records.append(record)
                    except Exception as e:
                        pbar.close()
                        print(
                            f"\n❌ Error processing: Run {run_id+1}, "
                            f"Question {question.id}, Agent {agent.name}, Temp {t}"
                        )
                        print(f"   Error: {e}")
                        raise

    pbar.close()
    return all_records


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function."""
    config = get_experiment_config()
    documents = get_documents()
    agent_A, agent_B = get_agents()
    questions = get_questions(documents)

    print("=" * 70)
    print("CONSENSUS EVALUATION EXPERIMENT (Synchronous)")
    print("=" * 70)
    print(f"Models: Agent={config.api_model_agent}, Eval={config.api_model_eval}")
    print(f"Bootstrap runs: {config.n_bootstrap_runs}")
    print(f"Temperatures: {config.temperatures}")
    print(f"Transcripts: {len(documents)}")
    print(f"Questions: {len(questions)}")
    print(f"Agents: {agent_A.name} vs {agent_B.name}")
    print("=" * 70)

    # Run experiment
    records = run_experiment(agent_A, agent_B, questions, documents, config)

    # Compute significance
    sig_results = compute_significance(
        records, agent_A.name, agent_B.name, config.perm_test_iterations
    )

    # Save results
    save_results(records, sig_results, prefix="sync_")

    print("\n✅ Experiment complete.")


if __name__ == "__main__":
    main()
