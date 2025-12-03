"""
RabbitMQ producer that enumerates Guidance-Eval tasks and pushes them onto a queue.
"""

from __future__ import annotations

import json
import os
from typing import List

import pika

from config import get_documents, get_experiment_config, get_agents, get_questions
from schemas import TaskPayload

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))

TASK_QUEUE = "extraction_tasks"
DLX_EXCHANGE = "dlx_exchange"
DLQ_QUEUE = "failed_tasks_dlq"
DLQ_ROUTING_KEY = "failed_tasks_dlq"


def _declare_topology(channel: pika.adapters.blocking_connection.BlockingChannel):
    channel.exchange_declare(exchange=DLX_EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=DLQ_QUEUE, durable=True)
    channel.queue_bind(queue=DLQ_QUEUE, exchange=DLX_EXCHANGE, routing_key=DLQ_ROUTING_KEY)
    channel.queue_declare(
        queue=TASK_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": DLX_EXCHANGE,
            "x-dead-letter-routing-key": DLQ_ROUTING_KEY,
        },
    )


def _publish_tasks(channel) -> int:
    config = get_experiment_config()
    documents = get_documents()
    agent_A, agent_B = get_agents()
    questions = get_questions(documents)

    doc_map = {doc.doc_id: doc for doc in documents}
    tasks_enqueued = 0

    for run_id in range(config.n_bootstrap_runs):
        for question in questions:
            document = doc_map.get(question.doc_id)
            if document is None:
                continue
            for agent in [agent_A, agent_B]:
                for temperature in config.temperatures:
                    payload = TaskPayload(
                        run_id=run_id,
                        doc_id=document.doc_id,
                        question_id=question.id,
                        agent_name=agent.name,
                        agent_model=agent.model,
                        temperature=temperature,
                        question_text=question.text,
                        criterion=question.criterion,
                    )
                    channel.basic_publish(
                        exchange="",
                        routing_key=TASK_QUEUE,
                        body=payload.model_dump_json(),
                        properties=pika.BasicProperties(
                            delivery_mode=2,  # persistent
                            content_type="application/json",
                        ),
                    )
                    tasks_enqueued += 1
    return tasks_enqueued


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
    )
    channel = connection.channel()
    try:
        _declare_topology(channel)
        total = _publish_tasks(channel)
        print(f"Enqueued {total} tasks onto '{TASK_QUEUE}'.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()



