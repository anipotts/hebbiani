"""
RabbitMQ worker that consumes extraction tasks and executes the LLM pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import pika
from openai import OpenAI

from consensus_core import AgentConfig, Document, Question, parse_final_answer
from config import get_documents, get_experiment_config, get_agents, get_questions
from pipeline import (
    process_single_turn_sync,
)
from schemas import ResultPayload, TaskPayload

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
TASK_QUEUE = "extraction_tasks"
DLX_EXCHANGE = "dlx_exchange"
DLQ_ROUTING_KEY = "failed_tasks_dlq"
DLQ_QUEUE = "failed_tasks_dlq"
RUN_LOG_PATH = os.getenv(
    "RABBITMQ_RUN_LOG", "experiment_logs/run_records_rabbitmq.jsonl"
)
MAX_TASK_RETRIES = 3

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it by running: export OPENAI_API_KEY='your-api-key-here'"
    )
client = OpenAI(api_key=api_key)


class TaskContext:
    def __init__(self):
        self.config = get_experiment_config()
        documents = get_documents()
        self.doc_map: Dict[str, Document] = {doc.doc_id: doc for doc in documents}
        agent_A, agent_B = get_agents()
        self.agent_map: Dict[str, AgentConfig] = {
            agent_A.name: agent_A,
            agent_B.name: agent_B,
        }
        questions = get_questions(documents)
        self.question_map: Dict[str, Question] = {q.id: q for q in questions}

    def document(self, doc_id: str) -> Document:
        if doc_id not in self.doc_map:
            raise ValueError(f"Unknown doc_id {doc_id}")
        return self.doc_map[doc_id]

    def agent(self, name: str) -> AgentConfig:
        if name not in self.agent_map:
            raise ValueError(f"Unknown agent {name}")
        return self.agent_map[name]

    def question(self, question_id: str) -> Question:
        if question_id not in self.question_map:
            raise ValueError(f"Unknown question {question_id}")
        return self.question_map[question_id]


context = TaskContext()
os.makedirs(Path(RUN_LOG_PATH).parent, exist_ok=True)


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


def append_record(record):
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.__dict__, default=float) + "\n")


def republish_task(channel, payload: TaskPayload):
    channel.basic_publish(
        exchange="",
        routing_key=TASK_QUEUE,
        body=payload.model_dump_json(),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )


def publish_to_dlq(channel, payload: ResultPayload):
    channel.basic_publish(
        exchange=DLX_EXCHANGE,
        routing_key=DLQ_ROUTING_KEY,
        body=payload.model_dump_json(),
        properties=pika.BasicProperties(content_type="application/json"),
    )


def process_payload(payload: TaskPayload):
    document = context.document(payload.doc_id)
    question = context.question(payload.question_id)
    agent = context.agent(payload.agent_name)

    record = process_single_turn_sync(
        client,
        agent,
        question,
        document,
        payload.temperature,
        context.config.api_model_eval,
        payload.run_id,
    )
    append_record(record)


def on_message(channel, method, properties, body):
    try:
        payload = TaskPayload.model_validate_json(body)
    except Exception as exc:  # pylint: disable=broad-except
        error_payload = ResultPayload(
            run_id=-1,
            doc_id="unknown",
            question_id="unknown",
            agent_name="unknown",
            agent_model="unknown",
            temperature=0.0,
            final_answer="",
            expected_score=0.0,
            error=f"Payload validation error: {exc}",
        )
        publish_to_dlq(channel, error_payload)
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    try:
        process_payload(payload)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as exc:  # pylint: disable=broad-except
        if payload.retry_count < MAX_TASK_RETRIES:
            payload.retry_count += 1
            republish_task(channel, payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            print(
                f"Requeued task {payload.question_id} (agent={payload.agent_name}) "
                f"retry={payload.retry_count}"
            )
        else:
            error_payload = ResultPayload(
                run_id=payload.run_id,
                doc_id=payload.doc_id,
                question_id=payload.question_id,
                agent_name=payload.agent_name,
                agent_model=payload.agent_model,
                temperature=payload.temperature,
                final_answer="",
                expected_score=0.0,
                error=str(exc),
            )
            publish_to_dlq(channel, error_payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
    )
    channel = connection.channel()
    _declare_topology(channel)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=TASK_QUEUE, on_message_callback=on_message)

    print("Worker listening for tasks. Press Ctrl+C to exit.")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()

