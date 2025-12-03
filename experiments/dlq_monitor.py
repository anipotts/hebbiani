"""
Utility script to inspect messages routed to the failed_tasks_dlq.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import pika

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
DLQ_QUEUE = "failed_tasks_dlq"


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
    )
    channel = connection.channel()
    channel.queue_declare(queue=DLQ_QUEUE, durable=True)

    error_counts = Counter()
    agent_counts = Counter()
    question_counts = Counter()

    print("Scanning DLQ (messages will be requeued)...")
    while True:
        method, properties, body = channel.basic_get(queue=DLQ_QUEUE, auto_ack=False)
        if method is None:
            break
        try:
            payload = json.loads(body)
            error = payload.get("error", "unknown_error")
            error_counts[error] += 1
            agent_counts[payload.get("agent_name", "unknown_agent")] += 1
            question_counts[payload.get("question_id", "unknown_question")] += 1
        except json.JSONDecodeError:
            error_counts["json_decode_error"] += 1
        finally:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    connection.close()

    if not error_counts:
        print("DLQ is empty.")
        return

    print("\nTop error types:")
    for error, count in error_counts.most_common(5):
        print(f"  {error}: {count}")

    print("\nTop agents impacted:")
    for agent, count in agent_counts.most_common(5):
        print(f"  {agent}: {count}")

    print("\nTop questions impacted:")
    for question, count in question_counts.most_common(5):
        print(f"  {question}: {count}")


if __name__ == "__main__":
    main()



