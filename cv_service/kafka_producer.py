# cv_service/kafka_producer.py
# Sends equipment analysis results to the Kafka topic.
# Each message is a JSON payload matching the assessment spec.

import json
import os
import time
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable


KAFKA_TOPIC = "equipment.events"

# FIX: read from env var so Docker containers can use internal broker address
# (e.g. "kafka:29092") while local processes use "localhost:9092"
KAFKA_SERVERS = os.getenv("KAFKA_SERVERS", "localhost:9092").split(",")


class EquipmentProducer:
    def __init__(self, retries: int = 10, retry_delay: float = 3.0):
        """
        Connect to Kafka with automatic retry.
        Retries are needed because Kafka takes ~15s to fully start
        after docker-compose up.
        """
        self.producer = None
        print(f"[Kafka] Connecting to {KAFKA_SERVERS}...")

        for attempt in range(1, retries + 1):
            try:
                print(f"[Kafka] Connecting... attempt {attempt}/{retries}")
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=3
                )
                print("[Kafka] Connected successfully.")
                return

            except NoBrokersAvailable:
                if attempt < retries:
                    print(f"[Kafka] Broker not ready, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        "[Kafka] Could not connect after max retries. "
                        "Is docker-compose running?"
                    )

    def send(self, payload: dict):
        """Send one payload dict to the equipment.events topic (non-blocking)."""
        if self.producer is None:
            return
        self.producer.send(KAFKA_TOPIC, value=payload)

    def flush(self):
        """Force-send any buffered messages. Call at end of video."""
        if self.producer:
            self.producer.flush()

    def close(self):
        """Clean shutdown."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            print("[Kafka] Producer closed.")