"""
stream_simulator.py
--------------------
Upgrades Review 1's clean batch replay into a realistic live stream:
publishes one sensor reading at a time over MQTT, with configurable
per-message delay and a configurable probability of a message being
dropped entirely (simulating real IoT network loss) -- rather than a
tidy, guaranteed-delivery batch read of the whole file.
"""

import json
import time
import random
import pandas as pd
import paho.mqtt.client as mqtt


class StreamSimulator:
    def __init__(self, broker_host="localhost", broker_port=1883, topic="sensors/bearing/run_3",
                 delay_seconds=0.02, dropout_prob=0.05, seed=42):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.delay_seconds = delay_seconds
        self.dropout_prob = dropout_prob
        self.rng = random.Random(seed)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="stream-simulator")
        self.n_published = 0
        self.n_dropped = 0

    def connect(self):
        self.client.connect(self.broker_host, self.broker_port, keepalive=30)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish_run(self, df: pd.DataFrame, feature_columns: list):
        """
        Publishes each row of df as one MQTT message, in order, with a
        realistic delay and random dropout. df must have a `file_index`
        column so the IngestionService can detect gaps caused by dropout.
        """
        self.connect()
        for _, row in df.iterrows():
            if self.rng.random() < self.dropout_prob:
                # simulate a lost packet: never sent, ingestion service
                # must detect the resulting gap in file_index on its own
                self.n_dropped += 1
                time.sleep(self.delay_seconds)
                continue

            payload = {
                "file_index": int(row["file_index"]),
                "features": {col: float(row[col]) for col in feature_columns},
                "sent_at": time.time(),
            }
            self.client.publish(self.topic, json.dumps(payload), qos=1)
            self.n_published += 1
            time.sleep(self.delay_seconds)

        # sentinel message marking end of stream
        self.client.publish(self.topic, json.dumps({"end_of_stream": True}), qos=1)
        time.sleep(0.2)
        self.disconnect()
        print(f"[StreamSimulator] published={self.n_published} dropped={self.n_dropped} "
              f"(dropout rate observed: {self.n_dropped/(self.n_published+self.n_dropped):.1%})")
