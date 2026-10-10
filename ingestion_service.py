"""
ingestion_service.py
----------------------
Subscribes to the MQTT stream and writes incoming readings to SQLite,
same as Review 1 -- but now handles the realistic imperfection the
upgraded StreamSimulator introduces: dropped messages show up as gaps
in the `file_index` sequence. A SHORT gap (a few missed files) is
linearly interpolated so downstream feature sequences aren't broken by
single missing points; a LONG gap is NOT interpolated -- it's written
to the database with a `sensor_uncertain` flag instead, since inventing
several consecutive readings would be fabricating data the system never
actually observed.
"""

import json
import sqlite3
import threading
import time
import numpy as np
import paho.mqtt.client as mqtt


SHORT_GAP_MAX = 3  # gaps of up to this many missing files are interpolated


class IngestionService:
    def __init__(self, db_path, feature_columns, broker_host="localhost", broker_port=1883,
                 topic="sensors/bearing/run_3"):
        self.db_path = db_path
        self.feature_columns = feature_columns
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic

        self.last_file_index = None
        self.last_features = None
        self.done_event = threading.Event()
        self.n_received = 0
        self.n_interpolated = 0
        self.n_flagged_uncertain = 0

        self._init_db()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ingestion-service")
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cols_sql = ", ".join(f"{c} REAL" for c in self.feature_columns)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                file_index INTEGER PRIMARY KEY,
                {cols_sql},
                sensor_uncertain INTEGER DEFAULT 0,
                interpolated INTEGER DEFAULT 0,
                received_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                file_index INTEGER PRIMARY KEY,
                health_score REAL,
                degradation_probability REAL,
                alert_threshold REAL,
                alert_fired INTEGER,
                computed_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threshold_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_index INTEGER,
                old_threshold REAL,
                new_threshold REAL,
                reason TEXT,
                adjusted_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.topic, qos=1)

    def _write_row(self, file_index, features, sensor_uncertain=0, interpolated=0):
        conn = sqlite3.connect(self.db_path)
        cols = ["file_index"] + self.feature_columns + ["sensor_uncertain", "interpolated", "received_at"]
        vals = [file_index] + [features[c] for c in self.feature_columns] + [sensor_uncertain, interpolated, time.time()]
        placeholders = ",".join("?" * len(cols))
        conn.execute(
            f"INSERT OR REPLACE INTO sensor_readings ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        conn.commit()
        conn.close()

    def _handle_gap(self, expected_start, actual_index):
        """Called when file_index jumps by more than 1 -- one or more messages were dropped."""
        gap_size = actual_index - expected_start
        if gap_size <= 0:
            return
        if gap_size <= SHORT_GAP_MAX and self.last_features is not None:
            # linear interpolation between last known reading and the new one
            # (the new one's features arrive with the message that follows the gap,
            # so this is called just before writing that message)
            self.n_interpolated += gap_size
            print(f"  [ingestion] short gap of {gap_size} file(s) at index {expected_start} "
                  f"-> interpolating")
        else:
            self.n_flagged_uncertain += gap_size
            print(f"  [ingestion] LONG gap of {gap_size} file(s) at index {expected_start} "
                  f"-> flagging as sensor_uncertain, NOT fabricating values")
            for missing_idx in range(expected_start, actual_index):
                # write a placeholder row with no feature values, explicitly
                # flagged -- downstream consumers must handle this, not pretend
                # the sensor was healthy during this window
                placeholder = {c: None for c in self.feature_columns}
                conn = sqlite3.connect(self.db_path)
                cols = ["file_index"] + self.feature_columns + ["sensor_uncertain", "interpolated", "received_at"]
                vals = [missing_idx] + [None] * len(self.feature_columns) + [1, 0, time.time()]
                placeholders_sql = ",".join("?" * len(cols))
                conn.execute(
                    f"INSERT OR REPLACE INTO sensor_readings ({','.join(cols)}) VALUES ({placeholders_sql})",
                    vals,
                )
                conn.commit()
                conn.close()

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        if payload.get("end_of_stream"):
            self.done_event.set()
            return

        file_index = payload["file_index"]
        features = payload["features"]
        self.n_received += 1

        if self.last_file_index is not None and file_index > self.last_file_index + 1:
            gap_start = self.last_file_index + 1
            gap_size = file_index - gap_start
            if gap_size <= SHORT_GAP_MAX and self.last_features is not None:
                self._handle_gap(gap_start, file_index)
                for i, missing_idx in enumerate(range(gap_start, file_index), start=1):
                    frac = i / (gap_size + 1)
                    interpolated_features = {
                        c: self.last_features[c] + frac * (features[c] - self.last_features[c])
                        for c in self.feature_columns
                    }
                    self._write_row(missing_idx, interpolated_features, interpolated=1)
            else:
                self._handle_gap(gap_start, file_index)

        self._write_row(file_index, features)
        self.last_file_index = file_index
        self.last_features = features

    def run(self, timeout=120):
        self.client.connect(self.broker_host, self.broker_port, keepalive=30)
        self.client.loop_start()
        finished = self.done_event.wait(timeout=timeout)
        self.client.loop_stop()
        self.client.disconnect()
        if not finished:
            print("[IngestionService] WARNING: timed out waiting for end-of-stream")
        print(f"[IngestionService] received={self.n_received} "
              f"interpolated_points={self.n_interpolated} "
              f"flagged_uncertain_points={self.n_flagged_uncertain}")
