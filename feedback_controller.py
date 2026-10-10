"""
feedback_controller.py
------------------------
The centerpiece of Review 2, and the direct response to Gap 1 (no
twin-to-physical feedback loop). Most published digital-twin systems
predict a failure and stop there; this controller closes the loop by
dynamically tightening the alert threshold as the twin's own health
score trends toward failure, and logging every adjustment it makes.

Two modes are compared directly in run_streaming_demo.py:
  - FIXED:    a single alert threshold (0.5) used for the entire run
  - ADAPTIVE: this FeedbackController, which lowers the threshold as
              health deteriorates (more sensitive when it matters most)
              and raises it back during sustained healthy periods
"""

import sqlite3
import time


class FeedbackController:
    def __init__(self, db_path, base_threshold=0.5, min_threshold=0.20, max_threshold=0.70,
                 sensitivity_window=15, trend_margin=0.05,
                 predictions_table="predictions", adjustments_table="threshold_adjustments",
                 adaptive=True):
        """
        base_threshold     : starting alert threshold (same as the FIXED baseline)
        min_threshold       : floor -- never becomes more sensitive than this
        max_threshold       : ceiling -- never becomes less sensitive than this
        sensitivity_window  : how many recent health scores to look at when
                              judging whether the trend is worsening or improving
        trend_margin        : how much the recent average must rise/fall
                              before the controller reacts, to avoid reacting
                              to single-point noise
        """
        self.db_path = db_path
        self.threshold = base_threshold
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.sensitivity_window = sensitivity_window
        self.trend_margin = trend_margin
        self.recent_scores = []
        self.predictions_table = predictions_table
        self.adjustments_table = adjustments_table
        # adaptive=False reproduces the FIXED baseline: process() still logs
        # everything, but the threshold never moves, giving a true apples-
        # to-apples comparison on the identical probability sequence.
        self.adaptive = adaptive
        self._init_tables()

    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.predictions_table} (
                file_index INTEGER PRIMARY KEY,
                health_score REAL,
                degradation_probability REAL,
                alert_threshold REAL,
                alert_fired INTEGER,
                computed_at REAL
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.adjustments_table} (
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

    def _log_adjustment(self, file_index, old_threshold, new_threshold, reason):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"INSERT INTO {self.adjustments_table} (file_index, old_threshold, new_threshold, reason, adjusted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_index, old_threshold, new_threshold, reason, time.time()),
        )
        conn.commit()
        conn.close()

    def _log_prediction(self, file_index, health_score, degradation_probability, alert_fired):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"INSERT OR REPLACE INTO {self.predictions_table} "
            "(file_index, health_score, degradation_probability, alert_threshold, alert_fired, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (file_index, health_score, degradation_probability, self.threshold, int(alert_fired), time.time()),
        )
        conn.commit()
        conn.close()

    def process(self, file_index: int, degradation_probability: float) -> dict:
        """
        Called once per incoming prediction. Updates the running trend,
        adjusts the threshold if warranted, fires an alert if the current
        probability crosses the (possibly just-adjusted) threshold, and
        logs everything.
        """
        health_score = 1.0 - degradation_probability
        self.recent_scores.append(degradation_probability)
        if len(self.recent_scores) > self.sensitivity_window:
            self.recent_scores.pop(0)

        if self.adaptive and len(self.recent_scores) >= self.sensitivity_window:
            recent_avg = sum(self.recent_scores) / len(self.recent_scores)
            older_reference = self.base_threshold  # compare recent trend against the resting baseline

            old_threshold = self.threshold
            if recent_avg > older_reference + self.trend_margin:
                # trending toward degradation -> become MORE sensitive (lower threshold)
                new_threshold = max(self.min_threshold, self.threshold - 0.05)
                if new_threshold != old_threshold:
                    self.threshold = new_threshold
                    self._log_adjustment(file_index, old_threshold, new_threshold,
                                          f"recent avg probability {recent_avg:.3f} trending up -> tightening")
            elif recent_avg < older_reference - self.trend_margin and self.threshold < self.base_threshold:
                # sustained healthy period -> relax back toward the base threshold
                new_threshold = min(self.base_threshold, self.threshold + 0.02)
                if new_threshold != old_threshold:
                    self.threshold = new_threshold
                    self._log_adjustment(file_index, old_threshold, new_threshold,
                                          f"recent avg probability {recent_avg:.3f} trending down -> relaxing")

        alert_fired = degradation_probability >= self.threshold
        self._log_prediction(file_index, health_score, degradation_probability, alert_fired)
        return {
            "file_index": file_index,
            "health_score": health_score,
            "degradation_probability": degradation_probability,
            "threshold": self.threshold,
            "alert_fired": alert_fired,
        }
