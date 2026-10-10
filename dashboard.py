"""
dashboard.py
------------
Dashboard v2: reads directly from digital_twin.db (populated by
run_streaming_demo.py) and auto-refreshes, rather than Review 1's
static, manually-reloaded view. Shows sensor trends, the twin's health
score over time, and -- new in Review 2 -- a visible, timestamped log
of every threshold adjustment the FeedbackController made, so the
feedback loop's behavior is actually observable, not just computed.

Run with: streamlit run dashboard.py
"""

import sqlite3
import time
import pandas as pd
import streamlit as st

DB_PATH = "/home/claude/review2/digital_twin.db"
REFRESH_SECONDS = 3

st.set_page_config(page_title="Digital Twin \u2014 Predictive Maintenance", layout="wide")
st.title("Digital Twin \u2014 Predictive Maintenance Dashboard (v2)")
st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s \u00b7 reading live from {DB_PATH}")


@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_table(table_name):
    conn = get_connection()
    try:
        return pd.read_sql(f"SELECT * FROM {table_name} ORDER BY file_index", conn)
    except Exception:
        return pd.DataFrame()


def load_adjustments(table_name):
    conn = get_connection()
    try:
        return pd.read_sql(f"SELECT * FROM {table_name} ORDER BY adjusted_at", conn)
    except Exception:
        return pd.DataFrame()


sensor_df = load_table("sensor_readings")
pred_fixed = load_table("predictions_fixed")
pred_adaptive = load_table("predictions_adaptive")
adjustments = load_adjustments("threshold_adjustments_adaptive")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Files received", len(sensor_df))
col2.metric("Interpolated points", int(sensor_df["interpolated"].sum()) if len(sensor_df) else 0)
col3.metric("Flagged sensor_uncertain", int(sensor_df["sensor_uncertain"].sum()) if len(sensor_df) else 0)
col4.metric("Threshold adjustments made", len(adjustments))

st.subheader("Sensor trend \u2014 RMS over time")
if len(sensor_df):
    st.line_chart(sensor_df.set_index("file_index")[["rms"]])
else:
    st.info("Waiting for data...")

st.subheader("Twin health score: FIXED vs ADAPTIVE threshold")
if len(pred_fixed) and len(pred_adaptive):
    chart_df = pd.DataFrame({
        "file_index": pred_fixed["file_index"],
        "degradation_probability": pred_fixed["degradation_probability"].values,
        "fixed_threshold": pred_fixed["alert_threshold"].values,
        "adaptive_threshold": pred_adaptive["alert_threshold"].values,
    }).set_index("file_index")
    st.line_chart(chart_df)
else:
    st.info("Waiting for predictions...")

st.subheader("Threshold adjustment log (FeedbackController)")
if len(adjustments):
    st.dataframe(
        adjustments[["file_index", "old_threshold", "new_threshold", "reason"]],
        use_container_width=True,
    )
else:
    st.info("No adjustments logged yet.")

st.subheader("Recent alerts (adaptive controller)")
if len(pred_adaptive):
    alerts = pred_adaptive[pred_adaptive["alert_fired"] == 1].tail(20)
    st.dataframe(alerts[["file_index", "degradation_probability", "alert_threshold"]],
                 use_container_width=True)
else:
    st.info("No alerts yet.")

time.sleep(REFRESH_SECONDS)
st.rerun()
