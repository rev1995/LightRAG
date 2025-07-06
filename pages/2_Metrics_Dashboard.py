# pages/2_Metrics_Dashboard.py

import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import pytz

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
IST = pytz.timezone('Asia/Kolkata')

st.set_page_config(
    page_title="Metrics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data(ttl=5)
def get_metrics():
    """Fetches the latest LLM metrics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=2)
        if response.status_code == 200:
            return response.json().get("llm_calls", [])
        return None
    except requests.RequestException:
        return None

def process_metrics_for_chart(metrics_data: list):
    """Processes raw metrics to create a time-series DataFrame for charting."""
    if not metrics_data:
        return pd.DataFrame(columns=["Time Bin", "Call Count"])

    df = pd.DataFrame(metrics_data)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['start_time'] = df['start_time'].dt.tz_convert(IST)
    
    time_window = datetime.now(IST) - timedelta(minutes=30)
    df = df[df['start_time'] >= time_window]

    if df.empty:
        return pd.DataFrame(columns=["Time Bin", "Call Count"])

    df = df.set_index('start_time').resample('30s').size().reset_index(name='Call Count')
    df.rename(columns={'start_time': 'Time Bin'}, inplace=True)
    return df

# --- UI Initialization ---
st.title("📊 LLM Call Metrics Dashboard")
st.caption("Live monitoring of API calls made to the language model. All times are in IST.")

if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

metrics_data = get_metrics()

if metrics_data:
    st.subheader("LLM Calls per 30s (last 30 mins)")
    chart_df = process_metrics_for_chart(metrics_data)
    if not chart_df.empty:
        st.bar_chart(chart_df, x="Time Bin", y="Call Count", color="#ffaa00")
    else:
        st.info("No recent LLM calls in the last 30 minutes.")

    st.divider()

    st.subheader("Recent LLM Call Log")
    df = pd.DataFrame(metrics_data)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['start_time'] = df['start_time'].dt.tz_convert(IST)
    df = df.sort_values(by="start_time", ascending=False)
    
    # --- FIX: Create an explicit copy to avoid SettingWithCopyWarning ---
    df_display = df[['start_time', 'purpose', 'status', 'duration_sec', 'pid', 'call_id']].copy()
    
    # Now, modifications are safe and will not raise a warning.
    df_display['start_time'] = df_display['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_display.rename(columns={
        'start_time': 'Timestamp (IST)',
        'purpose': 'Purpose',
        'status': 'Status',
        'duration_sec': 'Duration (s)',
        'pid': 'Process ID',
        'call_id': 'Call ID'
    }, inplace=True)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No metrics data available yet. Make a query or upload a file on the Chat page.")

time.sleep(10)
st.rerun()