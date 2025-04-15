import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image

# Set page config
st.set_page_config(page_title="PPE Violation Dashboard", layout="wide")

st.title("🦺 PPE Violation Monitoring Dashboard")

# Count violations
log_file = "violations.csv"
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    total_helmet = len(df[df["type"] == "No Helmet"])
    total_vest = len(df[df["type"] == "No Vest"])
    total_all = len(df)
else:
    df = pd.DataFrame(columns=["time", "type", "image"])
    total_helmet = total_vest = total_all = 0

# Show stats
st.metric("🚨 Total Violations", total_all)
st.metric("⛑️ Helmet Violations", total_helmet)
st.metric("🦺 Vest Violations", total_vest)

# Show latest image


#st.subheader("📸 Latest Violation Screenshot")

# Show log table
st.subheader("📋 Violation Log")
st.dataframe(df[::-1], use_container_width=True)
