import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("buyer_segments.csv")

# Dashboard title
st.title("Buyer Segmentation Dashboard")

# Show first 5 rows
st.write(df.head())

# Cluster distribution chart
st.subheader("Cluster Distribution")

st.bar_chart(df['Cluster'].value_counts())