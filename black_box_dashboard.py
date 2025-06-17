import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from shared_utils import load_model_and_data, get_sample_data

st.set_page_config(page_title="Black Box Dashboard", layout="wide")

# Load data
model, X_test, y_test, feature_names, _, _ = load_model_and_data()

if model is None:
    st.stop()

st.title("🔒 Black Box Model Dashboard")
st.markdown("**No explainability - only predictions**")

# Sample selection (shared component)
sample_idx, sample, actual_label, prediction, predicted_label, prediction_proba = get_sample_data(model, X_test, y_test)

# Transaction overview
st.header(f"📊 Transaction #{sample_idx} Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card {'fraud-card' if actual_label == '🚨 Fraud' else 'safe-card'}">
        <h3>Actual Label</h3>
        <h2>{actual_label}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    is_correct = actual_label.split()[1] == predicted_label.split()[1]
    st.markdown(f"""
    <div class="metric-card {'safe-card' if is_correct else 'fraud-card'}">
        <h3>Predicted Label</h3>
        <h2>{predicted_label}</h2>
        <p>{'✅ Correct' if is_correct else '❌ Incorrect'}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Fraud Probability</h3>
        <h2>{prediction_proba:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

# Risk assessment
st.subheader("Fraud Risk Assessment")
st.progress(float(prediction_proba))

if prediction_proba > 0.8:
    st.error("🚨 HIGH FRAUD RISK - Immediate investigation recommended")
elif prediction_proba > 0.5:
    st.warning("⚠️ MODERATE FRAUD RISK - Additional verification suggested")
else:
    st.success("✅ LOW FRAUD RISK - Transaction appears legitimate")

st.warning("""
**⚠️ Limitations of Black Box Models:**
- No transparency into decision-making process
- Cannot identify which factors influenced the decision
- Difficult to build trust with stakeholders
- Challenging to improve model based on domain knowledge
- Regulatory compliance issues in financial services
""")