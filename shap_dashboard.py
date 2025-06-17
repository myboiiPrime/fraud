import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from shared_utils import load_model_and_data, get_sample_data

st.set_page_config(page_title="SHAP Dashboard", layout="wide")

# Load data
model, X_test, y_test, feature_names, _, shap_explainer = load_model_and_data()

if model is None:
    st.stop()

st.title("⚡ SHAP Explanation Dashboard")
st.markdown("**SHapley Additive exPlanations**")

# Sample selection
sample_idx, sample, actual_label, prediction, predicted_label, prediction_proba = get_sample_data(model, X_test, y_test)

st.header(f"⚡ SHAP Analysis for Transaction #{sample_idx}")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown(f"""
    **Transaction Summary:**
    - **Actual**: {actual_label}
    - **Predicted**: {predicted_label}
    - **Confidence**: {prediction_proba:.1%}
    """)

with col1:
    with st.spinner("Generating SHAP explanation..."):
        # Get SHAP values for this instance
        shap_values = shap_explainer(sample.values.reshape(1, -1))
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

# SHAP values table
st.subheader("📊 SHAP Values Analysis")

shap_data = []
for i, feature in enumerate(feature_names):
    shap_val = shap_values.values[0][i]
    feature_val = sample[feature]
    direction = "Increases" if shap_val > 0 else "Decreases"
    impact = "High" if abs(shap_val) > 0.1 else "Medium" if abs(shap_val) > 0.05 else "Low"
    
    shap_data.append({
        "Feature": feature,
        "Value": f"{feature_val:.4f}",
        "SHAP Value": f"{shap_val:.4f}",
        "Direction": direction,
        "Impact": impact
    })

shap_df = pd.DataFrame(shap_data)
shap_df = shap_df.reindex(shap_df['SHAP Value'].abs().sort_values(ascending=False).index)
st.dataframe(shap_df, use_container_width=True)

st.info("""
**⚡ SHAP Explanation Method:**
- Based on game theory (Shapley values) for fair feature attribution
- Shows how each feature pushes the prediction from the base value
- Provides consistent, theoretically sound explanations
- Can reveal complex interactions between features
- Offers both global and local explanations
""")