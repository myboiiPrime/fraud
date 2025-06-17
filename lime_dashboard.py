import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shared_utils import load_model_and_data, get_sample_data

st.set_page_config(page_title="LIME Dashboard", layout="wide")

# Load data
model, X_test, y_test, feature_names, lime_explainer, _ = load_model_and_data()

if model is None:
    st.stop()

st.title("🔍 LIME Explanation Dashboard")
st.markdown("**Local Interpretable Model-agnostic Explanations**")

# Sample selection
sample_idx, sample, actual_label, prediction, predicted_label, prediction_proba = get_sample_data(model, X_test, y_test)

st.header(f"🔍 LIME Analysis for Transaction #{sample_idx}")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown(f"""
    **Transaction Summary:**
    - **Actual**: {actual_label}
    - **Predicted**: {predicted_label}
    - **Confidence**: {prediction_proba:.1%}
    """)

with col1:
    with st.spinner("Generating LIME explanation..."):
        # Get LIME explanation
        exp = lime_explainer.explain_instance(
            data_row=sample.values, 
            predict_fn=model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Create LIME plot
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(12, 8)
        st.pyplot(fig)

# Feature contributions table
st.subheader("📋 Feature Contributions Analysis")

lime_data = []
for feature, weight in exp.as_list():
    direction = "Increases" if weight > 0 else "Decreases"
    impact = "High" if abs(weight) > 0.1 else "Medium" if abs(weight) > 0.05 else "Low"
    lime_data.append({
        "Feature": feature,
        "Weight": f"{weight:.4f}",
        "Direction": direction,
        "Impact": impact
    })

lime_df = pd.DataFrame(lime_data)
st.dataframe(lime_df, use_container_width=True)

st.info("""
**🔍 LIME Explanation Method:**
- Creates a simpler, interpretable model around this specific transaction
- Shows which features contributed positively or negatively to the fraud prediction
- Provides local explanations that help understand individual decisions
""")