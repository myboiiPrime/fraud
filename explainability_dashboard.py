import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
import os

# Set page config
st.set_page_config(page_title="Fraud Detection Explainability Dashboard", layout="wide")

# Load model and data
@st.cache_resource
def load_model_and_data():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')['is_fraud']
    
    # Create explainers
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns,
        class_names=['Not Fraud', 'Fraud'],
        mode='classification',
        random_state=42
    )
    
    shap_explainer = shap.Explainer(model)
    
    return model, X_test, y_test, lime_explainer, shap_explainer

model, X_test, y_test, lime_explainer, shap_explainer = load_model_and_data()

# Title and introduction
st.title("Fraud Detection Model Explainability Comparison")

st.markdown("""
This dashboard demonstrates three different approaches to fraud detection model deployment:
1. **Black Box Model**: No explainability - only predictions are provided
2. **LIME Explanations**: Local explanations using LIME (Local Interpretable Model-agnostic Explanations)
3. **SHAP Explanations**: Global and local explanations using SHAP (SHapley Additive exPlanations)
""")

# Sample selection
st.sidebar.header("Sample Selection")

# Find fraud and non-fraud examples
fraud_indices = np.where(y_test == 1)[0]
non_fraud_indices = np.where(y_test == 0)[0]

sample_type = st.sidebar.radio("Select sample type:", ["Fraud Transaction", "Non-Fraud Transaction"])

if sample_type == "Fraud Transaction":
    sample_idx = st.sidebar.selectbox("Select fraud sample ID:", fraud_indices)
else:
    sample_idx = st.sidebar.selectbox("Select non-fraud sample ID:", non_fraud_indices)

# Get sample data
sample = X_test.iloc[sample_idx]
actual_label = "Fraud" if y_test.iloc[sample_idx] == 1 else "Not Fraud"
prediction = model.predict(sample.values.reshape(1, -1))[0]
predicted_label = "Fraud" if prediction == 1 else "Not Fraud"
prediction_proba = model.predict_proba(sample.values.reshape(1, -1))[0][1]

# Display sample information
st.header("Transaction Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Transaction Amount", f"${sample['transaction_amount']:.2f}")
with col2:
    st.metric("Transaction Hour", f"{int(sample['transaction_hour'])}:00")
with col3:
    st.metric("Customer Age", f"{int(sample['customer_age'])} years")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Account Age", f"{int(sample['account_age_days'])} days")
with col2:
    st.metric("Distance from Home", f"{sample['distance_from_home']:.1f} miles")
with col3:
    st.metric("Foreign Transaction", "Yes" if sample['foreign_transaction'] == 1 else "No")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("High Risk Merchant", "Yes" if sample['high_risk_merchant'] == 1 else "No")
with col2:
    st.metric("Online Transaction", "Yes" if sample['online_transaction'] == 1 else "No")
with col3:
    st.metric("Transactions (24h)", int(sample['n_transactions_last_24h']))

# Tabs for different explainability methods
tab1, tab2, tab3 = st.tabs(["Black Box Model", "LIME Explanation", "SHAP Explanation"])

# Tab 1: Black Box Model
with tab1:
    st.header("Black Box Model (No Explainability)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Label", actual_label)
    with col2:
        st.metric("Predicted Label", predicted_label, 
                 delta="Correct" if actual_label == predicted_label else "Incorrect")
    
    st.progress(prediction_proba)
    st.write(f"Fraud Probability: {prediction_proba:.2%}")
    
    st.warning("⚠️ No explanation is available for why this prediction was made.")
    
    st.markdown("""
    **Limitations of Black Box Models:**
    - No transparency into decision-making process
    - Difficult to identify potential biases or errors
    - Cannot provide meaningful feedback to customers
    - Challenging to improve based on domain knowledge
    """)

# Tab 2: LIME Explanation
with tab2:
    st.header("LIME Explanation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Label", actual_label)
    with col2:
        st.metric("Predicted Label", predicted_label, 
                 delta="Correct" if actual_label == predicted_label else "Incorrect")
    
    # Get LIME explanation
    exp = lime_explainer.explain_instance(
        data_row=sample.values, 
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    # Plot LIME explanation
    fig, ax = plt.subplots(figsize=(10, 6))
    exp.as_pyplot_figure(ax=ax)
    st.pyplot(fig)
    
    st.markdown("**Feature Contributions:**")
    for feature, weight in exp.as_list():
        if weight > 0:
            st.markdown(f"- 🔴 {feature}: **+{weight:.4f}** (increases fraud probability)")
        else:
            st.markdown(f"- 🔵 {feature}: **{weight:.4f}** (decreases fraud probability)")
    
    st.markdown("""
    **LIME Explanation Method:**
    - Creates a simpler, interpretable model around the specific instance
    - Shows which features contributed positively or negatively to the prediction
    - Helps understand why this particular transaction was flagged
    - Local explanation only - doesn't provide insights about the model's overall behavior
    """)

# Tab 3: SHAP Explanation
with tab3:
    st.header("SHAP Explanation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Label", actual_label)
    with col2:
        st.metric("Predicted Label", predicted_label, 
                 delta="Correct" if actual_label == predicted_label else "Incorrect")
    
    # Get SHAP values
    shap_values = shap_explainer(sample.values.reshape(1, -1))
    
    # Force plot
    st.subheader("SHAP Force Plot")
    st.write("Shows how each feature pushes the prediction from the base value (average model output) to the final prediction")
    
    fig, ax = plt.subplots(figsize=(12, 3))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    
    # Feature importance for this prediction
    st.subheader("Feature Contributions")
    
    # Get feature values and their SHAP values
    features_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Value': sample.values,
        'SHAP Value': shap_values.values[0]
    })
    
    # Sort by absolute SHAP value
    features_df['Abs SHAP'] = features_df['SHAP Value'].abs()
    features_df = features_df.sort_values('Abs SHAP', ascending=False)
    
    for _, row in features_df.iterrows():
        feature = row['Feature']
        value = row['Value']
        shap_value = row['SHAP Value']
        
        if shap_value > 0:
            st.markdown(f"- 🔴 **{feature}** = {value:.2f}: **+{shap_value:.4f}** (increases fraud probability)")
        else:
            st.markdown(f"- 🔵 **{feature}** = {value:.2f}: **{shap_value:.4f}** (decreases fraud probability)")
    
    st.markdown("""
    **SHAP Explanation Method:**
    - Based on game theory (Shapley values)
    - Shows how each feature contributes to pushing the prediction from the base value
    - Provides consistent, theoretically sound explanations
    - Can reveal complex interactions between features
    - Offers both global and local explanations
    """)

# Comparison section
st.header("Explainability Comparison")

st.markdown("""
| Aspect | Black Box | LIME | SHAP |
|--------|-----------|------|------|
| Transparency | ❌ None | ✅ Local | ✅ Local & Global |
| Theoretical Foundation | N/A | Locally weighted linear models | Game theory (Shapley values) |
| Consistency | N/A | May vary between runs | Consistent |
| Captures Feature Interactions | N/A | Limited | ✅ Yes |
| Computational Cost | Low | Medium | High |
| Implementation Complexity | Low | Medium | Medium-High |
| Suitable for | Simple applications with low risk | Medium-risk applications | High-risk, regulated applications |
""")

st.markdown("""
### Key Takeaways

1. **Black Box Models** provide no transparency, making them unsuitable for high-stakes decisions like fraud detection where explanations are crucial for stakeholder trust and regulatory compliance.

2. **LIME** offers a good balance between performance and explainability, providing intuitive local explanations that help understand individual predictions.

3. **SHAP** provides the most comprehensive explanations with strong theoretical guarantees, making it ideal for regulated environments where consistency and completeness of explanations are required.

For fraud detection systems, implementing either LIME or SHAP significantly improves transparency and trustworthiness compared to black box models, with SHAP being the preferred choice when thorough explanations are required.
""")