import pandas as pd
import numpy as np
import pickle
import streamlit as st
import shap
from lime import lime_tabular

@st.cache_resource
def load_model_and_data():
    try:
        with open('real_fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        X_test = pd.read_csv('X_test_real.csv')
        y_test_df = pd.read_csv('y_test_real.csv')
        y_test = y_test_df['is_fraud'].values  # Convert to numpy array
        
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Create explainers
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_test.values,
            feature_names=feature_names,
            class_names=['Not Fraud', 'Fraud'],
            mode='classification',
            random_state=42
        )
        
        shap_explainer = shap.Explainer(model)
        
        return model, X_test, y_test, feature_names, lime_explainer, shap_explainer
    
    except Exception as e:
        st.error(f"Error loading model and data: {e}")
        st.error("Please run real_fraud_detection.py first to generate the required files.")
        return None, None, None, None, None, None

def get_sample_data(model, X_test, y_test):
    # Ensure y_test is a proper numpy array
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    y_test = np.atleast_1d(y_test)
    
    # Sample selection logic
    fraud_indices = np.where(y_test == 1)[0]
    non_fraud_indices = np.where(y_test == 0)[0]
    
    sample_type = st.sidebar.radio(
        "Select transaction type:", 
        ["🚨 Fraudulent Transaction", "✅ Legitimate Transaction"]
    )
    
    if sample_type == "🚨 Fraudulent Transaction":
        if len(fraud_indices) > 0:
            sample_idx = st.sidebar.selectbox(
                "Select fraud sample:", 
                fraud_indices,
                format_func=lambda x: f"Transaction #{x}"
            )
        else:
            st.sidebar.error("No fraudulent transactions found in test set")
            st.stop()
    else:
        sample_idx = st.sidebar.selectbox(
            "Select legitimate sample:", 
            non_fraud_indices[:100],
            format_func=lambda x: f"Transaction #{x}"
        )
    
    # Get sample data
    sample = X_test.iloc[sample_idx]
    actual_label = "🚨 Fraud" if y_test[sample_idx] == 1 else "✅ Legitimate"
    prediction = model.predict(sample.values.reshape(1, -1))[0]
    predicted_label = "🚨 Fraud" if prediction == 1 else "✅ Legitimate"
    prediction_proba = model.predict_proba(sample.values.reshape(1, -1))[0][1]
    
    return sample_idx, sample, actual_label, prediction, predicted_label, prediction_proba