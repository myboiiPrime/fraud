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

# Add model performance context
def add_model_performance_context(model, X_test, y_test):
    """Add model performance metrics for context"""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Model Performance")
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display metrics with explanations
    st.sidebar.metric("Precision", f"{precision:.3f}", 
                     help="Of predicted frauds, how many were actually fraud?")
    st.sidebar.metric("Recall", f"{recall:.3f}", 
                     help="Of actual frauds, how many did we catch?")
    st.sidebar.metric("F1-Score", f"{f1:.3f}", 
                     help="Harmonic mean of precision and recall")
    st.sidebar.metric("AUC-ROC", f"{auc:.3f}", 
                     help="Area under ROC curve (0.5=random, 1.0=perfect)")

# Add this function to shared_utils.py
def add_statistical_context(sample, X_test, feature_names):
    """Add statistical context for better understanding"""
    st.subheader("📈 Statistical Context")
    
    # Feature percentiles
    percentiles_data = []
    for feature in feature_names:
        feature_value = sample[feature]
        percentile = (X_test[feature] <= feature_value).mean() * 100
        
        if percentile >= 95:
            status = "🔴 Very High (Top 5%)"
        elif percentile >= 75:
            status = "🟡 High (Top 25%)"
        elif percentile >= 25:
            status = "🟢 Normal (25-75%)"
        else:
            status = "🔵 Low (Bottom 25%)"
            
        percentiles_data.append({
            "Feature": feature,
            "Current Value": f"{feature_value:.4f}",
            "Percentile": f"{percentile:.1f}%",
            "Status": status,
            "Dataset Mean": f"{X_test[feature].mean():.4f}",
            "Dataset Std": f"{X_test[feature].std():.4f}"
        })
    
    percentiles_df = pd.DataFrame(percentiles_data)
    st.dataframe(percentiles_df, use_container_width=True)
    
    return percentiles_df


def create_executive_header(sample_idx, sample, actual_label, predicted_label, prediction_proba):
    """Create executive-style header with risk alert banner"""
    import streamlit as st
    
    # Dynamic risk alert banner
    if prediction_proba > 0.8:
        st.error("🚨 **HIGH FRAUD RISK DETECTED** - Immediate attention required")
        risk_color = "#ff4444"
        risk_text = "HIGH RISK"
    elif prediction_proba > 0.5:
        st.warning("⚠️ **MODERATE FRAUD RISK** - Additional verification recommended")
        risk_color = "#ffaa00"
        risk_text = "MODERATE RISK"
    else:
        st.success("✅ **LOW FRAUD RISK** - Transaction appears legitimate")
        risk_color = "#44ff44"
        risk_text = "LOW RISK"
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Executive summary metrics - Split into two rows for better spacing
    st.markdown("### 📊 Transaction Overview")
    
    # First row - Main risk info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color}22, {risk_color}44); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; 
                    border-left: 4px solid {risk_color}; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: {risk_color};">🎯 RISK ASSESSMENT</h3>
            <h2 style="margin: 0.5rem 0; color: {risk_color};">{risk_text}</h2>
            <p style="margin: 0; font-size: 1.5em; font-weight: bold;">{prediction_proba:.0%} Fraud Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = max(prediction_proba, 1-prediction_proba)
        st.metric("🎯 Model Confidence", f"{confidence:.0%}", 
                 delta="High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low")
    
    with col3:
        st.metric("📋 Transaction ID", f"#{sample_idx}")
        st.metric("🏷️ Actual Status", actual_label.replace('🚨 ', '').replace('✅ ', ''))
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second row - Transaction details
    st.markdown("### 💳 Transaction Details")
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        amount_risk = "High" if sample['amt'] > 1000 else "Normal"
        st.metric("💰 Amount", f"${sample['amt']:.2f}", delta=amount_risk)
    
    with col5:
        time_risk = "Night" if sample['is_night'] == 1 else "Day"
        st.metric("🕐 Time", f"{int(sample['hour'])}:00", delta=time_risk)
    
    with col6:
        location_risk = "Far" if sample['distance_from_home'] > 50 else "Near"
        st.metric("📍 Distance", f"{sample['distance_from_home']:.1f} mi", delta=location_risk)
    
    with col7:
        age_info = f"{sample['age']:.0f} years" if 'age' in sample else "N/A"
        st.metric("👤 Customer Age", age_info)
    
    # Add final spacing
    st.markdown("<br>", unsafe_allow_html=True)

def explain_algorithm_simply(method_name):
    """Provide customer-friendly algorithm explanations"""
    import streamlit as st
    
    explanations = {
        "fraud_detection": {
            "title": "🤖 How Our Fraud Detection Works",
            "simple": """
            **Think of it like a smart security guard:**
            
            1. **🔍 Pattern Recognition**: Our system has learned from millions of past transactions
            2. **📊 Risk Scoring**: It looks at 14 different factors about your transaction
            3. **⚡ Instant Decision**: In milliseconds, it calculates a risk score (0-100%)
            4. **🎯 Smart Alerts**: Only flags transactions that look suspicious
            
            **What makes a transaction suspicious?**
            - Unusual spending amounts for you
            - Purchases far from your usual locations
            - Shopping at odd hours
            - Merchant types you don't normally use
            """
        },
        "lime": {
            "title": "🔍 LIME: Why This Decision?",
            "simple": """
            **Like asking 'What if?' questions:**
            
            1. **🎲 Testing Scenarios**: LIME changes small details about your transaction
            2. **📈 Measuring Impact**: It sees how each change affects the fraud score
            3. **🎯 Finding Key Factors**: Identifies which details matter most
            4. **📋 Simple Report**: Shows you exactly why this decision was made
            
            **Example**: "If the amount was $50 instead of $500, the fraud score would drop by 30%"
            
            **Perfect for**: Understanding individual transaction decisions
            """
        },
        "shap": {
            "title": "⚡ SHAP: The Complete Picture",
            "simple": """
            **Like a detailed financial audit:**
            
            1. **🏦 Baseline**: Starts with the average fraud risk (what's normal)
            2. **➕➖ Factor Analysis**: Each transaction detail adds or subtracts risk
            3. **🧮 Mathematical Precision**: Uses game theory for fair attribution
            4. **📊 Visual Breakdown**: Shows exactly how we reached the final score
            
            **Example**: "Base risk: 2% + High amount: +15% + Night time: +8% = 25% total risk"
            
            **Perfect for**: Detailed analysis and compliance reporting
            """
        },
        "black_box": {
            "title": "🔒 Black Box: Just the Answer",
            "simple": """
            **Like a simple yes/no decision:**
            
            1. **⚡ Instant Result**: Just tells you if it's fraud or not
            2. **🎯 High Accuracy**: Very reliable, but no explanation
            3. **🚀 Fast Processing**: Perfect for high-volume transactions
            4. **🔒 No Details**: Protects the algorithm from being gamed
            
            **When to use**: When you just need a quick decision
            **When not to use**: When you need to explain the decision to customers
            """
        }
    }
    
    if method_name in explanations:
        exp = explanations[method_name]
        st.markdown(f"### {exp['title']}")
        st.markdown(exp['simple'])

def create_action_panel(sample_idx, prediction_proba, actual_label, predicted_label, context="main"):
    """Create executive action panel with unique keys for each context"""
    import streamlit as st
    
    st.markdown("### 🎯 Quick Actions")
    
    # Action buttons with unique keys
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve Transaction", type="primary", use_container_width=True, key=f"approve_{context}_{sample_idx}"):
            st.success("Transaction approved for processing")
    
    with col2:
        if st.button("🔍 Investigate Further", use_container_width=True, key=f"investigate_{context}_{sample_idx}"):
            st.info("Flagged for manual review")
    
    # Investigation checklist
    st.markdown("### 📋 Investigation Checklist")
    
    checks = [
        "Verify customer identity",
        "Check recent transaction history", 
        "Confirm merchant legitimacy",
        "Review location patterns",
        "Contact customer if needed"
    ]
    
    for i, check in enumerate(checks):
        st.checkbox(check, key=f"check_{context}_{sample_idx}_{i}")
    
    # Export options
    st.markdown("### 📤 Export Options")
    
    if st.button("📊 Generate Report", use_container_width=True, key=f"report_{context}_{sample_idx}"):
        st.download_button(
            "📥 Download Analysis",
            data=f"Transaction {sample_idx} Analysis\nRisk: {prediction_proba:.1%}\nActual: {actual_label}\nPredicted: {predicted_label}",
            file_name=f"transaction_{sample_idx}_analysis.txt",
            mime="text/plain",
            key=f"download_{context}_{sample_idx}"
        )