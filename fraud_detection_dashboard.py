import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from shared_utils import load_model_and_data, get_sample_data, create_executive_header, create_action_panel, explain_algorithm_simply
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🏦 Fraud Detection - Main Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3a8a, #3b82f6);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.explanation-tab {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
}
.risk-high { border-left-color: #ef4444; background: #fef2f2; }
.risk-medium { border-left-color: #f59e0b; background: #fffbeb; }
.risk-low { border-left-color: #10b981; background: #f0fdf4; }
</style>
""", unsafe_allow_html=True)

# Load data
model, X_test, y_test, feature_names, lime_explainer, shap_explainer = load_model_and_data()

if model is None:
    st.stop()

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🏦 Fraud Detection Analytics Hub</h1>
    <p>Comprehensive AI-Powered Transaction Analysis with Multiple Explanation Methods</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Enhanced with model performance
st.sidebar.header("🎯 Control Panel")

# Add fraud percentage charts to sidebar
y_pred = model.predict(X_test)
actual_fraud_rate = (y_test == 1).mean() * 100
predicted_fraud_rate = (y_pred == 1).mean() * 100

# Create circle charts for actual fraud percentage
fig_actual, ax_actual = plt.subplots(figsize=(3, 3))
wedges, texts, autotexts = ax_actual.pie(
    [actual_fraud_rate, 100 - actual_fraud_rate],
    labels=['Fraud', 'Not Fraud'],
    colors=['#ff4444', '#44ff44'],
    autopct='%1.1f%%',
    startangle=90,
    explode=(0.05, 0),
    shadow=True,
    textprops={'fontsize': 9, 'weight': 'bold'}
)
ax_actual.set_title('Actual Fraud Distribution\n(Ground Truth)', 
                   fontsize=11, fontweight='bold', pad=15)
st.sidebar.pyplot(fig_actual)

# Add summary metrics
st.sidebar.metric("Total Transactions", f"{len(y_test):,}")
st.sidebar.metric("Actual Fraud Rate", f"{actual_fraud_rate:.2f}%")
st.sidebar.metric("Model Accuracy", f"{((y_test == y_pred).mean() * 100):.2f}%")

# Sample selection
sample_idx, sample, actual_label, prediction, predicted_label, prediction_proba = get_sample_data(model, X_test, y_test)

# Executive Header
create_executive_header(sample_idx, sample, actual_label, predicted_label, prediction_proba)

st.markdown("---")

# Main Analysis Area with Explanation Method Tabs
st.header(f"🔍 Multi-Method Analysis for Transaction #{sample_idx}")

# Create tabs for different explanation methods
tab1, tab2, tab3, tab4 = st.tabs(["🔒 Black Box Model", "🔍 LIME Explanation", "⚡ SHAP Analysis", "📊 Comparison"])

# Tab 1: Black Box Model
with tab1:
    col_main, col_action = st.columns([2, 1])
    
    with col_main:
        st.subheader("🔒 Black Box Model Analysis")
        
        # Risk assessment with enhanced visualization
        st.markdown("#### 🎯 Risk Assessment")
        
        # Enhanced risk meter
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Create risk meter
        risk_zones = [0.3, 0.7, 1.0]
        colors = ['#44ff44', '#ffaa00', '#ff4444']
        labels = ['LOW RISK', 'MODERATE RISK', 'HIGH RISK']
        
        for i, (zone, color, label) in enumerate(zip(risk_zones, colors, labels)):
            start = risk_zones[i-1] if i > 0 else 0
            ax.barh(0, zone - start, left=start, height=0.5, color=color, alpha=0.7)
            ax.text((start + zone) / 2, 0, label, ha='center', va='center', fontweight='bold')
        
        # Add current risk indicator
        ax.axvline(x=prediction_proba, color='black', linewidth=3, linestyle='--')
        ax.text(prediction_proba, 0.3, f'{prediction_proba:.1%}', ha='center', fontweight='bold', fontsize=12)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, 0.8)
        ax.set_xlabel('Fraud Risk Probability', fontweight='bold')
        ax.set_title('Black Box Risk Assessment', fontweight='bold', fontsize=14)
        ax.set_yticks([])
        
        st.pyplot(fig)
        
        # Confidence calculation
        confidence = abs(prediction_proba - 0.5) * 2
        uncertainty = 1 - confidence
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{prediction_proba:.1%}")
        with col2:
            st.metric("Prediction Confidence", f"{confidence:.1%}")
        with col3:
            st.metric("Uncertainty", f"{uncertainty:.1%}")
        
        # Black Box limitations
        st.warning("""
        **⚠️ Black Box Limitations:**
        - No transparency into decision-making process
        - Cannot identify which factors influenced the decision
        - Difficult to build trust with stakeholders
        - For explainable analysis, use LIME or SHAP tabs
        """)
    
    with col_action:
        create_action_panel(sample_idx, prediction_proba, actual_label, predicted_label, "blackbox")

# Tab 2: LIME Explanation
with tab2:
    col_main, col_action = st.columns([2, 1])
    
    with col_main:
        st.subheader("🔍 LIME Local Explanation")
        
        with st.spinner("Generating LIME explanation..."):
            # Get LIME explanation
            num_samples = 5000
            exp = lime_explainer.explain_instance(
                data_row=sample.values, 
                predict_fn=model.predict_proba,
                num_features=len(feature_names),
                num_samples=num_samples
            )
            
            # Create LIME plot
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(12, 8)
            st.pyplot(fig)
        
        # Feature contributions table
        st.markdown("#### 📋 Feature Contributions Analysis")
        
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
        
        st.info(f"""
        **🔍 LIME Process for Transaction #{sample_idx}:**
        - Created {num_samples} synthetic samples around this transaction
        - Trained local linear model with accuracy: {exp.score:.3f}
        - Shows which features contributed positively or negatively
        """)
    
    with col_action:
        create_action_panel(sample_idx, prediction_proba, actual_label, predicted_label, "lime")

# Tab 3: SHAP Analysis
with tab3:
    col_main, col_action = st.columns([2, 1])
    
    with col_main:
        st.subheader("⚡ SHAP Value Analysis")
        
        with st.spinner("Generating SHAP explanation..."):
            # Get SHAP values
            shap_values = shap_explainer(sample.values.reshape(1, -1))
            
            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        
        # SHAP summary
        st.markdown("#### 📊 SHAP Value Breakdown")
        
        # Get base value and prediction
        base_value = shap_explainer.expected_value[1] if hasattr(shap_explainer.expected_value, '__len__') else shap_explainer.expected_value
        shap_sum = base_value + shap_values.values[0].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Value", f"{base_value:.4f}")
        with col2:
            st.metric("SHAP Sum", f"{shap_sum:.4f}")
        with col3:
            st.metric("Prediction", f"{prediction_proba:.1%}")
        
        # Top contributing features
        feature_importance = list(zip(feature_names, shap_values.values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        st.markdown("**Top 5 Contributing Features:**")
        for i, (feature, value) in enumerate(feature_importance[:5]):
            direction = "↗️ Increases" if value > 0 else "↘️ Decreases"
            st.write(f"{i+1}. **{feature}**: {value:.4f} ({direction} fraud risk)")
        
        st.info("""
        **⚡ SHAP Explanation:**
        - Shows exact contribution of each feature to the final prediction
        - Base value represents average model output
        - Each feature adds or subtracts from the base value
        - Sum of all contributions equals the final prediction
        """)
    
    with col_action:
        create_action_panel(sample_idx, prediction_proba, actual_label, predicted_label, "shap")

# Tab 4: Comparison
with tab4:
    st.subheader("📊 Method Comparison & Summary")
    
    # Create comparison table
    comparison_data = {
        "Method": ["🔒 Black Box", "🔍 LIME", "⚡ SHAP"],
        "Prediction": [f"{prediction_proba:.1%}", f"{prediction_proba:.1%}", f"{prediction_proba:.1%}"],
        "Explainability": ["None", "Local", "Global + Local"],
        "Speed": ["Very Fast", "Medium", "Medium"],
        "Trust Level": ["Low", "Medium", "High"],
        "Use Case": ["Quick decisions", "Individual explanations", "Detailed analysis"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Method recommendations
    st.markdown("#### 🎯 When to Use Each Method")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔒 Black Box Model**
        - ✅ High-volume processing
        - ✅ Speed is critical
        - ❌ Regulatory compliance
        - ❌ Customer explanations
        """)
    
    with col2:
        st.markdown("""
        **🔍 LIME Explanations**
        - ✅ Individual case analysis
        - ✅ Customer-facing explanations
        - ✅ Dispute resolution
        - ❌ Global model understanding
        """)
    
    with col3:
        st.markdown("""
        **⚡ SHAP Analysis**
        - ✅ Regulatory reporting
        - ✅ Model debugging
        - ✅ Feature importance
        - ✅ Comprehensive analysis
        """)
    
    # Overall recommendation
    risk_level = "HIGH" if prediction_proba > 0.7 else "MEDIUM" if prediction_proba > 0.3 else "LOW"
    
    if risk_level == "HIGH":
        st.error(f"🚨 **HIGH RISK TRANSACTION** - Recommend using SHAP for detailed analysis and documentation")
    elif risk_level == "MEDIUM":
        st.warning(f"⚠️ **MEDIUM RISK TRANSACTION** - LIME explanation sufficient for review")
    else:
        st.success(f"✅ **LOW RISK TRANSACTION** - Black box prediction acceptable")

st.markdown("---")

# Enhanced Algorithm Explanations
explain_algorithm_simply("fraud_detection")

# Detailed Analysis Sections
with st.expander("🔍 Complete Transaction Profile"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**💰 Financial**")
        st.write(f"Amount: ${sample['amt']:.2f}")
        st.write(f"High Amount: {'Yes' if sample['high_amount'] == 1 else 'No'}")
        st.write(f"Category: {sample['category_encoded']} (encoded)")
    
    with col2:
        st.markdown("**📍 Geographic**")
        st.write(f"Distance: {sample['distance_from_home']:.1f} mi")
        st.write(f"City Pop: {int(sample['city_pop']):,}")
        st.write(f"State: {sample['state_encoded']} (encoded)")
    
    with col3:
        st.markdown("**⏰ Temporal**")
        st.write(f"Hour: {int(sample['hour'])}:00")
        st.write(f"Day of Week: {int(sample['day_of_week'])}")
        st.write(f"Month: {int(sample['month'])}")
        st.write(f"Weekend: {'Yes' if sample['is_weekend'] == 1 else 'No'}")
        st.write(f"Night: {'Yes' if sample['is_night'] == 1 else 'No'}")
    
    with col4:
        st.markdown("**👤 Customer**")
        st.write(f"Age: {sample['age']:.0f} years")
        st.write(f"Gender: {sample['gender_encoded']} (encoded)")
        st.write(f"Job: {sample['job_encoded']} (encoded)")

with st.expander("📈 Statistical Context & Benchmarks"):
    st.markdown("### How This Transaction Compares to Dataset")
    
    # Enhanced statistical context
    metrics = ['amt', 'distance_from_home', 'age', 'city_pop', 'hour']
    
    for metric in metrics:
        if metric in sample:
            value = sample[metric]
            percentile = (X_test[metric] <= value).mean() * 100
            mean_val = X_test[metric].mean()
            std_val = X_test[metric].std()
            
            if percentile >= 95:
                status = "🔴 Very High (Top 5%)"
                risk = "High"
            elif percentile >= 75:
                status = "🟡 Above Average (Top 25%)"
                risk = "Medium"
            elif percentile <= 5:
                status = "🔵 Very Low (Bottom 5%)"
                risk = "Low"
            else:
                status = "🟢 Normal Range"
                risk = "Low"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{metric.title()}", f"{value:.2f}")
            with col2:
                st.metric("Percentile", f"{percentile:.1f}%")
            with col3:
                st.metric("Dataset Avg", f"{mean_val:.2f}")
            with col4:
                st.write(f"**Status:** {status}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🏦 Fraud Detection Analytics Hub</h4>
    <p>Comprehensive AI-Powered Transaction Analysis | Built with Streamlit, SHAP, LIME & Advanced ML</p>
    <p><em>Combining Black Box Efficiency, LIME Interpretability, and SHAP Precision</em></p>
</div>
""", unsafe_allow_html=True)