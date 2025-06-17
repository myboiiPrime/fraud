import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Real Fraud Detection Explainability Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #ff6b6b;
}
.fraud-card {
    border-left-color: #ff6b6b;
}
.safe-card {
    border-left-color: #51cf66;
}
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model_and_data():
    try:
        with open('real_fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        X_test = pd.read_csv('X_test_real.csv')
        y_test = pd.read_csv('y_test_real.csv')['is_fraud']
        
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

# Load everything
model, X_test, y_test, feature_names, lime_explainer, shap_explainer = load_model_and_data()

if model is None:
    st.stop()

# Title and introduction
st.title("🏦 Real Fraud Detection Model Explainability Dashboard")

st.markdown("""
### Comparing Explainability Approaches on Real Banking Data

This dashboard demonstrates three approaches to fraud detection using a **real-world credit card transaction dataset**:

1. **🔒 Black Box Model**: No explainability - only predictions
2. **🔍 LIME Explanations**: Local interpretable model-agnostic explanations  
3. **⚡ SHAP Explanations**: SHapley Additive exPlanations with global and local insights

**Dataset Features**: Transaction amount, customer demographics, geographic data, merchant information, and temporal patterns.
""")

# Sidebar for sample selection
st.sidebar.header("🎯 Transaction Selection")

# Dataset overview
st.sidebar.markdown(f"""
**Dataset Overview:**
- Total transactions: {len(X_test):,}
- Fraud rate: {y_test.mean():.2%}
- Fraudulent: {y_test.sum():,}
- Legitimate: {(len(y_test) - y_test.sum()):,}
""")

# Find fraud and non-fraud examples
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
        non_fraud_indices[:100],  # Limit for performance
        format_func=lambda x: f"Transaction #{x}"
    )

# Get sample data
sample = X_test.iloc[sample_idx]
actual_label = "🚨 Fraud" if y_test.iloc[sample_idx] == 1 else "✅ Legitimate"
prediction = model.predict(sample.values.reshape(1, -1))[0]
predicted_label = "🚨 Fraud" if prediction == 1 else "✅ Legitimate"
prediction_proba = model.predict_proba(sample.values.reshape(1, -1))[0][1]

# Main content area
st.header(f"📊 Transaction #{sample_idx} Analysis")

# Transaction overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Amount", 
        f"${sample['amt']:.2f}",
        delta="High" if sample['amt'] > X_test['amt'].quantile(0.9) else "Normal"
    )

with col2:
    st.metric(
        "🕐 Hour", 
        f"{int(sample['hour'])}:00",
        delta="Night" if sample['is_night'] == 1 else "Day"
    )

with col3:
    st.metric(
        "📍 Distance", 
        f"{sample['distance_from_home']:.1f} mi",
        delta="Far" if sample['distance_from_home'] > 50 else "Near"
    )

with col4:
    st.metric(
        "👤 Age", 
        f"{int(sample['age'])} years",
        delta="Senior" if sample['age'] > 65 else "Adult"
    )

# Additional transaction details
with st.expander("🔍 Detailed Transaction Information"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Geographic Info:**")
        st.write(f"City Population: {int(sample['city_pop']):,}")
        st.write(f"State: {sample['state_encoded']} (encoded)")
        st.write(f"Weekend: {'Yes' if sample['is_weekend'] == 1 else 'No'}")
    
    with col2:
        st.write("**Customer Info:**")
        st.write(f"Gender: {sample['gender_encoded']} (encoded)")
        st.write(f"Job: {sample['job_encoded']} (encoded)")
        st.write(f"High Amount: {'Yes' if sample['high_amount'] == 1 else 'No'}")
    
    with col3:
        st.write("**Temporal Info:**")
        st.write(f"Day of Week: {int(sample['day_of_week'])}")
        st.write(f"Month: {int(sample['month'])}")
        st.write(f"Night Transaction: {'Yes' if sample['is_night'] == 1 else 'No'}")

# Tabs for different explainability methods
tab1, tab2, tab3, tab4 = st.tabs([
    "🔒 Black Box Model", 
    "🔍 LIME Explanation", 
    "⚡ SHAP Explanation",
    "📈 Model Insights"
])

# Tab 1: Black Box Model
with tab1:
    st.header("🔒 Black Box Model (No Explainability)")
    
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
    
    # Probability bar
    st.subheader("Fraud Risk Assessment")
    progress_color = "red" if prediction_proba > 0.5 else "green"
    st.progress(prediction_proba)
    
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

# Tab 2: LIME Explanation
with tab2:
    st.header("🔍 LIME Explanation")
    
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
    - Feature ranges show the actual values that influenced the prediction
    """)

# Tab 3: SHAP Explanation
with tab3:
    st.header("⚡ SHAP Explanation")
    
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
    - Shows how each feature pushes the prediction from the base value (average model output)
    - Provides consistent, theoretically sound explanations
    - Can reveal complex interactions between features
    - Offers both global and local explanations
    """)

# Tab 4: Model Insights
with tab4:
    st.header("📈 Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance Metrics")
        
        # Calculate metrics on a sample for display
        sample_size = min(1000, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
        
        y_pred_sample = model.predict(X_sample)
        y_prob_sample = model.predict_proba(X_sample)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "Accuracy": accuracy_score(y_sample, y_pred_sample),
            "Precision": precision_score(y_sample, y_pred_sample),
            "Recall": recall_score(y_sample, y_pred_sample),
            "F1 Score": f1_score(y_sample, y_pred_sample)
        }
        
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.3f}")
    
    with col2:
        st.subheader("🏆 Feature Importance")
        
        # Feature importance plot
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('XGBoost Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Comparison table
    st.subheader("🔄 Explainability Methods Comparison")
    
    comparison_data = {
        "Aspect": [
            "Transparency",
            "Theoretical Foundation", 
            "Consistency",
            "Feature Interactions",
            "Computational Cost",
            "Global Insights",
            "Local Insights",
            "Regulatory Compliance"
        ],
        "Black Box": [
            "❌ None",
            "N/A",
            "N/A", 
            "❌ No",
            "✅ Low",
            "❌ No",
            "❌ No",
            "❌ Poor"
        ],
        "LIME": [
            "✅ Local",
            "Linear approximation",
            "⚠️ May vary",
            "⚠️ Limited", 
            "⚠️ Medium",
            "❌ No",
            "✅ Yes",
            "✅ Good"
        ],
        "SHAP": [
            "✅ Local & Global",
            "Game theory (Shapley)",
            "✅ Consistent",
            "✅ Yes",
            "❌ High", 
            "✅ Yes",
            "✅ Yes",
            "✅ Excellent"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.success("""
    **🎯 Key Takeaways for Real Fraud Detection:**
    
    1. **Real-world features** like transaction amount, distance from home, and timing patterns provide meaningful insights that stakeholders can understand and act upon.
    
    2. **SHAP explanations** offer the most comprehensive view, showing both how individual features contribute and how they interact with each other.
    
    3. **LIME explanations** provide intuitive local insights that are easy to communicate to non-technical stakeholders.
    
    4. **Black box models** lack the transparency required for high-stakes financial decisions and regulatory compliance.
    
    5. **Explainable AI** is essential for building trust, ensuring fairness, and meeting regulatory requirements in financial services.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏦 Real Fraud Detection Explainability Dashboard | Built with Streamlit, SHAP, and LIME</p>
</div>
""", unsafe_allow_html=True)