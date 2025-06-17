import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Loading Real Fraud Detection Dataset")
print("===================================\n")

# Load the real dataset
try:
    # Load training data
    df_train = pd.read_csv('fraudTrain.csv')
    print(f"Training data loaded: {len(df_train)} transactions")
    
    # Load test data if available
    try:
        df_test = pd.read_csv('fraudTest.csv')
        print(f"Test data loaded: {len(df_test)} transactions")
        # Combine for initial analysis
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"Combined dataset: {len(df)} transactions")
    except:
        df = df_train.copy()
        print("Using only training data")
        
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure fraudTrain.csv is in the current directory")
    exit()

# Display basic information
print(f"\nFraud rate: {df['is_fraud'].mean():.2%}")
print(f"Total fraudulent transactions: {df['is_fraud'].sum()}")
print(f"Total legitimate transactions: {len(df) - df['is_fraud'].sum()}")

# Display dataset info
print("\nDataset columns:")
print(df.columns.tolist())

print("\nDataset info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

# Feature Engineering
print("\nFeature Engineering...")

# Convert datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

# Extract time-based features
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['month'] = df['trans_date_trans_time'].dt.month

# Calculate age
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25

# Calculate distance from home (using Haversine formula approximation)
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles"""
    from math import radians, sin, cos, sqrt, atan2
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Earth's radius in miles
    r = 3956
    return r * c

# Calculate distance from home to merchant
df['distance_from_home'] = df.apply(
    lambda row: calculate_distance(row['lat'], row['long'], row['merch_lat'], row['merch_long']), 
    axis=1
)

# Encode categorical variables
le_category = LabelEncoder()
le_gender = LabelEncoder()
le_state = LabelEncoder()
le_job = LabelEncoder()

df['category_encoded'] = le_category.fit_transform(df['category'])
df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['state_encoded'] = le_state.fit_transform(df['state'])
df['job_encoded'] = le_job.fit_transform(df['job'])

# Create additional features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
df['high_amount'] = (df['amt'] > df['amt'].quantile(0.95)).astype(int)

# Select features for modeling
feature_columns = [
    'amt', 'age', 'hour', 'day_of_week', 'month', 'distance_from_home',
    'city_pop', 'category_encoded', 'gender_encoded', 'state_encoded', 'job_encoded',
    'is_weekend', 'is_night', 'high_amount'
]

X = df[feature_columns].copy()
y = df['is_fraud'].copy()

# Handle any missing values
X = X.fillna(X.median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training fraud rate: {y_train.mean():.2%}")
print(f"Test fraud rate: {y_test.mean():.2%}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Real Fraud Detection Dataset')
plt.tight_layout()
plt.savefig('real_fraud_confusion_matrix.png')
plt.show()

# Feature importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance - Real Fraud Detection Dataset')
plt.tight_layout()
plt.savefig('real_fraud_feature_importance.png')
plt.show()

# Save the model and processed data
import pickle

with open('real_fraud_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'category': le_category,
        'gender': le_gender,
        'state': le_state,
        'job': le_job
    }, f)

# Save processed test data for explainability scripts
X_test.to_csv('X_test_real.csv', index=False)
y_test.to_csv('y_test_real.csv', index=False)

# Save feature names
with open('feature_names.txt', 'w') as f:
    for feature in feature_columns:
        f.write(f"{feature}\n")

print("\nModel and processed data saved successfully!")
print("Files created:")
print("- real_fraud_model.pkl")
print("- scaler.pkl")
print("- label_encoders.pkl")
print("- X_test_real.csv")
print("- y_test_real.csv")
print("- feature_names.txt")