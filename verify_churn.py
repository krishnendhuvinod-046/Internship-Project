import pandas as pd
import sqlite3
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import shap
import os

# Set working directory to where the csv is
# os.chdir('c:/For Maalu/project')

print("Starting verification...")

# Step 1: Load CSV
try:
    df = pd.read_csv('c:/For Maalu/project/customer_churn.csv')
    print("CSV loaded successfully.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Step 2: Rename columns to snake_case
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
print("Columns renamed.")

# Load to SQLite
try:
    conn = sqlite3.connect('c:/For Maalu/project/telecom.db')
    df.to_sql('telecom_data', conn, if_exists='replace', index=False)
    print("Data loaded to SQLite.")
except Exception as e:
    print(f"Error loading to SQLite: {e}")
    exit(1)

# Step 3: High Value At Risk Feature
query = """
SELECT 
    *,
    CASE 
        WHEN customer_value > (SELECT AVG(customer_value) FROM telecom_data) AND complains = 1 THEN 1 
        ELSE 0 
    END AS high_value_at_risk
FROM telecom_data;
"""
try:
    df_enriched = pd.read_sql(query, conn)
    print("Feature created. Shape:", df_enriched.shape)
except Exception as e:
    print(f"Error in SQL query: {e}")
    exit(1)

# Phase 2: Predictive Modeling
X = df_enriched.drop(['churn'], axis=1)
y = df_enriched['churn']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE
print("Applying SMOTE...")
try:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("SMOTE applied.")
except Exception as e:
    print(f"Error with SMOTE: {e}")
    exit(1)

# XGBoost
print("Training XGBoost...")
try:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    print("Model trained.")
except Exception as e:
    print(f"Error training XGBoost: {e}")
    exit(1)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("F1 Score:", f1_score(y_test, y_pred))

# Phase 3: Explainability
print("Running SHAP...")
try:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    print("SHAP values calculated.")
    
    # Check mean SHAP values for key features
    # shap_values.values is (n_samples, n_features)
    # We need to map feature names to indices
    feature_names = X.columns.tolist()
    
    with open('shap_stats.txt', 'w') as f:
        for feat in ['frequency_of_use', 'call_failure', 'complains']:
            if feat in feature_names:
                idx = feature_names.index(feat)
                vals = shap_values.values[:, idx]
                f.write(f"Mean SHAP value for {feat}: {vals.mean()}\n")
                # Correlation with Churn (approximate via mean value for churners vs non-churners in test set if we had them separated, 
                # but here just global impact or correlation of feature with SHAP value)
                import numpy as np
                corr = np.corrcoef(X_test[feat], vals)[0, 1]
                f.write(f"Correlation between {feat} and its SHAP value: {corr}\n")
            
except Exception as e:
    print(f"Error with SHAP: {e}")
    with open('shap_stats.txt', 'w') as f:
        f.write(f"Error: {e}")

print("Verification complete.")

print("Verification complete.")
