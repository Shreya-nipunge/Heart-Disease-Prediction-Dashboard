# heartdiseaseprediction_model.py
# Train models on local heart.csv and save the best pipeline as heart_model.pkl
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import joblib

print("Starting training script...")

# Try to import xgboost; if not available, we'll skip it
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("xgboost not available. XGBoost model will be skipped.")

# 1. Load dataset from local file 'heart.csv'
DATA_FILENAME = "heart.csv"
if not os.path.exists(DATA_FILENAME):
    raise FileNotFoundError(f"'{DATA_FILENAME}' not found in current directory. Please place heart.csv here.")

df = pd.read_csv(DATA_FILENAME)
print(f"Loaded {DATA_FILENAME} — shape: {df.shape}")

# Ensure target column is named 'target'
if 'target' not in df.columns:
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
    print("Renamed last column to 'target'")

# 2. Prepare X, y
X = df.drop(columns=['target'])
y = df['target']

# 3. Detect categorical columns
cat_cols = [c for c in X.columns if X[c].nunique() <= 6]
num_cols = [c for c in X.columns if c not in cat_cols]

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, stratify=y, random_state=42)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# 5. Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
], remainder='drop')

# 6. Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
}
if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

results = []

# 7. Cross-validation + fit
for name, estimator in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('clf', estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        mean_cv = scores.mean()
    except Exception as e:
        mean_cv = None
        print(f"Warning: cross_val_score failed for {name}: {e}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} | CV acc: {mean_cv if mean_cv else 'N/A'} | Test acc: {acc:.4f}")
    results.append((name, acc, pipe))

# 8. Select best model
best_name, best_acc, best_pipe = max(results, key=lambda x: x[1])
print(f"\nBest model: {best_name} with test accuracy {best_acc:.4f}")

# 9. Evaluate
y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("Saved ROC curve to 'roc_curve.png'.")

# 10. Save model
joblib.dump(best_pipe, "heart_model.pkl")
print("Saved best model pipeline to 'heart_model.pkl'")

print("\nTraining script completed successfully.")