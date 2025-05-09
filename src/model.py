# model.py
import warnings

warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_preprocessed_data
from data_loader import dataset_raw

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from config import OUTPUT_DIR

# ---- Step 1: Preprocess Data ----
(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    X_resampled,
    y_resampled,
    all_features,
) = get_preprocessed_data(dataset_raw)

# ---- Step 2: Train Models ----

# Logistic Regression
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train_processed, y_train)
y_pred_lr = lr.predict(X_test_processed)
y_proba_lr = lr.predict_proba(X_test_processed)[:, 1]

# Random Forest
rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)
y_pred_rf = rf.predict(X_test_processed)
y_proba_rf = rf.predict_proba(X_test_processed)[:, 1]

# XGBoost
xgb = XGBClassifier(
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_resampled, y_resampled)
y_pred_xgb = xgb.predict(X_test_processed)
y_proba_xgb = xgb.predict_proba(X_test_processed)[:, 1]


# ---- Step 3: Evaluate Models ----
def evaluate_model(y_true, y_pred, y_proba, model_name, feature_importances=None):
    print(f"\n====================== {model_name} ======================")
    print(classification_report(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_proba))

    if feature_importances is not None:
        print("\nTop 10 Important Features:")
        print(feature_importances.sort_values(ascending=False).head(10))


def get_feature_importance(model, model_name, all_features):
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            model.feature_importances_, index=all_features
        )  # Tree-based models like RandomForest and XGBoost use this attribute
    elif hasattr(model, "coef_"):
        return pd.Series(
            model.coef_[0], index=all_features
        )  # Linear models like LogisticRegression use coefficients
    else:
        return None  # Model doesn't provide feature importances


# ---- Step 4: Run Models ----
evaluate_model(
    y_test,
    y_pred_lr,
    y_proba_lr,
    "Logistic Regression",
    get_feature_importance(lr, "lr", all_features),
)
evaluate_model(
    y_test,
    y_pred_rf,
    y_proba_rf,
    "Random Forest",
    get_feature_importance(rf, "rf", all_features),
)
evaluate_model(
    y_test,
    y_pred_xgb,
    y_proba_xgb,
    "XGBoost",
    get_feature_importance(xgb, "xgb", all_features),
)

# ---- Step 5: ROC Curve Comparison ----
plt.figure(figsize=(8, 6))
for probs, name in zip(
    [y_proba_lr, y_proba_rf, y_proba_xgb],
    ["Logistic Regression", "Random Forest", "XGBoost"],
):
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}roc_curve_comparison.png")
plt.show()

print("===========================================")
print(f'Chart Saved: "{OUTPUT_DIR}"')
