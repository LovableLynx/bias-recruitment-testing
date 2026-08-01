import joblib
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def load_model(path):
    return joblib.load(path)

def compute_fairness_metrics(model, X_test, y_true, sensitive_features):
    y_pred = model.predict(X_test)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return {"demographic_parity_difference": dp_diff, "equalized_odds_difference": eo_diff}

import joblib
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_model(path):
    return joblib.load(path)

def compute_fairness_metrics(model, X_test, y_true, sensitive_features):
    y_pred = model.predict(X_test)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return {"demographic_parity_difference": dp_diff, "equalized_odds_difference": eo_diff}

def compute_performance_metrics(model, X_test, y_true):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba)
    }