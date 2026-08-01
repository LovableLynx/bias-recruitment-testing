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