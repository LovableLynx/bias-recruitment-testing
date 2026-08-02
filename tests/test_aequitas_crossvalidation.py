from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from aequitas_checks import run_aequitas_audit

FPR_FNR_THRESHOLD = 2.0  # disparity ratio ceiling — adjust as needed

def test_aequitas_flags_gender_bias(test_data):
    X_test, y_true = test_data
    result = run_aequitas_audit(
        PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl',
        X_test, y_true, sensitive_col_idx=1, attribute_name='gender',
        group_map={0: 'Male', 1: 'Female'}, ref_group='Male'
    )
    female_row = result[result['attribute_value'] == 'Female'].iloc[0]
    assert female_row['fnr_disparity'] > FPR_FNR_THRESHOLD, \
        "Expected Aequitas to detect FNR disparity on known-biased gender model"

def test_aequitas_confirms_fair_model_gender(test_data):
    X_test, y_true = test_data
    result = run_aequitas_audit(
        PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl',
        X_test, y_true, sensitive_col_idx=1, attribute_name='gender',
        group_map={0: 'Male', 1: 'Female'}, ref_group='Male'
    )
    female_row = result[result['attribute_value'] == 'Female'].iloc[0]
    assert female_row['fnr_disparity'] < FPR_FNR_THRESHOLD, \
        "Expected Aequitas to confirm fair model has low FNR disparity"