import joblib
import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias

def run_aequitas_audit(model_path, X_test, y_true, sensitive_col_idx, attribute_name, group_map, ref_group):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    df = pd.DataFrame({
        'score': y_pred,
        'label_value': y_true,
        attribute_name: X_test[:, sensitive_col_idx].astype(int)
    })
    df[attribute_name] = df[attribute_name].map(group_map)

    g = Group()
    xtab, _ = g.get_crosstabs(df)
    b = Bias()
    bdf = b.get_disparity_predefined_groups(xtab, original_df=df, ref_groups_dict={attribute_name: ref_group})

    return bdf[['attribute_name', 'attribute_value', 'fpr_disparity', 'fnr_disparity', 'ppr_disparity']]