import pandas as pd
from sklearn.inspection import permutation_importance


def explain(model, X, y=None):
    # Model-agnostic explanation that works without requiring SHAP at runtime.
    try:
        result = permutation_importance(model, X, y if y is not None else model.predict(X), n_repeats=5, random_state=42)
        names = X.columns
        return pd.DataFrame({"feature": names, "importance": result.importances_mean}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame({"feature": X.columns, "importance": [0.0] * len(X.columns)})
