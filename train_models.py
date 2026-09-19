"""Train and compare baseline models. Run: python train_models.py"""
import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from src.data_pipeline import load_data
from src.model import build_model, FEATURES

os.makedirs("models", exist_ok=True)
df = load_data()
X, y = df[FEATURES], df["Pollution_Flag"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
models = {"Random Forest": build_model(), "ANN (MLP)": build_model()}
results = {}
for name, model in models.items():
    if name.startswith("ANN"):
        # A compact ANN-style model is kept as an optional benchmark; RF is the deployable model.
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        pre = ColumnTransformer([("num", Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]), FEATURES[1:]), ("cat",OneHotEncoder(handle_unknown="ignore"), [FEATURES[0]])])
        model = Pipeline([("pre",pre),("mlp",MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, random_state=42))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else pred
    results[name] = {"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,prob)}
    if name == "Random Forest":
        import joblib; joblib.dump(model,"models/random_forest_pipeline.pkl")
with open("models/metrics.json","w") as f: json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
