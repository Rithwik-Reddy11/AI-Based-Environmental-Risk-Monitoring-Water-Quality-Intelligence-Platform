import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["Industry_Type","pH","Nitrate","Water_Temperature","Turbidity","Dissolved_Oxygen","Conductivity"]
NUMERIC = FEATURES[1:]
CATEGORICAL = [FEATURES[0]]


def build_model(random_state=42):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("preprocessor", pre), ("model", RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced"))])
