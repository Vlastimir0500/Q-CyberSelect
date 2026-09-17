import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(".")
TRAIN_PATH = BASE / "data" / "unsw_nb15" / "UNSW_NB15_training-set.csv"
OUT_CSV = BASE / "results" / "joint_perturbation_robustness_matrix.csv"
OUT_JSON = BASE / "results" / "joint_perturbation_robustness_summary.json"

FEATURES = [
    "sbytes",
    "sttl",
    "dbytes",
    "ct_state_ttl",
    "dttl",
    "rate",
    "sload",
    "dur",
]

PERTURBATION_LEVEL = 0.10
RANDOM_STATE = 42

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["id", "attack_cat", "label"]).copy()
y = df["label"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model),
])

pipeline.fit(X_train, y_train)

baseline_pred = pipeline.predict(X_val)
baseline_f1 = f1_score(y_val, baseline_pred)

def perturb_series(series, feature, level):
    s = pd.to_numeric(series, errors="coerce")

    if feature == "sttl":
        return (s * (1.0 + level)).clip(lower=1, upper=255)

    if feature in {"ct_state_ttl"}:
        return np.maximum(1, np.rint(s * (1.0 + level)))

    return np.maximum(0, s * (1.0 + level))

def evaluate_pair(X_base, feature_a, feature_b):
    directional = []

    for direction in [1.0, -1.0]:
        Xp = X_base.copy()

        Xp[feature_a] = perturb_series(
            Xp[feature_a],
            feature_a,
            PERTURBATION_LEVEL * direction,
        )

        Xp[feature_b] = perturb_series(
            Xp[feature_b],
            feature_b,
            PERTURBATION_LEVEL * direction,
        )

        pred = pipeline.predict(Xp)
        perturbed_f1 = f1_score(y_val, pred)
        degradation = max(0.0, baseline_f1 - perturbed_f1)

        directional.append({
            "direction": "positive" if direction > 0 else "negative",
            "f1": float(perturbed_f1),
            "degradation": float(degradation),
        })

    mean_degradation = float(
        np.mean([x["degradation"] for x in directional])
    )

    worst_degradation = float(
        np.max([x["degradation"] for x in directional])
    )

    return mean_degradation, worst_degradation, directional

matrix = pd.DataFrame(
    np.zeros((len(FEATURES), len(FEATURES))),
    index=FEATURES,
    columns=FEATURES,
)

records = []

for feature_a, feature_b in combinations(FEATURES, 2):
    mean_deg, worst_deg, directional = evaluate_pair(
        X_val,
        feature_a,
        feature_b,
    )

    matrix.loc[feature_a, feature_b] = mean_deg
    matrix.loc[feature_b, feature_a] = mean_deg

    records.append({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "perturbation_level": PERTURBATION_LEVEL,
        "mean_f1_degradation": mean_deg,
        "worst_f1_degradation": worst_deg,
        "positive_f1": directional[0]["f1"],
        "negative_f1": directional[1]["f1"],
        "positive_degradation": directional[0]["degradation"],
        "negative_degradation": directional[1]["degradation"],
    })

pair_df = pd.DataFrame(records).sort_values(
    "mean_f1_degradation",
    ascending=False,
)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

matrix.to_csv(OUT_CSV)

summary = {
    "dataset": "UNSW-NB15 training split",
    "validation_fraction": 0.20,
    "random_state": RANDOM_STATE,
    "model": "RandomForestClassifier",
    "n_estimators": 300,
    "features": FEATURES,
    "perturbation_level": PERTURBATION_LEVEL,
    "baseline_f1": float(baseline_f1),
    "mean_pairwise_degradation": float(
        pair_df["mean_f1_degradation"].mean()
    ),
    "max_pairwise_degradation": float(
        pair_df["mean_f1_degradation"].max()
    ),
    "min_pairwise_degradation": float(
        pair_df["mean_f1_degradation"].min()
    ),
    "top_pairs": pair_df.head(10).to_dict(orient="records"),
    "matrix_path": str(OUT_CSV),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Baseline F1: {baseline_f1:.6f}")
print()
print("Top joint perturbation pairs:")
print(
    pair_df[
        [
            "feature_a",
            "feature_b",
            "mean_f1_degradation",
            "worst_f1_degradation",
        ]
    ].head(10).to_string(index=False)
)
print()
print("Joint robustness matrix:")
print(matrix.round(6).to_string())
print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")
