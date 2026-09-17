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
JOINT_PATH = BASE / "results" / "joint_perturbation_robustness_matrix.csv"
SINGLE_CSV = BASE / "results" / "matched_single_perturbation_robustness.csv"
SINGLE_JSON = BASE / "results" / "matched_single_perturbation_robustness.json"
SYNERGY_CSV = BASE / "results" / "pairwise_robustness_synergy_matrix.csv"
SYNERGY_JSON = BASE / "results" / "pairwise_robustness_synergy_summary.json"

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

categorical_cols = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numeric_cols = [
    c for c in X_train.columns
    if c not in categorical_cols
]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

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

    if feature == "ct_state_ttl":
        return np.maximum(1, np.rint(s * (1.0 + level)))

    return np.maximum(0, s * (1.0 + level))

single_records = []

for feature in FEATURES:
    directional = []

    for direction in [1.0, -1.0]:
        Xp = X_val.copy()

        Xp[feature] = perturb_series(
            Xp[feature],
            feature,
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

    mean_deg = float(
        np.mean([x["degradation"] for x in directional])
    )

    worst_deg = float(
        np.max([x["degradation"] for x in directional])
    )

    single_records.append({
        "feature": feature,
        "perturbation_level": PERTURBATION_LEVEL,
        "baseline_f1": float(baseline_f1),
        "positive_f1": directional[0]["f1"],
        "negative_f1": directional[1]["f1"],
        "positive_degradation": directional[0]["degradation"],
        "negative_degradation": directional[1]["degradation"],
        "mean_f1_degradation": mean_deg,
        "worst_f1_degradation": worst_deg,
    })

single_df = pd.DataFrame(single_records)

single_map = dict(
    zip(
        single_df["feature"],
        single_df["mean_f1_degradation"],
    )
)

joint_matrix = pd.read_csv(JOINT_PATH, index_col=0)

synergy = pd.DataFrame(
    np.zeros((len(FEATURES), len(FEATURES))),
    index=FEATURES,
    columns=FEATURES,
)

records = []

for feature_a, feature_b in combinations(FEATURES, 2):
    joint_deg = float(
        joint_matrix.loc[feature_a, feature_b]
    )

    single_a = float(single_map[feature_a])
    single_b = float(single_map[feature_b])

    additive_reference = 0.5 * (single_a + single_b)

    interaction = max(
        0.0,
        joint_deg - additive_reference,
    )

    synergy.loc[feature_a, feature_b] = interaction
    synergy.loc[feature_b, feature_a] = interaction

    records.append({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "joint_degradation": joint_deg,
        "single_a": single_a,
        "single_b": single_b,
        "additive_reference": additive_reference,
        "pairwise_synergy": interaction,
    })

records_df = pd.DataFrame(records).sort_values(
    "pairwise_synergy",
    ascending=False,
)

single_df.to_csv(SINGLE_CSV, index=False)
synergy.to_csv(SYNERGY_CSV)

summary = {
    "dataset": "UNSW-NB15 training split",
    "validation_fraction": 0.20,
    "random_state": RANDOM_STATE,
    "model": "RandomForestClassifier",
    "n_estimators": 300,
    "perturbation_level": PERTURBATION_LEVEL,
    "features": FEATURES,
    "baseline_f1": float(baseline_f1),
    "mean_single_degradation": float(
        single_df["mean_f1_degradation"].mean()
    ),
    "max_single_degradation": float(
        single_df["mean_f1_degradation"].max()
    ),
    "mean_pairwise_synergy": float(
        records_df["pairwise_synergy"].mean()
    ),
    "max_pairwise_synergy": float(
        records_df["pairwise_synergy"].max()
    ),
    "nonzero_synergy_pairs": int(
        (records_df["pairwise_synergy"] > 0).sum()
    ),
    "num_pairs": int(len(records_df)),
    "top_synergy_pairs": records_df.head(15).to_dict(
        orient="records"
    ),
    "single_results_path": str(SINGLE_CSV),
    "synergy_matrix_path": str(SYNERGY_CSV),
}

with open(SINGLE_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "baseline_f1": float(baseline_f1),
            "features": FEATURES,
            "perturbation_level": PERTURBATION_LEVEL,
            "results": single_records,
        },
        f,
        indent=2,
    )

with open(SYNERGY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Baseline F1: {baseline_f1:.6f}")
print()
print("Matched single-feature robustness:")
print(
    single_df[
        [
            "feature",
            "mean_f1_degradation",
            "worst_f1_degradation",
        ]
    ].sort_values(
        "mean_f1_degradation",
        ascending=False,
    ).to_string(index=False)
)

print()
print("Top pairwise synergy:")
print(
    records_df[
        [
            "feature_a",
            "feature_b",
            "joint_degradation",
            "additive_reference",
            "pairwise_synergy",
        ]
    ].head(15).to_string(index=False)
)

print()
print(
    f"Nonzero synergy pairs: "
    f"{summary['nonzero_synergy_pairs']}/{summary['num_pairs']}"
)

print()
print("Pairwise synergy matrix:")
print(synergy.round(6).to_string())

print()
print(f"Saved: {SINGLE_CSV}")
print(f"Saved: {SINGLE_JSON}")
print(f"Saved: {SYNERGY_CSV}")
print(f"Saved: {SYNERGY_JSON}")
