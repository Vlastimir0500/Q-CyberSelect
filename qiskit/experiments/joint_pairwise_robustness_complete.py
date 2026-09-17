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

JOINT_MATRIX = BASE / "results" / "joint_perturbation_robustness_matrix.csv"
JOINT_RECORDS = BASE / "results" / "joint_perturbation_robustness_all_pairs.csv"
JOINT_JSON = BASE / "results" / "joint_perturbation_robustness_summary.json"

SINGLE_CSV = BASE / "results" / "matched_single_perturbation_robustness.csv"
SINGLE_JSON = BASE / "results" / "matched_single_perturbation_robustness.json"

INTERACTION_CSV = BASE / "results" / "directional_pairwise_interaction_matrix.csv"
INTERACTION_JSON = BASE / "results" / "directional_pairwise_interaction_summary.json"

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
    directional = {}

    for direction_name, direction in [
        ("positive", 1.0),
        ("negative", -1.0),
    ]:
        Xp = X_val.copy()

        Xp[feature] = perturb_series(
            Xp[feature],
            feature,
            PERTURBATION_LEVEL * direction,
        )

        pred = pipeline.predict(Xp)
        perturbed_f1 = f1_score(y_val, pred)
        degradation = max(0.0, baseline_f1 - perturbed_f1)

        directional[direction_name] = {
            "f1": float(perturbed_f1),
            "degradation": float(degradation),
        }

    single_records.append({
        "feature": feature,
        "perturbation_level": PERTURBATION_LEVEL,
        "baseline_f1": float(baseline_f1),
        "positive_f1": directional["positive"]["f1"],
        "negative_f1": directional["negative"]["f1"],
        "positive_degradation": directional["positive"]["degradation"],
        "negative_degradation": directional["negative"]["degradation"],
        "mean_f1_degradation": float(
            0.5 * (
                directional["positive"]["degradation"]
                + directional["negative"]["degradation"]
            )
        ),
        "worst_f1_degradation": float(
            max(
                directional["positive"]["degradation"],
                directional["negative"]["degradation"],
            )
        ),
    })

single_df = pd.DataFrame(single_records)

single_df.to_csv(SINGLE_CSV, index=False)

with open(SINGLE_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "baseline_f1": float(baseline_f1),
            "perturbation_level": PERTURBATION_LEVEL,
            "features": FEATURES,
            "results": single_records,
        },
        f,
        indent=2,
    )

records = []

matrix = pd.DataFrame(
    np.zeros((len(FEATURES), len(FEATURES))),
    index=FEATURES,
    columns=FEATURES,
)

for feature_a, feature_b in combinations(FEATURES, 2):
    directional = {}

    for direction_name, direction in [
        ("positive", 1.0),
        ("negative", -1.0),
    ]:
        Xp = X_val.copy()

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

        directional[direction_name] = {
            "f1": float(perturbed_f1),
            "degradation": float(degradation),
        }

    mean_deg = float(
        0.5 * (
            directional["positive"]["degradation"]
            + directional["negative"]["degradation"]
        )
    )

    worst_deg = float(
        max(
            directional["positive"]["degradation"],
            directional["negative"]["degradation"],
        )
    )

    matrix.loc[feature_a, feature_b] = mean_deg
    matrix.loc[feature_b, feature_a] = mean_deg

    records.append({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "perturbation_level": PERTURBATION_LEVEL,
        "positive_f1": directional["positive"]["f1"],
        "negative_f1": directional["negative"]["f1"],
        "positive_degradation": directional["positive"]["degradation"],
        "negative_degradation": directional["negative"]["degradation"],
        "mean_f1_degradation": mean_deg,
        "worst_f1_degradation": worst_deg,
    })

joint_df = pd.DataFrame(records)

joint_df.to_csv(JOINT_RECORDS, index=False)
matrix.to_csv(JOINT_MATRIX)

with open(JOINT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "baseline_f1": float(baseline_f1),
            "perturbation_level": PERTURBATION_LEVEL,
            "features": FEATURES,
            "num_pairs": len(joint_df),
            "records": records,
        },
        f,
        indent=2,
    )

single_map = single_df.set_index("feature")

interaction_matrix = pd.DataFrame(
    np.zeros((len(FEATURES), len(FEATURES))),
    index=FEATURES,
    columns=FEATURES,
)

interaction_records = []

for row in records:
    feature_a = row["feature_a"]
    feature_b = row["feature_b"]

    a_pos = float(
        single_map.loc[feature_a, "positive_degradation"]
    )
    b_pos = float(
        single_map.loc[feature_b, "positive_degradation"]
    )

    a_neg = float(
        single_map.loc[feature_a, "negative_degradation"]
    )
    b_neg = float(
        single_map.loc[feature_b, "negative_degradation"]
    )

    pos_joint = float(row["positive_degradation"])
    neg_joint = float(row["negative_degradation"])

    pos_additive = a_pos + b_pos
    neg_additive = a_neg + b_neg

    pos_interaction = max(
        0.0,
        pos_joint - pos_additive,
    )

    neg_interaction = max(
        0.0,
        neg_joint - neg_additive,
    )

    mean_interaction = 0.5 * (
        pos_interaction + neg_interaction
    )

    interaction_matrix.loc[feature_a, feature_b] = mean_interaction
    interaction_matrix.loc[feature_b, feature_a] = mean_interaction

    interaction_records.append({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "positive_joint_degradation": pos_joint,
        "positive_additive_reference": pos_additive,
        "positive_interaction": pos_interaction,
        "negative_joint_degradation": neg_joint,
        "negative_additive_reference": neg_additive,
        "negative_interaction": neg_interaction,
        "mean_interaction": mean_interaction,
    })

interaction_df = pd.DataFrame(
    interaction_records
).sort_values(
    "mean_interaction",
    ascending=False,
)

interaction_matrix.to_csv(INTERACTION_CSV)

interaction_summary = {
    "definition": "max(0, joint directional degradation - sum of corresponding single-feature directional degradations)",
    "baseline_f1": float(baseline_f1),
    "perturbation_level": PERTURBATION_LEVEL,
    "features": FEATURES,
    "num_pairs": len(interaction_df),
    "nonzero_pairs": int(
        (interaction_df["mean_interaction"] > 0).sum()
    ),
    "mean_interaction": float(
        interaction_df["mean_interaction"].mean()
    ),
    "max_interaction": float(
        interaction_df["mean_interaction"].max()
    ),
    "top_pairs": interaction_df.head(15).to_dict(
        orient="records"
    ),
}

with open(INTERACTION_JSON, "w", encoding="utf-8") as f:
    json.dump(
        interaction_summary,
        f,
        indent=2,
    )

print(f"Baseline F1: {baseline_f1:.6f}")
print()
print("Top true non-additive pairwise interactions:")
print(
    interaction_df[
        [
            "feature_a",
            "feature_b",
            "positive_interaction",
            "negative_interaction",
            "mean_interaction",
        ]
    ].head(15).to_string(index=False)
)
print()
print(
    f"Nonzero interaction pairs: "
    f"{interaction_summary['nonzero_pairs']}/"
    f"{interaction_summary['num_pairs']}"
)
print()
print("Interaction matrix:")
print(
    interaction_matrix.round(6).to_string()
)
print()
print(f"Saved: {JOINT_RECORDS}")
print(f"Saved: {JOINT_MATRIX}")
print(f"Saved: {JOINT_JSON}")
print(f"Saved: {SINGLE_CSV}")
print(f"Saved: {SINGLE_JSON}")
print(f"Saved: {INTERACTION_CSV}")
print(f"Saved: {INTERACTION_JSON}")
