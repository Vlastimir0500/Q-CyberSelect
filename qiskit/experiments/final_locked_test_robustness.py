import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(".")

TRAIN_PATH = BASE / "data" / "unsw_nb15" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE / "data" / "unsw_nb15" / "UNSW_NB15_testing-set.csv"

OUT_CSV = BASE / "results" / "final_locked_test_robustness.csv"
OUT_JSON = BASE / "results" / "final_locked_test_robustness.json"

RANDOM_STATE = 42
N_ESTIMATORS = 300
PERTURBATION_LEVELS = [0.05, 0.10, 0.20]

SELECTED = {
    "All-42": None,
    "QAOA-4": [
        "sttl",
        "ct_state_ttl",
        "dload",
        "rate",
    ],
    "MI-4": [
        "dttl",
        "dbytes",
        "sttl",
        "sbytes",
    ],
    "ANOVA-4": [
        "ct_dst_sport_ltm",
        "dload",
        "ct_state_ttl",
        "sttl",
    ],
    "L1-4": [
        "dload",
        "swin",
        "proto",
        "dttl",
    ],
    "RF-4": [
        "rate",
        "sload",
        "ct_state_ttl",
        "sttl",
    ],
}

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(
    columns=["id", "attack_cat", "label"]
).copy()

y_train = train_df["label"].astype(int)

X_test = test_df.drop(
    columns=["id", "attack_cat", "label"]
).copy()

y_test = test_df["label"].astype(int)

ALL_FEATURES = X_train.columns.tolist()

SELECTED["All-42"] = ALL_FEATURES

def perturb_series(series, feature, level):
    s = pd.to_numeric(series, errors="coerce")

    if feature == "sttl":
        return (s * (1.0 + level)).clip(
            lower=1,
            upper=255,
        )

    if feature == "ct_state_ttl":
        return np.maximum(
            1,
            np.rint(s * (1.0 + level)),
        )

    return np.maximum(
        0,
        s * (1.0 + level),
    )

def build_model(feature_list):
    categorical_cols = [
        c for c in feature_list
        if not pd.api.types.is_numeric_dtype(
            X_train[c]
        )
    ]

    numeric_cols = [
        c for c in feature_list
        if c not in categorical_cols
    ]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                )
            ]),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]),
            categorical_cols,
        ),
    ])

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

rows = []

for method, feature_list in SELECTED.items():
    print(
        f"Evaluating {method} "
        f"({len(feature_list)} features)..."
    )

    model = build_model(feature_list)

    model.fit(
        X_train[feature_list],
        y_train,
    )

    clean_pred = model.predict(
        X_test[feature_list]
    )

    clean_prob = model.predict_proba(
        X_test[feature_list]
    )[:, 1]

    clean_f1 = f1_score(
        y_test,
        clean_pred,
        zero_division=0,
    )

    clean_balanced_accuracy = balanced_accuracy_score(
        y_test,
        clean_pred,
    )

    clean_roc_auc = roc_auc_score(
        y_test,
        clean_prob,
    )

    degradations = []

    for level in PERTURBATION_LEVELS:
        directional_f1 = []

        for direction_name, direction in [
            ("positive", 1.0),
            ("negative", -1.0),
        ]:
            Xp = X_test[feature_list].copy()

            perturbable = [
                f
                for f in [
                    "sbytes",
                    "sttl",
                    "dbytes",
                    "ct_state_ttl",
                    "dttl",
                    "rate",
                    "sload",
                    "dur",
                ]
                if f in feature_list
            ]

            for feature in perturbable:
                Xp[feature] = perturb_series(
                    Xp[feature],
                    feature,
                    level * direction,
                )

            pred = model.predict(Xp)

            perturbed_f1 = f1_score(
                y_test,
                pred,
                zero_division=0,
            )

            degradation = max(
                0.0,
                clean_f1 - perturbed_f1,
            )

            directional_f1.append({
                "direction": direction_name,
                "f1": float(perturbed_f1),
                "degradation": float(degradation),
            })

        mean_deg = float(
            np.mean([
                x["degradation"]
                for x in directional_f1
            ])
        )

        worst_deg = float(
            np.max([
                x["degradation"]
                for x in directional_f1
            ])
        )

        mean_perturbed_f1 = float(
            np.mean([
                x["f1"]
                for x in directional_f1
            ])
        )

        rows.append({
            "method": method,
            "feature_count": len(feature_list),
            "perturbation_level": level,
            "clean_f1": float(clean_f1),
            "clean_balanced_accuracy": float(
                clean_balanced_accuracy
            ),
            "clean_roc_auc": float(
                clean_roc_auc
            ),
            "perturbed_mean_f1": mean_perturbed_f1,
            "mean_f1_degradation": mean_deg,
            "worst_f1_degradation": worst_deg,
            "positive_f1": directional_f1[0]["f1"],
            "negative_f1": directional_f1[1]["f1"],
        })

results = pd.DataFrame(rows)

results.to_csv(
    OUT_CSV,
    index=False,
)

summary = {
    "dataset": "UNSW-NB15 official testing split",
    "evaluation_locked": True,
    "random_state": RANDOM_STATE,
    "n_estimators": N_ESTIMATORS,
    "perturbation_levels": PERTURBATION_LEVELS,
    "selected_features": SELECTED,
    "results": rows,
}

with open(
    OUT_JSON,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print()
print("FINAL LOCKED TEST ROBUSTNESS")
print()

print(
    results[
        [
            "method",
            "perturbation_level",
            "clean_f1",
            "perturbed_mean_f1",
            "mean_f1_degradation",
            "worst_f1_degradation",
        ]
    ].to_string(index=False)
)

print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")
