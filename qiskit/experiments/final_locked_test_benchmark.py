import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(".")

TRAIN_PATH = BASE / "data" / "unsw_nb15" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE / "data" / "unsw_nb15" / "UNSW_NB15_testing-set.csv"

OUT_CSV = BASE / "results" / "final_locked_test_benchmark.csv"
OUT_JSON = BASE / "results" / "final_locked_test_benchmark.json"
OUT_FEATURES = BASE / "results" / "final_locked_selected_features.csv"

RANDOM_STATE = 42
ALL_FEATURE_COUNT = 42
N_ESTIMATORS = 300

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

all_features = X_train.columns.tolist()

selected_features = {
    "All-42": all_features,
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

def build_classifier(feature_list):
    categorical_cols = [
        c for c in feature_list
        if not pd.api.types.is_numeric_dtype(X_train[c])
    ]

    numeric_cols = [
        c for c in feature_list
        if c not in categorical_cols
    ]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
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

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline([
        ("prep", preprocessor),
        ("clf", classifier),
    ])

records = []
selection_rows = []

for method, feature_list in selected_features.items():
    print(f"Running {method} with {len(feature_list)} features...")

    model = build_classifier(feature_list)

    model.fit(
        X_train[feature_list],
        y_train,
    )

    pred = model.predict(
        X_test[feature_list]
    )

    prob = model.predict_proba(
        X_test[feature_list]
    )[:, 1]

    feature_count = len(feature_list)

    records.append({
        "method": method,
        "feature_count": feature_count,
        "feature_reduction_percent": (
            100.0
            * (
                1.0
                - feature_count / ALL_FEATURE_COUNT
            )
        ),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            pred,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            prob,
        ),
        "pr_auc": average_precision_score(
            y_test,
            prob,
        ),
        "mcc": matthews_corrcoef(
            y_test,
            pred,
        ),
        "selected_features": "|".join(
            feature_list
        ),
    })

    for rank, feature in enumerate(
        feature_list,
        start=1,
    ):
        selection_rows.append({
            "method": method,
            "rank": rank,
            "feature": feature,
        })

results_df = pd.DataFrame(records)

method_order = [
    "All-42",
    "QAOA-4",
    "MI-4",
    "ANOVA-4",
    "L1-4",
    "RF-4",
]

results_df["method"] = pd.Categorical(
    results_df["method"],
    categories=method_order,
    ordered=True,
)

results_df = (
    results_df
    .sort_values("method")
    .reset_index(drop=True)
)

selection_df = pd.DataFrame(
    selection_rows
)

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUT_CSV,
    index=False,
)

selection_df.to_csv(
    OUT_FEATURES,
    index=False,
)

summary = {
    "dataset_train": str(TRAIN_PATH),
    "dataset_test": str(TEST_PATH),
    "test_evaluation_locked": True,
    "feature_selection_frozen_from_development": True,
    "random_state": RANDOM_STATE,
    "classifier": "RandomForestClassifier",
    "n_estimators": N_ESTIMATORS,
    "all_feature_count": ALL_FEATURE_COUNT,
    "selected_features": selected_features,
    "results": results_df.to_dict(
        orient="records"
    ),
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
print("FINAL LOCKED UNSW-NB15 TEST BENCHMARK")
print()

print(
    results_df[
        [
            "method",
            "feature_count",
            "feature_reduction_percent",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "mcc",
        ]
    ].to_string(index=False)
)

print()
print("SELECTED FEATURES")
print()

for method in method_order:
    print(
        f"{method}: "
        + ", ".join(
            selected_features[method]
        )
    )

print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")
print(f"Saved: {OUT_FEATURES}")
