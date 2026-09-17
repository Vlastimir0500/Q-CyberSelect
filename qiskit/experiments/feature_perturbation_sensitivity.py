import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")

target = "label"

candidate_features = [
    "sttl",
    "ct_state_ttl",
    "dload",
    "ct_dst_sport_ltm",
    "dmean",
    "rate",
    "swin",
    "dwin"
]

X = df[candidate_features].copy()
y = df[target].astype(int)

for col in candidate_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

perturbation_levels = [0.05, 0.10, 0.20]

records = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y),
    start=1
):
    Xtr = X.iloc[train_idx].copy()
    Xva = X.iloc[val_idx].copy()
    ytr = y.iloc[train_idx]
    yva = y.iloc[val_idx]

    imputer = SimpleImputer(strategy="median")

    Xtr_imp = pd.DataFrame(
        imputer.fit_transform(Xtr),
        columns=candidate_features,
        index=Xtr.index
    )

    Xva_imp = pd.DataFrame(
        imputer.transform(Xva),
        columns=candidate_features,
        index=Xva.index
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        Xtr_imp,
        ytr
    )

    clean_pred = model.predict(Xva_imp)
    clean_prob = model.predict_proba(Xva_imp)[:, 1]

    clean_f1 = f1_score(
        yva,
        clean_pred,
        zero_division=0
    )

    clean_balanced_accuracy = balanced_accuracy_score(
        yva,
        clean_pred
    )

    clean_roc_auc = roc_auc_score(
        yva,
        clean_prob
    )

    for feature in candidate_features:
        feature_std = float(Xtr_imp[feature].std())
        feature_mean = float(Xtr_imp[feature].mean())

        for level in perturbation_levels:
            rng = np.random.default_rng(
                1000 + fold * 100 + int(level * 100)
            )

            Xpert = Xva_imp.copy()

            noise = rng.normal(
                loc=0.0,
                scale=max(feature_std * level, 1e-12),
                size=len(Xpert)
            )

            Xpert[feature] = Xpert[feature] + noise

            pert_pred = model.predict(Xpert)
            pert_prob = model.predict_proba(Xpert)[:, 1]

            pert_f1 = f1_score(
                yva,
                pert_pred,
                zero_division=0
            )

            pert_balanced_accuracy = balanced_accuracy_score(
                yva,
                pert_pred
            )

            pert_roc_auc = roc_auc_score(
                yva,
                pert_prob
            )

            records.append({
                "fold": fold,
                "feature": feature,
                "perturbation_level": level,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "clean_f1": clean_f1,
                "perturbed_f1": float(pert_f1),
                "delta_f1": float(pert_f1 - clean_f1),
                "clean_balanced_accuracy": float(clean_balanced_accuracy),
                "perturbed_balanced_accuracy": float(pert_balanced_accuracy),
                "delta_balanced_accuracy": float(
                    pert_balanced_accuracy -
                    clean_balanced_accuracy
                ),
                "clean_roc_auc": float(clean_roc_auc),
                "perturbed_roc_auc": float(pert_roc_auc),
                "delta_roc_auc": float(
                    pert_roc_auc -
                    clean_roc_auc
                )
            })

result = pd.DataFrame(records)

summary = (
    result
    .groupby(["feature", "perturbation_level"])
    .agg(
        mean_delta_f1=("delta_f1", "mean"),
        std_delta_f1=("delta_f1", "std"),
        mean_delta_balanced_accuracy=(
            "delta_balanced_accuracy",
            "mean"
        ),
        mean_delta_roc_auc=("delta_roc_auc", "mean")
    )
    .reset_index()
)

robustness = (
    result
    .groupby("feature")
    .agg(
        mean_delta_f1=("delta_f1", "mean"),
        worst_delta_f1=("delta_f1", "min"),
        mean_delta_balanced_accuracy=(
            "delta_balanced_accuracy",
            "mean"
        ),
        mean_delta_roc_auc=("delta_roc_auc", "mean")
    )
    .reset_index()
)

robustness["sensitivity_cost"] = (
    -robustness["mean_delta_f1"]
)

robustness = robustness.sort_values(
    "sensitivity_cost",
    ascending=False
)

print("PERTURBATION SUMMARY")
print(summary.to_string(index=False))

print()
print("FEATURE SENSITIVITY")
print(robustness.to_string(index=False))

result.to_csv(
    "results/feature_perturbation_sensitivity.csv",
    index=False
)

summary.to_csv(
    "results/feature_perturbation_summary.csv",
    index=False
)

robustness.to_csv(
    "results/feature_robustness_scores.csv",
    index=False
)

with open(
    "results/feature_robustness_scores.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        robustness.to_dict(orient="records"),
        f,
        indent=2
    )
