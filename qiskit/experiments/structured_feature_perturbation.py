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

perturbation_levels = [0.05, 0.10, 0.20]

ttl_features = {"sttl"}
count_features = {"ct_state_ttl", "ct_dst_sport_ltm"}
multiplicative_features = {"dload", "dmean", "rate", "swin", "dwin"}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

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
        for level in perturbation_levels:
            for direction in [-1, 1]:
                Xpert = Xva_imp.copy()

                if feature in multiplicative_features:
                    Xpert[feature] = (
                        Xpert[feature] *
                        (1.0 + direction * level)
                    )

                elif feature in ttl_features:
                    shift = max(
                        1.0,
                        float(
                            Xtr_imp[feature].std()
                        ) * level * 10.0
                    )

                    Xpert[feature] = (
                        Xpert[feature] +
                        direction * shift
                    )

                    Xpert[feature] = (
                        Xpert[feature]
                        .clip(
                            lower=0,
                            upper=255
                        )
                    )

                elif feature in count_features:
                    shift = np.maximum(
                        1.0,
                        np.round(
                            Xtr_imp[feature].std() *
                            level
                        )
                    )

                    Xpert[feature] = (
                        Xpert[feature] +
                        direction * shift
                    )

                    Xpert[feature] = (
                        Xpert[feature]
                        .clip(lower=0)
                        .round()
                    )

                pert_pred = model.predict(Xpert)
                pert_prob = model.predict_proba(
                    Xpert
                )[:, 1]

                pert_f1 = f1_score(
                    yva,
                    pert_pred,
                    zero_division=0
                )

                pert_balanced_accuracy = (
                    balanced_accuracy_score(
                        yva,
                        pert_pred
                    )
                )

                pert_roc_auc = roc_auc_score(
                    yva,
                    pert_prob
                )

                records.append({
                    "fold": fold,
                    "feature": feature,
                    "perturbation_level": level,
                    "direction": direction,
                    "clean_f1": float(clean_f1),
                    "perturbed_f1": float(pert_f1),
                    "delta_f1": float(
                        pert_f1 - clean_f1
                    ),
                    "clean_balanced_accuracy": float(
                        clean_balanced_accuracy
                    ),
                    "perturbed_balanced_accuracy": float(
                        pert_balanced_accuracy
                    ),
                    "delta_balanced_accuracy": float(
                        pert_balanced_accuracy -
                        clean_balanced_accuracy
                    ),
                    "clean_roc_auc": float(
                        clean_roc_auc
                    ),
                    "perturbed_roc_auc": float(
                        pert_roc_auc
                    ),
                    "delta_roc_auc": float(
                        pert_roc_auc -
                        clean_roc_auc
                    )
                })

result = pd.DataFrame(records)

result["f1_degradation"] = (
    result["clean_f1"] -
    result["perturbed_f1"]
).clip(lower=0)

summary = (
    result
    .groupby(["feature", "perturbation_level"])
    .agg(
        mean_f1_degradation=(
            "f1_degradation",
            "mean"
        ),
        worst_f1_degradation=(
            "f1_degradation",
            "max"
        ),
        mean_delta_f1=("delta_f1", "mean"),
        mean_delta_balanced_accuracy=(
            "delta_balanced_accuracy",
            "mean"
        ),
        mean_delta_roc_auc=(
            "delta_roc_auc",
            "mean"
        )
    )
    .reset_index()
)

robustness = (
    result
    .groupby("feature")
    .agg(
        mean_f1_degradation=(
            "f1_degradation",
            "mean"
        ),
        worst_f1_degradation=(
            "f1_degradation",
            "max"
        ),
        mean_delta_f1=("delta_f1", "mean"),
        mean_delta_balanced_accuracy=(
            "delta_balanced_accuracy",
            "mean"
        ),
        mean_delta_roc_auc=(
            "delta_roc_auc",
            "mean"
        )
    )
    .reset_index()
)

robustness["robustness_cost"] = (
    0.5 * robustness["mean_f1_degradation"] +
    0.5 * robustness["worst_f1_degradation"]
)

robustness = robustness.sort_values(
    "robustness_cost",
    ascending=False
)

print("STRUCTURED PERTURBATION SUMMARY")
print(summary.to_string(index=False))

print()
print("ROBUSTNESS COST")
print(robustness.to_string(index=False))

result.to_csv(
    "results/structured_feature_perturbation.csv",
    index=False
)

summary.to_csv(
    "results/structured_perturbation_summary.csv",
    index=False
)

robustness.to_csv(
    "results/structured_feature_robustness_scores.csv",
    index=False
)

with open(
    "results/structured_feature_robustness_scores.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        robustness.to_dict(orient="records"),
        f,
        indent=2
    )
