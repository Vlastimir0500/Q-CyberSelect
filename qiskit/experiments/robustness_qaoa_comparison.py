import itertools
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

DATA = "data/unsw_nb15/UNSW_NB15_training-set.csv"

df = pd.read_csv(DATA)

target = "label"
drop_cols = ["id", "attack_cat", target]

all_features = [
    c for c in df.columns
    if c not in drop_cols
]

categorical = {"proto", "service", "state"}

numeric_features = [
    c for c in all_features
    if c not in categorical
]

y = df[target].astype(int)

k = 4
candidate_count = 8
redundancy_weight = 0.25
robust_gamma = 1.0

perturbation_levels = [0.05, 0.10, 0.20]

def prepare_numeric(frame, features, medians=None):
    out = frame[features].copy()

    for feature in features:
        out[feature] = pd.to_numeric(
            out[feature],
            errors="coerce"
        )

    out = out.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if medians is None:
        medians = out.median()

    out = out.fillna(medians)

    return out, medians

def normalize(series):
    minimum = float(series.min())
    maximum = float(series.max())

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )

def compute_robustness_cost(
    frame,
    labels,
    candidates,
    random_state
):
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state
    )

    feature_scores = {
        feature: []
        for feature in candidates
    }

    X = frame[candidates].reset_index(drop=True)
    y_local = labels.reset_index(drop=True)

    for train_idx, val_idx in cv.split(X, y_local):
        Xtr_raw = X.iloc[train_idx]
        Xva_raw = X.iloc[val_idx]

        ytr = y_local.iloc[train_idx]
        yva = y_local.iloc[val_idx]

        Xtr, medians = prepare_numeric(
            Xtr_raw,
            candidates
        )

        Xva, _ = prepare_numeric(
            Xva_raw,
            candidates,
            medians
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(
            Xtr,
            ytr
        )

        clean_pred = model.predict(Xva)

        clean_f1 = f1_score(
            yva,
            clean_pred,
            zero_division=0
        )

        for feature in candidates:
            degradations = []

            for level in perturbation_levels:
                for direction in [-1, 1]:
                    perturbed = Xva.copy()

                    values = perturbed[
                        feature
                    ].to_numpy()

                    if feature == "sttl":
                        shift = np.ceil(
                            np.maximum(
                                1.0,
                                np.abs(values) * level
                            )
                        )

                        values = np.clip(
                            values + direction * shift,
                            0,
                            255
                        )

                    elif feature in {
                        "ct_state_ttl",
                        "ct_dst_sport_ltm"
                    }:
                        shift = np.ceil(
                            np.maximum(
                                1.0,
                                np.abs(values) * level
                            )
                        )

                        values = np.clip(
                            np.round(
                                values + direction * shift
                            ),
                            0,
                            None
                        )

                    else:
                        values = values * (
                            1.0 + direction * level
                        )

                    perturbed[feature] = values

                    pert_pred = model.predict(
                        perturbed
                    )

                    pert_f1 = f1_score(
                        yva,
                        pert_pred,
                        zero_division=0
                    )

                    degradations.append(
                        max(
                            0.0,
                            clean_f1 - pert_f1
                        )
                    )

            feature_scores[
                feature
            ].extend(degradations)

    return pd.Series({
        feature: (
            0.5 * np.mean(
                feature_scores[feature]
            )
            +
            0.5 * np.max(
                feature_scores[feature]
            )
        )
        for feature in candidates
    })

def objective_for_combo(
    combo,
    relevance,
    robustness,
    redundancy,
    gamma
):
    relevance_term = sum(
        float(relevance[f])
        for f in combo
    )

    robustness_term = sum(
        float(robustness[f])
        for f in combo
    )

    redundancy_term = 0.0

    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            redundancy_term += float(
                redundancy.loc[
                    combo[i],
                    combo[j]
                ]
            )

    objective = (
        relevance_term
        - gamma * robustness_term
        - redundancy_weight * redundancy_term
    )

    return (
        objective,
        relevance_term,
        robustness_term,
        redundancy_term
    )

def solve_qaoa(
    candidates,
    relevance,
    robustness,
    redundancy,
    gamma,
    seed
):
    qp = QuadraticProgram(
        name=f"robustness_comparison_gamma_{gamma}"
    )

    for feature in candidates:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={
            feature: 1
            for feature in candidates
        },
        sense="==",
        rhs=k,
        name="select_k"
    )

    qp.maximize(
        linear={
            feature: float(
                relevance[feature]
                - gamma * robustness[feature]
            )
            for feature in candidates
        },
        quadratic={
            (
                candidates[i],
                candidates[j]
            ): -redundancy_weight * float(
                redundancy.loc[
                    candidates[i],
                    candidates[j]
                ]
            )
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
        }
    )

    qaoa = QAOA(
        sampler=StatevectorSampler(
            seed=seed
        ),
        optimizer=COBYLA(
            maxiter=12
        ),
        reps=1
    )

    result = MinimumEigenOptimizer(
        qaoa
    ).solve(qp)

    selected = [
        variable.name
        for variable, value in zip(
            qp.variables,
            result.x
        )
        if value > 0.5
    ]

    return selected

def evaluate_subset(
    train,
    validation,
    y_train,
    y_validation,
    selected,
    seed
):
    Xtr, medians = prepare_numeric(
        train,
        selected
    )

    Xva, _ = prepare_numeric(
        validation,
        selected,
        medians
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        Xtr,
        y_train
    )

    clean_pred = model.predict(Xva)
    clean_prob = model.predict_proba(Xva)[:, 1]

    clean_f1 = f1_score(
        y_validation,
        clean_pred,
        zero_division=0
    )

    clean_balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            clean_pred
        )
    )

    clean_roc_auc = roc_auc_score(
        y_validation,
        clean_prob
    )

    perturbation_records = []

    for feature in selected:
        for level in perturbation_levels:
            for direction in [-1, 1]:
                perturbed = Xva.copy()

                values = perturbed[
                    feature
                ].to_numpy()

                if feature == "sttl":
                    shift = np.ceil(
                        np.maximum(
                            1.0,
                            np.abs(values) * level
                        )
                    )

                    values = np.clip(
                        values + direction * shift,
                        0,
                        255
                    )

                elif feature in {
                    "ct_state_ttl",
                    "ct_dst_sport_ltm"
                }:
                    shift = np.ceil(
                        np.maximum(
                            1.0,
                            np.abs(values) * level
                        )
                    )

                    values = np.clip(
                        np.round(
                            values + direction * shift
                        ),
                        0,
                        None
                    )

                else:
                    values = values * (
                        1.0 + direction * level
                    )

                perturbed[feature] = values

                pert_pred = model.predict(
                    perturbed
                )

                pert_prob = model.predict_proba(
                    perturbed
                )[:, 1]

                pert_f1 = f1_score(
                    y_validation,
                    pert_pred,
                    zero_division=0
                )

                pert_balanced_accuracy = (
                    balanced_accuracy_score(
                        y_validation,
                        pert_pred
                    )
                )

                pert_roc_auc = roc_auc_score(
                    y_validation,
                    pert_prob
                )

                perturbation_records.append({
                    "feature": feature,
                    "level": level,
                    "direction": direction,
                    "f1": float(pert_f1),
                    "f1_degradation": float(
                        max(
                            0.0,
                            clean_f1 - pert_f1
                        )
                    ),
                    "balanced_accuracy": float(
                        pert_balanced_accuracy
                    ),
                    "roc_auc": float(
                        pert_roc_auc
                    )
                })

    perturbation_df = pd.DataFrame(
        perturbation_records
    )

    return {
        "clean_f1": float(clean_f1),
        "clean_balanced_accuracy": float(
            clean_balanced_accuracy
        ),
        "clean_roc_auc": float(clean_roc_auc),
        "mean_perturbed_f1": float(
            perturbation_df["f1"].mean()
        ),
        "worst_perturbed_f1": float(
            perturbation_df["f1"].min()
        ),
        "mean_f1_degradation": float(
            perturbation_df[
                "f1_degradation"
            ].mean()
        ),
        "worst_f1_degradation": float(
            perturbation_df[
                "f1_degradation"
            ].max()
        ),
        "mean_perturbed_balanced_accuracy": float(
            perturbation_df[
                "balanced_accuracy"
            ].mean()
        ),
        "mean_perturbed_roc_auc": float(
            perturbation_df[
                "roc_auc"
            ].mean()
        )
    }

outer_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

fold_records = []

for outer_fold, (
    train_idx,
    val_idx
) in enumerate(
    outer_cv.split(df, y),
    start=1
):
    outer_train = df.iloc[
        train_idx
    ].reset_index(drop=True)

    outer_val = df.iloc[
        val_idx
    ].reset_index(drop=True)

    y_train = y.iloc[
        train_idx
    ].reset_index(drop=True)

    y_val = y.iloc[
        val_idx
    ].reset_index(drop=True)

    mi = mutual_info_classif(
        prepare_numeric(
            outer_train,
            numeric_features
        )[0],
        y_train,
        random_state=42
    )

    mi_scores = pd.Series(
        mi,
        index=numeric_features
    )

    candidates = (
        mi_scores
        .sort_values(
            ascending=False
        )
        .head(candidate_count)
        .index
        .tolist()
    )

    candidate_data, medians = prepare_numeric(
        outer_train,
        candidates
    )

    redundancy = (
        candidate_data
        .corr()
        .abs()
        .copy()
    )

    redundancy = redundancy.mask(
        np.eye(
            len(redundancy),
            dtype=bool
        ),
        0.0
    )

    relevance = normalize(
        mi_scores.loc[candidates]
    )

    robustness_raw = compute_robustness_cost(
        outer_train,
        y_train,
        candidates,
        1200 + outer_fold
    )

    robustness = normalize(
        robustness_raw
    )

    baseline_selected = solve_qaoa(
        candidates,
        relevance,
        robustness,
        redundancy,
        0.0,
        1300 + outer_fold
    )

    robust_selected = solve_qaoa(
        candidates,
        relevance,
        robustness,
        redundancy,
        robust_gamma,
        1400 + outer_fold
    )

    baseline_eval = evaluate_subset(
        outer_train,
        outer_val,
        y_train,
        y_val,
        baseline_selected,
        1500 + outer_fold
    )

    robust_eval = evaluate_subset(
        outer_train,
        outer_val,
        y_train,
        y_val,
        robust_selected,
        1600 + outer_fold
    )

    fold_records.append({
        "outer_fold": outer_fold,
        "candidates": candidates,
        "baseline_selected": baseline_selected,
        "robust_selected": robust_selected,
        "baseline": baseline_eval,
        "robust": robust_eval,
        "clean_f1_change": (
            robust_eval["clean_f1"] -
            baseline_eval["clean_f1"]
        ),
        "mean_degradation_change": (
            robust_eval[
                "mean_f1_degradation"
            ]
            -
            baseline_eval[
                "mean_f1_degradation"
            ]
        ),
        "worst_degradation_change": (
            robust_eval[
                "worst_f1_degradation"
            ]
            -
            baseline_eval[
                "worst_f1_degradation"
            ]
        )
    })

comparison_rows = []

for record in fold_records:
    comparison_rows.append({
        "outer_fold": record["outer_fold"],
        "method": "Baseline-QAOA-gamma0",
        "selected_features": record[
            "baseline_selected"
        ],
        **record["baseline"]
    })

    comparison_rows.append({
        "outer_fold": record["outer_fold"],
        "method": "Robust-QAOA-gamma1",
        "selected_features": record[
            "robust_selected"
        ],
        **record["robust"]
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

summary_rows = []

for method in [
    "Baseline-QAOA-gamma0",
    "Robust-QAOA-gamma1"
]:
    subset = comparison_df[
        comparison_df["method"] == method
    ]

    summary_rows.append({
        "method": method,
        "mean_clean_f1": subset[
            "clean_f1"
        ].mean(),
        "std_clean_f1": subset[
            "clean_f1"
        ].std(),
        "mean_clean_balanced_accuracy": subset[
            "clean_balanced_accuracy"
        ].mean(),
        "mean_clean_roc_auc": subset[
            "clean_roc_auc"
        ].mean(),
        "mean_perturbed_f1": subset[
            "mean_perturbed_f1"
        ].mean(),
        "mean_f1_degradation": subset[
            "mean_f1_degradation"
        ].mean(),
        "worst_f1_degradation": subset[
            "worst_f1_degradation"
        ].mean(),
        "mean_perturbed_balanced_accuracy": subset[
            "mean_perturbed_balanced_accuracy"
        ].mean(),
        "mean_perturbed_roc_auc": subset[
            "mean_perturbed_roc_auc"
        ].mean()
    })

summary_df = pd.DataFrame(
    summary_rows
)

print("OUTER FOLD COMPARISON")
print(
    comparison_df.to_string(
        index=False
    )
)

print()
print("SUMMARY")
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("ROBUSTNESS DELTA")
for record in fold_records:
    print(
        record["outer_fold"],
        "clean_f1_change=",
        record["clean_f1_change"],
        "mean_degradation_change=",
        record[
            "mean_degradation_change"
        ],
        "worst_degradation_change=",
        record[
            "worst_degradation_change"
        ]
    )

comparison_df.to_csv(
    "results/robustness_qaoa_comparison.csv",
    index=False
)

summary_df.to_csv(
    "results/robustness_qaoa_comparison_summary.csv",
    index=False
)

with open(
    "results/robustness_qaoa_comparison.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        fold_records,
        f,
        indent=2
    )
