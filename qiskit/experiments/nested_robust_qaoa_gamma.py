import itertools
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)

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

categorical = ["proto", "service", "state"]

numeric_features = [
    c for c in all_features
    if c not in categorical
]

y = df[target].astype(int)

gammas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
k = 4
candidate_count = 8
redundancy_weight = 0.25

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

def feature_mi_scores(frame, labels, features):
    Xnum, _ = prepare_numeric(frame, features)

    mi = mutual_info_classif(
        Xnum,
        labels,
        random_state=42
    )

    return pd.Series(
        mi,
        index=features
    )

def normalized(series):
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

def calculate_robustness_cost(
    frame,
    labels,
    candidates,
    random_state
):
    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state
    )

    records = []

    Xcandidate = frame[candidates].copy()

    for feature in candidates:
        fold_degradations = []

        for inner_train_idx, inner_val_idx in inner_cv.split(
            Xcandidate,
            labels
        ):
            inner_train = frame.iloc[inner_train_idx][candidates].copy()
            inner_val = frame.iloc[inner_val_idx][candidates].copy()

            y_train = labels.iloc[inner_train_idx]
            y_val = labels.iloc[inner_val_idx]

            Xtr, medians = prepare_numeric(
                inner_train,
                candidates
            )

            Xva, _ = prepare_numeric(
                inner_val,
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
                y_train
            )

            clean_pred = model.predict(Xva)

            clean_f1 = f1_score(
                y_val,
                clean_pred,
                zero_division=0
            )

            for level in [0.05, 0.10, 0.20]:
                perturbed = Xva.copy()

                values = perturbed[feature].to_numpy()

                if feature == "sttl":
                    shift = np.ceil(
                        np.maximum(
                            1.0,
                            np.abs(values) * level
                        )
                    )

                    values = np.clip(
                        values + shift,
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
                        np.round(values + shift),
                        0,
                        None
                    )

                else:
                    values = values * (
                        1.0 + level
                    )

                perturbed[feature] = values

                perturbed_pred = model.predict(
                    perturbed
                )

                perturbed_f1 = f1_score(
                    y_val,
                    perturbed_pred,
                    zero_division=0
                )

                degradation = max(
                    0.0,
                    clean_f1 - perturbed_f1
                )

                fold_degradations.append(
                    degradation
                )

        records.append({
            "feature": feature,
            "mean_f1_degradation": float(
                np.mean(fold_degradations)
            ),
            "worst_f1_degradation": float(
                np.max(fold_degradations)
            )
        })

    robustness = pd.DataFrame(records)

    robustness["robustness_cost"] = (
        0.5 *
        robustness["mean_f1_degradation"] +
        0.5 *
        robustness["worst_f1_degradation"]
    )

    return robustness.set_index("feature")[
        "robustness_cost"
    ]

def exact_solution(
    candidates,
    relevance,
    robustness,
    redundancy
):
    best = None

    for combo in itertools.combinations(
        candidates,
        k
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
            - robustness_term
            - redundancy_weight *
              redundancy_term
        )

        if (
            best is None
            or objective > best["objective"]
        ):
            best = {
                "selected": list(combo),
                "objective": float(objective),
                "relevance_term": float(
                    relevance_term
                ),
                "robustness_term": float(
                    robustness_term
                ),
                "redundancy_term": float(
                    redundancy_term
                )
            }

    return best

def solve_qaoa(
    candidates,
    relevance,
    robustness,
    redundancy,
    seed
):
    qp = QuadraticProgram(
        name="nested_robust_qaoa"
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
                - robustness[feature]
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

outer_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

outer_records = []
inner_records = []

for outer_fold, (
    outer_train_idx,
    outer_val_idx
) in enumerate(
    outer_cv.split(df, y),
    start=1
):
    outer_train = df.iloc[
        outer_train_idx
    ].copy()

    outer_val = df.iloc[
        outer_val_idx
    ].copy()

    y_outer_train = y.iloc[
        outer_train_idx
    ].reset_index(drop=True)

    y_outer_val = y.iloc[
        outer_val_idx
    ].reset_index(drop=True)

    outer_train = outer_train.reset_index(
        drop=True
    )

    outer_val = outer_val.reset_index(
        drop=True
    )

    mi_scores = feature_mi_scores(
        outer_train,
        y_outer_train,
        numeric_features
    )

    candidates = (
        mi_scores
        .sort_values(ascending=False)
        .head(candidate_count)
        .index
        .tolist()
    )

    candidate_data, medians = prepare_numeric(
        outer_train,
        candidates
    )

    redundancy_raw = (
        candidate_data
        .corr(method="pearson")
        .abs()
        .copy()
    )

    redundancy = redundancy_raw.mask(
        np.eye(
            len(redundancy_raw),
            dtype=bool
        ),
        0.0
    )

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=100 + outer_fold
    )

    gamma_scores = {
        gamma: []
        for gamma in gammas
    }

    for inner_fold, (
        inner_train_idx,
        inner_val_idx
    ) in enumerate(
        inner_cv.split(
            outer_train,
            y_outer_train
        ),
        start=1
    ):
        inner_train = outer_train.iloc[
            inner_train_idx
        ].copy()

        inner_val = outer_train.iloc[
            inner_val_idx
        ].copy()

        y_inner_train = y_outer_train.iloc[
            inner_train_idx
        ]

        y_inner_val = y_outer_train.iloc[
            inner_val_idx
        ]

        inner_mi = feature_mi_scores(
            inner_train,
            y_inner_train,
            candidates
        )

        inner_relevance = normalized(
            inner_mi
        )

        inner_robustness_raw = calculate_robustness_cost(
            inner_train,
            y_inner_train.reset_index(
                drop=True
            ),
            candidates,
            500 + outer_fold * 10 + inner_fold
        )

        inner_robustness = normalized(
            inner_robustness_raw
        )

        inner_data, inner_medians = prepare_numeric(
            inner_train,
            candidates
        )

        inner_redundancy = (
            inner_data
            .corr(method="pearson")
            .abs()
            .copy()
        )

        inner_redundancy = inner_redundancy.mask(
            np.eye(
                len(inner_redundancy),
                dtype=bool
            ),
            0.0
        )

        for gamma in gammas:
            combo_best = None

            for combo in itertools.combinations(
                candidates,
                k
            ):
                relevance_term = sum(
                    float(inner_relevance[f])
                    for f in combo
                )

                robustness_term = sum(
                    float(inner_robustness[f])
                    for f in combo
                )

                redundancy_term = 0.0

                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        redundancy_term += float(
                            inner_redundancy.loc[
                                combo[i],
                                combo[j]
                            ]
                        )

                objective = (
                    relevance_term
                    - gamma *
                      robustness_term
                    - redundancy_weight *
                      redundancy_term
                )

                if (
                    combo_best is None
                    or objective >
                       combo_best["objective"]
                ):
                    combo_best = {
                        "selected": list(combo),
                        "objective": float(
                            objective
                        )
                    }

            inner_train_X, fitted_medians = prepare_numeric(
                inner_train,
                combo_best["selected"]
            )

            inner_val_X, _ = prepare_numeric(
                inner_val,
                combo_best["selected"],
                fitted_medians
            )

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )

            model.fit(
                inner_train_X,
                y_inner_train
            )

            inner_pred = model.predict(
                inner_val_X
            )

            inner_f1 = f1_score(
                y_inner_val,
                inner_pred,
                zero_division=0
            )

            gamma_scores[gamma].append(
                float(inner_f1)
            )

            inner_records.append({
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "gamma": gamma,
                "selected": combo_best[
                    "selected"
                ],
                "f1": float(inner_f1)
            })

    gamma_summary = {
        gamma: {
            "mean_f1": float(
                np.mean(scores)
            ),
            "std_f1": float(
                np.std(scores)
            )
        }
        for gamma, scores in gamma_scores.items()
    }

    selected_gamma = max(
        gamma_summary,
        key=lambda gamma:
        gamma_summary[gamma]["mean_f1"]
    )

    outer_robustness_raw = calculate_robustness_cost(
        outer_train,
        y_outer_train,
        candidates,
        900 + outer_fold
    )

    outer_relevance = normalized(
        mi_scores
    )

    outer_robustness = normalized(
        outer_robustness_raw
    )

    exact = exact_solution(
        candidates,
        outer_relevance,
        selected_gamma *
        outer_robustness,
        redundancy
    )

    qaoa_selected = solve_qaoa(
        candidates,
        outer_relevance,
        selected_gamma *
        outer_robustness,
        redundancy,
        800 + outer_fold
    )

    outer_train_X, selected_medians = prepare_numeric(
        outer_train,
        qaoa_selected
    )

    outer_val_X, _ = prepare_numeric(
        outer_val,
        qaoa_selected,
        selected_medians
    )

    final_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    final_model.fit(
        outer_train_X,
        y_outer_train
    )

    outer_pred = final_model.predict(
        outer_val_X
    )

    outer_prob = final_model.predict_proba(
        outer_val_X
    )[:, 1]

    qaoa_objective_components = exact_solution(
        candidates,
        outer_relevance,
        selected_gamma *
        outer_robustness,
        redundancy
    )

    qaoa_exact_components = None

    selected_combo = tuple(
        qaoa_selected
    )

    relevance_term = sum(
        float(outer_relevance[f])
        for f in selected_combo
    )

    robustness_term = sum(
        float(
            selected_gamma *
            outer_robustness[f]
        )
        for f in selected_combo
    )

    redundancy_term = 0.0

    for i in range(len(selected_combo)):
        for j in range(
            i + 1,
            len(selected_combo)
        ):
            redundancy_term += float(
                redundancy.loc[
                    selected_combo[i],
                    selected_combo[j]
                ]
            )

    qaoa_objective = (
        relevance_term
        - robustness_term
        - redundancy_weight *
          redundancy_term
    )

    approximation_ratio = (
        qaoa_objective /
        exact["objective"]
        if abs(exact["objective"]) > 1e-12
        else np.nan
    )

    outer_records.append({
        "outer_fold": outer_fold,
        "candidates": candidates,
        "selected_gamma": float(
            selected_gamma
        ),
        "inner_gamma_summary": gamma_summary,
        "qaoa_selected": qaoa_selected,
        "exact_selected": exact["selected"],
        "qaoa_objective": float(
            qaoa_objective
        ),
        "exact_objective": float(
            exact["objective"]
        ),
        "approximation_ratio": float(
            approximation_ratio
        ),
        "f1": float(
            f1_score(
                y_outer_val,
                outer_pred,
                zero_division=0
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_outer_val,
                outer_pred
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_outer_val,
                outer_prob
            )
        )
    })

summary = {
    "mean_f1": float(
        np.mean(
            [r["f1"] for r in outer_records]
        )
    ),
    "std_f1": float(
        np.std(
            [r["f1"] for r in outer_records],
            ddof=1
        )
    ),
    "mean_balanced_accuracy": float(
        np.mean(
            [
                r["balanced_accuracy"]
                for r in outer_records
            ]
        )
    ),
    "mean_roc_auc": float(
        np.mean(
            [
                r["roc_auc"]
                for r in outer_records
            ]
        )
    ),
    "mean_approximation_ratio": float(
        np.mean(
            [
                r["approximation_ratio"]
                for r in outer_records
            ]
        )
    )
}

print("OUTER FOLD RESULTS")
print(
    pd.DataFrame(outer_records)[
        [
            "outer_fold",
            "selected_gamma",
            "qaoa_selected",
            "exact_selected",
            "qaoa_objective",
            "exact_objective",
            "approximation_ratio",
            "f1",
            "balanced_accuracy",
            "roc_auc"
        ]
    ].to_string(index=False)
)

print()
print("SUMMARY")
print(
    json.dumps(
        summary,
        indent=2
    )
)

print()
print("SELECTED GAMMAS")
for record in outer_records:
    print(
        record["outer_fold"],
        record["selected_gamma"],
        record["qaoa_selected"]
    )

with open(
    "results/nested_robust_qaoa_gamma.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "outer_folds": outer_records,
            "inner_results": inner_records,
            "summary": summary
        },
        f,
        indent=2
    )

pd.DataFrame(
    outer_records
).to_csv(
    "results/nested_robust_qaoa_gamma_outer.csv",
    index=False
)

pd.DataFrame(
    inner_records
).to_csv(
    "results/nested_robust_qaoa_gamma_inner.csv",
    index=False
)
