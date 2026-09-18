import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

BASE = Path(".")
TRAIN_PATH = BASE / "data" / "unsw_nb15" / "UNSW_NB15_training-set.csv"
INTERACTION_PATH = BASE / "results" / "directional_pairwise_interaction_matrix.csv"

OUT_CSV = BASE / "results" / "pairwise_robust_qaoa_sweep.csv"
OUT_JSON = BASE / "results" / "pairwise_robust_qaoa_sweep.json"

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

K = 4
LAMBDA = 0.0
DELTAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
RANDOM_STATE = 42
QAOA_REPS = 2
QAOA_MAXITER = 80
QAOA_SHOTS = 4096

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

X_train_candidates = X_train[FEATURES].copy()
X_val_candidates = X_val[FEATURES].copy()

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train_candidates),
    columns=FEATURES,
    index=X_train_candidates.index,
)
X_val_imp = pd.DataFrame(
    imputer.transform(X_val_candidates),
    columns=FEATURES,
    index=X_val_candidates.index,
)

from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(
    X_train_imp,
    y_train,
    random_state=RANDOM_STATE,
)

relevance_raw = pd.Series(mi, index=FEATURES, dtype=float)

if relevance_raw.max() > relevance_raw.min():
    relevance = (
        (relevance_raw - relevance_raw.min())
        / (relevance_raw.max() - relevance_raw.min())
    )
else:
    relevance = pd.Series(
        np.ones(len(FEATURES)),
        index=FEATURES,
        dtype=float,
    )

interaction = pd.read_csv(
    INTERACTION_PATH,
    index_col=0,
)

interaction = interaction.loc[FEATURES, FEATURES].astype(float)

interaction_max = float(interaction.values.max())

if interaction_max > 0:
    interaction_norm = interaction / interaction_max
else:
    interaction_norm = interaction.copy()

def objective(selection, delta):
    value = float(
        sum(relevance[f] for f in selection)
    )

    for a, b in combinations(selection, 2):
        value -= float(
            delta * interaction_norm.loc[a, b]
        )

    return value

def exact_solve(delta):
    best_selection = None
    best_value = -np.inf

    for combo in combinations(FEATURES, K):
        value = objective(combo, delta)

        if value > best_value:
            best_value = value
            best_selection = list(combo)

    return best_selection, float(best_value)

def build_qp(delta):
    qp = QuadraticProgram("pairwise_robust_qaoa")

    for feature in FEATURES:
        qp.binary_var(feature)

    linear = {
        feature: float(relevance[feature])
        for feature in FEATURES
    }

    quadratic = {}

    for a, b in combinations(FEATURES, 2):
        coeff = -float(
            LAMBDA * 0.0
            + delta * interaction_norm.loc[a, b]
        )

        if coeff != 0.0:
            quadratic[(a, b)] = coeff

    qp.maximize(
        linear=linear,
        quadratic=quadratic,
    )

    qp.linear_constraint(
        linear={feature: 1.0 for feature in FEATURES},
        sense="==",
        rhs=K,
        name="cardinality",
    )

    return qp

def qaoa_solve(delta, seed=42):
    qp = build_qp(delta)

    rng = np.random.default_rng(seed)

    initial_point = rng.uniform(
        -np.pi,
        np.pi,
        size=2 * QAOA_REPS,
    )

    sampler = StatevectorSampler(
        default_shots=QAOA_SHOTS,
        seed=seed,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=QAOA_MAXITER),
        reps=QAOA_REPS,
        initial_point=initial_point,
    )

    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    selected = [
        feature
        for feature in FEATURES
        if result.x[FEATURES.index(feature)] > 0.5
    ]

    value = objective(selected, delta)

    return selected, float(value)

def evaluate_subset(features):
    train_x = X_train_imp[features]
    val_x = X_val_imp[features]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model.fit(train_x, y_train)

    pred = model.predict(val_x)
    prob = model.predict_proba(val_x)[:, 1]

    return {
        "f1": float(f1_score(y_val, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_val, pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, prob)
        ),
    }

results = []

for delta in DELTAS:
    exact_selection, exact_value = exact_solve(delta)
    exact_metrics = evaluate_subset(exact_selection)

    qaoa_selection, qaoa_value = qaoa_solve(
        delta,
        seed=RANDOM_STATE,
    )
    qaoa_metrics = evaluate_subset(qaoa_selection)

    approximation_ratio = (
        qaoa_value / exact_value
        if exact_value != 0
        else np.nan
    )

    results.append({
        "delta": delta,
        "exact_selection": "|".join(exact_selection),
        "qaoa_selection": "|".join(qaoa_selection),
        "exact_objective": exact_value,
        "qaoa_objective": qaoa_value,
        "approximation_ratio": float(approximation_ratio),
        "exact_f1": exact_metrics["f1"],
        "exact_balanced_accuracy": exact_metrics[
            "balanced_accuracy"
        ],
        "exact_roc_auc": exact_metrics["roc_auc"],
        "qaoa_f1": qaoa_metrics["f1"],
        "qaoa_balanced_accuracy": qaoa_metrics[
            "balanced_accuracy"
        ],
        "qaoa_roc_auc": qaoa_metrics["roc_auc"],
    })

results_df = pd.DataFrame(results)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(OUT_CSV, index=False)

summary = {
    "features": FEATURES,
    "k": K,
    "lambda": LAMBDA,
    "deltas": DELTAS,
    "random_state": RANDOM_STATE,
    "qaoa_reps": QAOA_REPS,
    "qaoa_maxiter": QAOA_MAXITER,
    "qaoa_shots": QAOA_SHOTS,
    "relevance_raw": {
        k: float(v)
        for k, v in relevance_raw.items()
    },
    "relevance_normalized": {
        k: float(v)
        for k, v in relevance.items()
    },
    "interaction_max": interaction_max,
    "nonzero_interactions": int((interaction.values > 0).sum() // 2),
    "results": results,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Normalized relevance:")
print(relevance.sort_values(ascending=False).round(6).to_string())

print()
print("Normalized pairwise interaction matrix:")
print(interaction_norm.round(6).to_string())

print()
print("Pairwise robust QAOA sweep:")
print(results_df.to_string(index=False))

print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")

