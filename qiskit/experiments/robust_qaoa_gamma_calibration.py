import itertools
import json
import numpy as np
import pandas as pd

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

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

gammas = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0
]

train = pd.read_csv(
    "data/unsw_nb15/UNSW_NB15_training-set.csv"
)

relevance_df = pd.read_csv(
    "results/unsw_feature_relevance.csv",
    index_col=0
)

relevance = relevance_df.iloc[:, 0].astype(float)

robustness_df = pd.read_csv(
    "results/structured_feature_robustness_scores_v2.csv"
).set_index("feature")

X = train[candidate_features].copy()

for feature in candidate_features:
    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

redundancy = X.corr(method="pearson").abs().copy()

redundancy = redundancy.mask(np.eye(len(redundancy), dtype=bool), 0.0)

relevance = relevance.loc[candidate_features]

robustness = robustness_df.loc[
    candidate_features,
    "robustness_cost"
].astype(float)

def minmax(series):
    minimum = float(series.min())
    maximum = float(series.max())

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - minimum) /
        (maximum - minimum)
    )

relevance_norm = minmax(relevance)
robustness_norm = minmax(robustness)

rows = []

for gamma in gammas:
    def score(combo):
        relevance_term = sum(
            float(relevance_norm[f])
            for f in combo
        )

        robustness_term = sum(
            float(robustness_norm[f])
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
            - 0.25 * redundancy_term
        )

        return (
            objective,
            relevance_term,
            robustness_term,
            redundancy_term
        )

    exact_best = None

    for combo in itertools.combinations(
        candidate_features,
        4
    ):
        components = score(combo)

        if (
            exact_best is None
            or components[0] >
               exact_best["objective"]
        ):
            exact_best = {
                "selected": list(combo),
                "objective": components[0],
                "relevance_term": components[1],
                "robustness_term": components[2],
                "redundancy_term": components[3]
            }

    qp = QuadraticProgram(
        name=f"robust_qaoa_gamma_{gamma}"
    )

    for feature in candidate_features:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={
            feature: 1
            for feature in candidate_features
        },
        sense="==",
        rhs=4,
        name="select_k"
    )

    linear = {
        feature: float(
            relevance_norm[feature]
            - gamma * robustness_norm[feature]
        )
        for feature in candidate_features
    }

    quadratic = {}

    for i in range(
        len(candidate_features)
    ):
        for j in range(
            i + 1,
            len(candidate_features)
        ):
            a = candidate_features[i]
            b = candidate_features[j]

            quadratic[(a, b)] = (
                -0.25 *
                float(
                    redundancy.loc[a, b]
                )
            )

    qp.maximize(
        linear=linear,
        quadratic=quadratic
    )

    qaoa = QAOA(
        sampler=StatevectorSampler(
            seed=700 + int(gamma * 100)
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

    qaoa_components = score(selected)

    exact_objective = exact_best["objective"]

    approximation_ratio = (
        qaoa_components[0] /
        exact_objective
        if abs(exact_objective) > 1e-12
        else np.nan
    )

    rows.append({
        "gamma": gamma,
        "qaoa_selected": selected,
        "qaoa_objective": float(
            qaoa_components[0]
        ),
        "qaoa_relevance_term": float(
            qaoa_components[1]
        ),
        "qaoa_robustness_term": float(
            qaoa_components[2]
        ),
        "qaoa_redundancy_term": float(
            qaoa_components[3]
        ),
        "exact_selected": exact_best[
            "selected"
        ],
        "exact_objective": float(
            exact_best["objective"]
        ),
        "exact_relevance_term": float(
            exact_best["relevance_term"]
        ),
        "exact_robustness_term": float(
            exact_best["robustness_term"]
        ),
        "exact_redundancy_term": float(
            exact_best["redundancy_term"]
        ),
        "approximation_ratio": float(
            approximation_ratio
        )
    })

result_df = pd.DataFrame(rows)

normalized = pd.DataFrame({
    "feature": candidate_features,
    "raw_relevance": relevance.values,
    "normalized_relevance": relevance_norm.values,
    "raw_robustness_cost": robustness.values,
    "normalized_robustness_cost": robustness_norm.values
})

print("NORMALIZED COEFFICIENTS")
print(
    normalized.to_string(
        index=False
    )
)

print()
print("GAMMA CALIBRATION")
print(
    result_df.to_string(
        index=False
    )
)

result_df.to_csv(
    "results/robust_qaoa_gamma_calibration.csv",
    index=False
)

normalized.to_csv(
    "results/robust_qaoa_normalized_coefficients.csv",
    index=False
)

redundancy.to_csv(
    "results/robust_qaoa_redundancy_matrix.csv"
)

with open(
    "results/robust_qaoa_gamma_calibration.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        rows,
        f,
        indent=2
    )

with open(
    "results/robust_qaoa_normalization.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "candidate_features": candidate_features,
            "k": 4,
            "redundancy_weight": 0.25,
            "gamma_grid": gammas,
            "relevance_min": float(
                relevance.min()
            ),
            "relevance_max": float(
                relevance.max()
            ),
            "robustness_min": float(
                robustness.min()
            ),
            "robustness_max": float(
                robustness.max()
            )
        },
        f,
        indent=2
    )


