import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
PAIR_PATH = BASE / "results" / "joint_perturbation_robustness_matrix.csv"
SINGLE_PATH = BASE / "results" / "structured_feature_robustness_scores.csv"
OUT_CSV = BASE / "results" / "pairwise_robustness_synergy_matrix.csv"
OUT_JSON = BASE / "results" / "pairwise_robustness_synergy_summary.json"

pair_matrix = pd.read_csv(PAIR_PATH, index_col=0)

single = pd.read_csv(SINGLE_PATH)

feature_col = None
for candidate in ["feature", "feature_name", "Feature"]:
    if candidate in single.columns:
        feature_col = candidate
        break

if feature_col is None:
    raise ValueError(
        f"Could not identify feature column. Columns: {single.columns.tolist()}"
    )

degradation_col = None
for candidate in [
    "mean_f1_degradation",
    "mean_degradation",
    "robustness_cost",
]:
    if candidate in single.columns:
        degradation_col = candidate
        break

if degradation_col is None:
    raise ValueError(
        f"Could not identify degradation column. Columns: {single.columns.tolist()}"
    )

single_map = dict(
    zip(
        single[feature_col].astype(str),
        single[degradation_col].astype(float),
    )
)

features = pair_matrix.index.tolist()

missing = [f for f in features if f not in single_map]

if missing:
    raise ValueError(
        f"Missing single-feature robustness values for: {missing}"
    )

single_vector = pd.Series(
    {f: single_map[f] for f in features},
    dtype=float,
)

synergy = pd.DataFrame(
    np.zeros((len(features), len(features))),
    index=features,
    columns=features,
)

records = []

for i, feature_a in enumerate(features):
    for j, feature_b in enumerate(features):
        if i >= j:
            continue

        joint = float(pair_matrix.loc[feature_a, feature_b])
        additive = float(
            0.5
            * (
                single_vector[feature_a]
                + single_vector[feature_b]
            )
        )

        interaction = max(0.0, joint - additive)

        synergy.loc[feature_a, feature_b] = interaction
        synergy.loc[feature_b, feature_a] = interaction

        records.append({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "joint_degradation": joint,
            "single_a": float(single_vector[feature_a]),
            "single_b": float(single_vector[feature_b]),
            "additive_reference": additive,
            "pairwise_synergy": interaction,
        })

records_df = pd.DataFrame(records).sort_values(
    "pairwise_synergy",
    ascending=False,
)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

synergy.to_csv(OUT_CSV)

summary = {
    "features": features,
    "definition": "max(0, joint degradation - average single-feature degradation)",
    "num_pairs": len(records_df),
    "mean_synergy": float(records_df["pairwise_synergy"].mean()),
    "max_synergy": float(records_df["pairwise_synergy"].max()),
    "nonzero_pairs": int(
        (records_df["pairwise_synergy"] > 0).sum()
    ),
    "top_pairs": records_df.head(10).to_dict(orient="records"),
    "synergy_matrix_path": str(OUT_CSV),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Single-feature robustness:")
print(single_vector.sort_values(ascending=False).round(6).to_string())

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
    f"{summary['nonzero_pairs']}/{summary['num_pairs']}"
)

print()
print("Pairwise synergy matrix:")
print(synergy.round(6).to_string())

print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")
