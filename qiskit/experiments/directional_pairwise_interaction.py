import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

BASE = Path(".")
SINGLE_PATH = BASE / "results" / "matched_single_perturbation_robustness.csv"
JOINT_PATH = BASE / "results" / "joint_perturbation_robustness_matrix.csv"

OUT_CSV = BASE / "results" / "directional_pairwise_interaction_matrix.csv"
OUT_JSON = BASE / "results" / "directional_pairwise_interaction_summary.json"

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

single = pd.read_csv(SINGLE_PATH)
joint = pd.read_csv(JOINT_PATH, index_col=0)

single_map = single.set_index("feature")

positive = pd.DataFrame(
    np.zeros((len(FEATURES), len(FEATURES))),
    index=FEATURES,
    columns=FEATURES,
)

negative = positive.copy()

mean_interaction = positive.copy()

records = []

for feature_a, feature_b in combinations(FEATURES, 2):
    single_a_pos = float(single_map.loc[feature_a, "positive_degradation"])
    single_b_pos = float(single_map.loc[feature_b, "positive_degradation"])

    single_a_neg = float(single_map.loc[feature_a, "negative_degradation"])
    single_b_neg = float(single_map.loc[feature_b, "negative_degradation"])

    pair_row = None

    pair_csv = pd.read_csv(
        BASE / "results" / "joint_perturbation_robustness_summary.json"
    ) if False else None

    summary_path = BASE / "results" / "joint_perturbation_robustness_summary.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for item in summary["top_pairs"]:
        if {
            item["feature_a"],
            item["feature_b"],
        } == {feature_a, feature_b}:
            pair_row = item
            break

    if pair_row is None:
        raise ValueError(
            f"Pair not found in joint summary: {feature_a}, {feature_b}"
        )

    pos_joint = float(pair_row["positive_degradation"])
    neg_joint = float(pair_row["negative_degradation"])

    pos_interaction = max(
        0.0,
        pos_joint - single_a_pos - single_b_pos,
    )

    neg_interaction = max(
        0.0,
        neg_joint - single_a_neg - single_b_neg,
    )

    avg_interaction = 0.5 * (
        pos_interaction + neg_interaction
    )

    positive.loc[feature_a, feature_b] = pos_interaction
    positive.loc[feature_b, feature_a] = pos_interaction

    negative.loc[feature_a, feature_b] = neg_interaction
    negative.loc[feature_b, feature_a] = neg_interaction

    mean_interaction.loc[feature_a, feature_b] = avg_interaction
    mean_interaction.loc[feature_b, feature_a] = avg_interaction

    records.append({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "positive_joint_degradation": pos_joint,
        "positive_additive_reference": single_a_pos + single_b_pos,
        "positive_interaction": pos_interaction,
        "negative_joint_degradation": neg_joint,
        "negative_additive_reference": single_a_neg + single_b_neg,
        "negative_interaction": neg_interaction,
        "mean_interaction": avg_interaction,
    })

records_df = pd.DataFrame(records).sort_values(
    "mean_interaction",
    ascending=False,
)

mean_interaction.to_csv(OUT_CSV)

summary = {
    "definition": "max(0, directional_joint_degradation - directional_single_a_degradation - directional_single_b_degradation)",
    "features": FEATURES,
    "num_pairs": len(records_df),
    "nonzero_pairs": int(
        (records_df["mean_interaction"] > 0).sum()
    ),
    "mean_interaction": float(
        records_df["mean_interaction"].mean()
    ),
    "max_interaction": float(
        records_df["mean_interaction"].max()
    ),
    "top_pairs": records_df.head(15).to_dict(
        orient="records"
    ),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Top directional pairwise interactions:")
print(
    records_df[
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
    f"{summary['nonzero_pairs']}/{summary['num_pairs']}"
)

print()
print("Mean directional interaction matrix:")
print(mean_interaction.round(6).to_string())

print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")
