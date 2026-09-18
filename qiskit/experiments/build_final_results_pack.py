import json
from pathlib import Path

import pandas as pd

BASE = Path(".")
RESULTS = BASE / "results"
PAPER = BASE / "paper"

PAPER.mkdir(parents=True, exist_ok=True)

benchmark_path = RESULTS / "final_locked_test_benchmark.csv"
robustness_path = RESULTS / "final_locked_test_robustness.csv"

required_files = [
    benchmark_path,
    robustness_path,
    RESULTS / "qaoa_cv_stability.json",
    RESULTS / "qaoa_nested_cv.json",
    RESULTS / "qaoa_stability_summary.json",
    RESULTS / "selector_stability_summary.json",
    RESULTS / "qaoa_paired_statistics.json",
    RESULTS / "robust_qaoa_gamma_calibration.json",
    RESULTS / "nested_robust_qaoa_gamma.json",
    RESULTS / "robustness_qaoa_comparison.json",
    RESULTS / "pairwise_robust_qaoa_sweep.json",
    RESULTS / "directional_pairwise_interaction_summary.json",
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing expected artifacts:")
    for path in missing_files:
        print(path)
    raise SystemExit(1)

benchmark = pd.read_csv(benchmark_path)
robustness = pd.read_csv(robustness_path)

qaoa_row = benchmark[
    benchmark["method"] == "QAOA-4"
].iloc[0]

all_row = benchmark[
    benchmark["method"] == "All-42"
].iloc[0]

feature_reduction = float(
    qaoa_row["feature_reduction_percent"]
)

qaoa_features = str(
    qaoa_row["selected_features"]
).split("|")

selector_rows = []

for _, row in benchmark.iterrows():
    method = row["method"]
    features = str(row["selected_features"]).split("|")

    for rank, feature in enumerate(features, start=1):
        selector_rows.append({
            "method": method,
            "rank": rank,
            "feature": feature,
        })

selected_features_df = pd.DataFrame(
    selector_rows
)

benchmark_output = RESULTS / "FINAL_TABLE_TEST_BENCHMARK.csv"
robustness_output = RESULTS / "FINAL_TABLE_TEST_ROBUSTNESS.csv"
features_output = RESULTS / "FINAL_TABLE_SELECTED_FEATURES.csv"
master_output = RESULTS / "FINAL_RESULTS_MASTER.json"
summary_output = PAPER / "FINAL_EXPERIMENTAL_SUMMARY.md"

benchmark.to_csv(
    benchmark_output,
    index=False,
)

robustness.to_csv(
    robustness_output,
    index=False,
)

selected_features_df.to_csv(
    features_output,
    index=False,
)

with open(
    RESULTS / "qaoa_cv_stability.json",
    "r",
    encoding="utf-8",
) as f:
    qaoa_cv = json.load(f)

with open(
    RESULTS / "qaoa_stability_summary.json",
    "r",
    encoding="utf-8",
) as f:
    qaoa_stability = json.load(f)

with open(
    RESULTS / "selector_stability_summary.json",
    "r",
    encoding="utf-8",
) as f:
    selector_stability = json.load(f)

with open(
    RESULTS / "qaoa_paired_statistics.json",
    "r",
    encoding="utf-8",
) as f:
    paired_statistics = json.load(f)

with open(
    RESULTS / "pairwise_robust_qaoa_sweep.json",
    "r",
    encoding="utf-8",
) as f:
    pairwise_qaoa = json.load(f)

with open(
    RESULTS / "directional_pairwise_interaction_summary.json",
    "r",
    encoding="utf-8",
) as f:
    pairwise_interaction = json.load(f)

master = {
    "project": "Q-CyberSelect",
    "title": "Quantum Graph-Theoretic Feature Selection for Robust Network Intrusion Detection",
    "dataset": {
        "name": "UNSW-NB15",
        "predictor_count": 42,
        "training_file": "data/unsw_nb15/UNSW_NB15_training-set.csv",
        "testing_file": "data/unsw_nb15/UNSW_NB15_testing-set.csv",
    },
    "frozen_evaluation": {
        "locked_test_evaluation": True,
        "random_state": 42,
        "classifier": "RandomForestClassifier",
        "n_estimators": 300,
        "qaoa_feature_count": 4,
        "qaoa_features": qaoa_features,
        "feature_reduction_percent": feature_reduction,
    },
    "locked_test_results": {
        "qaoa_4": {
            "precision": float(qaoa_row["precision"]),
            "recall": float(qaoa_row["recall"]),
            "f1": float(qaoa_row["f1"]),
            "balanced_accuracy": float(
                qaoa_row["balanced_accuracy"]
            ),
            "roc_auc": float(
                qaoa_row["roc_auc"]
            ),
            "pr_auc": float(
                qaoa_row["pr_auc"]
            ),
            "mcc": float(qaoa_row["mcc"]),
        },
        "all_42": {
            "precision": float(all_row["precision"]),
            "recall": float(all_row["recall"]),
            "f1": float(all_row["f1"]),
            "balanced_accuracy": float(
                all_row["balanced_accuracy"]
            ),
            "roc_auc": float(
                all_row["roc_auc"]
            ),
            "pr_auc": float(
                all_row["pr_auc"]
            ),
            "mcc": float(all_row["mcc"]),
        },
    },
    "selector_comparison": benchmark.to_dict(
        orient="records"
    ),
    "selected_features": (
        selected_features_df.to_dict(
            orient="records"
        )
    ),
    "cross_validation": {
        "qaoa_cv": qaoa_cv,
        "qaoa_stability": qaoa_stability,
        "selector_stability": selector_stability,
        "paired_statistics": paired_statistics,
    },
    "robustness": {
        "locked_test_results": robustness.to_dict(
            orient="records"
        ),
        "pairwise_qaoa_ablation": pairwise_qaoa,
        "pairwise_interaction_analysis": pairwise_interaction,
    },
    "interpretation": {
        "primary_result": (
            "QAOA produced a four-feature representation from a "
            "42-feature predictor space, corresponding to "
            f"{feature_reduction:.5f}% dimensionality reduction."
        ),
        "predictive_performance": (
            "The locked test evaluation does not demonstrate "
            "predictive superiority of QAOA feature selection over "
            "the evaluated classical feature-selection baselines."
        ),
        "quantum_advantage_claim": (
            "No quantum computational advantage is claimed. "
            "QAOA solutions were evaluated against exact classical "
            "optimization on the controlled problem instances used "
            "in the experiments."
        ),
        "robustness_ablation": (
            "The evaluated pairwise robustness penalty did not "
            "change the optimal four-feature subset across the "
            "tested penalty values and is therefore treated as an "
            "ablation rather than the principal model."
        ),
        "test_protocol": (
            "Feature-selection configurations used for the locked "
            "test evaluation were frozen from the development-stage "
            "experiments before evaluation on the held-out UNSW-NB15 "
            "testing split."
        ),
    },
}

with open(
    master_output,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        master,
        f,
        indent=2,
    )

def fmt(value):
    return f"{float(value):.6f}"

benchmark_table = benchmark[
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
].copy()

benchmark_table.columns = [
    "Method",
    "Features",
    "Reduction (%)",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "ROC-AUC",
    "PR-AUC",
    "MCC",
]

benchmark_lines = [
    "| Method | Features | Reduction (%) | Precision | Recall | F1 | Balanced Accuracy | ROC-AUC | PR-AUC | MCC |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for _, row in benchmark_table.iterrows():
    benchmark_lines.append(
        "| "
        + str(row["Method"])
        + " | "
        + str(int(row["Features"]))
        + " | "
        + fmt(row["Reduction (%)"])
        + " | "
        + fmt(row["Precision"])
        + " | "
        + fmt(row["Recall"])
        + " | "
        + fmt(row["F1"])
        + " | "
        + fmt(row["Balanced Accuracy"])
        + " | "
        + fmt(row["ROC-AUC"])
        + " | "
        + fmt(row["PR-AUC"])
        + " | "
        + fmt(row["MCC"])
        + " |"
    )

robustness_lines = [
    "| Method | Perturbation | Clean F1 | Perturbed Mean F1 | Mean Degradation | Worst Degradation |",
    "|---|---:|---:|---:|---:|---:|",
]

for _, row in robustness.iterrows():
    robustness_lines.append(
        "| "
        + str(row["method"])
        + " | "
        + fmt(row["perturbation_level"])
        + " | "
        + fmt(row["clean_f1"])
        + " | "
        + fmt(row["perturbed_mean_f1"])
        + " | "
        + fmt(row["mean_f1_degradation"])
        + " | "
        + fmt(row["worst_f1_degradation"])
        + " |"
    )

selection_lines = []

for method in benchmark["method"]:
    selected = benchmark[
        benchmark["method"] == method
    ].iloc[0]["selected_features"]

    selection_lines.append(
        f"**{method}:** "
        + ", ".join(str(selected).split("|"))
    )

summary = f"""# Q-CyberSelect — Final Experimental Summary

## 1. Study Overview

**Project:** Q-CyberSelect

**Study title:** Quantum Graph-Theoretic Feature Selection for Robust Network Intrusion Detection

**Dataset:** UNSW-NB15

**Predictor space:** 42 predictors after excluding `id`, `attack_cat`, and `label`.

The study evaluates graph-theoretic feature selection formulated as a binary quadratic optimization problem and solved using the Quantum Approximate Optimization Algorithm (QAOA). The experimental program includes classical feature-selection baselines, exact optimization checks, cross-validation, selection stability analysis, statistical comparison, and structured perturbation robustness analysis.

## 2. Frozen QAOA Configuration

The QAOA feature set used for the locked held-out test evaluation was:

{", ".join(f"`{feature}`" for feature in qaoa_features)}

The selected representation contains four predictors, corresponding to **{feature_reduction:.5f}% feature reduction** relative to the 42-feature predictor space.

## 3. Locked Held-Out Test Evaluation

The held-out UNSW-NB15 testing split was evaluated after the feature-selection configurations had been frozen during development.

{chr(10).join(benchmark_lines)}

The locked test results do not support a claim of predictive superiority for QAOA relative to the evaluated classical feature-selection baselines. The principal empirical outcome is dimensionality reduction through a quantum-optimization-based feature-selection formulation.

## 4. Selected Feature Sets

{chr(10).join(selection_lines)}

## 5. Robustness Evaluation

Structured perturbations were applied to the predefined robustness-feature set. Perturbations were evaluated at 5%, 10%, and 20% levels where the corresponding features were present in each model.

{chr(10).join(robustness_lines)}

The robustness analysis indicates heterogeneous sensitivity across feature-selection methods. These results are reported descriptively and are not used to alter the locked test configuration.

## 6. Pairwise Robustness Analysis

Joint perturbation experiments examined 28 feature pairs among the eight robustness candidates.

The non-additive interaction analysis identified three pairs with positive mean interaction under the specified perturbation protocol. The strongest interaction involved `sbytes` and `sload`.

A pairwise robustness-aware QUBO was subsequently evaluated across the tested penalty range. The exact optimizer and QAOA returned the same four-feature solution at every tested penalty value, and the selected subset did not change across the sweep.

Accordingly, the pairwise robustness formulation is treated as an **ablation and sensitivity analysis**, not as the principal contribution.

## 7. Optimization Validation

Controlled optimization experiments included exact enumeration for small candidate instances and comparison with QAOA solutions.

Where exact enumeration was feasible, QAOA solutions were evaluated against the exact objective value. The study does not claim quantum computational advantage; the experiments instead assess whether QAOA can reproduce or approximate solutions to the corresponding binary quadratic formulations.

## 8. Stability and Validation

The experimental program includes:

- QAOA seed stability analysis.
- Cross-validation stability.
- Nested feature-selection evaluation.
- Comparisons against mutual information, ANOVA, L1-regularized selection, and random-forest-based selection.
- Exploratory paired statistical comparisons.
- Structured perturbation analysis.
- Pairwise robustness sensitivity analysis.

## 9. Interpretation

The evidence supports the use of QAOA as an optimization mechanism for graph-theoretic feature selection in the evaluated intrusion-detection setting.

The results do **not** establish predictive superiority or quantum advantage. Instead, the study demonstrates a reproducible quantum-optimization-based feature-selection pipeline that reduces the dimensionality of the cybersecurity feature space substantially while retaining nontrivial predictive performance.

## 10. Limitations

The current experimental results are subject to several limitations.

First, the optimization instances are small enough for exact classical verification, so they do not constitute evidence of quantum speedup.

Second, the robustness analysis depends on the specified synthetic structured-perturbation protocol and should not be interpreted as a comprehensive model of real adversarial traffic manipulation.

Third, the locked test benchmark evaluates one frozen experimental configuration and should not be generalized beyond the UNSW-NB15 setting without additional datasets and external validation.

Fourth, the results demonstrate performance differences among feature-selection methods rather than an unconditional advantage of QAOA over classical methods.

## 11. Reproducibility Artifacts

Primary consolidated artifacts:

- `results/FINAL_RESULTS_MASTER.json`
- `results/FINAL_TABLE_TEST_BENCHMARK.csv`
- `results/FINAL_TABLE_TEST_ROBUSTNESS.csv`
- `results/FINAL_TABLE_SELECTED_FEATURES.csv`

The repository retains the underlying experiment scripts and intermediate result artifacts required to reconstruct the experimental workflow.
"""

summary_output.write_text(
    summary,
    encoding="utf-8",
)

print("FINAL EXPERIMENTAL PACKAGE")
print()
print(f"Dataset: UNSW-NB15")
print(f"Predictor count: {len(all_row['selected_features'].split('|'))}")
print(f"Frozen QAOA features: {', '.join(qaoa_features)}")
print(
    f"Feature reduction: "
    f"{feature_reduction:.5f}%"
)
print(
    f"Locked QAOA test F1: "
    f"{qaoa_row['f1']:.6f}"
)
print(
    f"Locked QAOA test ROC-AUC: "
    f"{qaoa_row['roc_auc']:.6f}"
)
print()
print("Primary artifacts:")
print(f"  {master_output}")
print(f"  {benchmark_output}")
print(f"  {robustness_output}")
print(f"  {features_output}")
print(f"  {summary_output}")
print()
print("All required experiment artifacts are present.")
