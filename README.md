# Q-CyberSelect

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-black?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-QAOA-black?style=flat-square&logo=ibm&logoColor=white)](https://www.ibm.com/quantum/qiskit)
[![C%2B%2B](https://img.shields.io/badge/C%2B%2B-Classical%20Baselines-black?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Rust](https://img.shields.io/badge/Rust-Security%20Pipeline-black?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Visualization-black?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-black?style=flat-square)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Quantum Graph-Theoretic Feature Selection for Network Intrusion Detection

A research implementation that formulates cybersecurity feature selection as a graph-structured binary quadratic optimization problem, solves controlled instances with the Quantum Approximate Optimization Algorithm (QAOA), validates solutions against exact classical optimization where tractable, and evaluates compact feature representations on UNSW-NB15.

</div>

<p align="center">
  <img src="docs/qcyberselect-hero.svg" alt="Q-CyberSelect overview" width="100%">
</p>

---

## Abstract

Modern network intrusion-detection datasets contain heterogeneous traffic measurements with overlapping and redundant information. Feature selection can reduce computational cost and simplify downstream detection models, but the search for a compact subset is combinatorial when feature interactions are considered explicitly.

**Q-CyberSelect** studies a graph-theoretic formulation of this problem. Features are represented as graph vertices, measured relationships provide graph structure, and candidate subsets are encoded using binary decision variables. The resulting selection problem is expressed as a **Quadratic Unconstrained Binary Optimization (QUBO)** model and solved with the **Quantum Approximate Optimization Algorithm (QAOA)**.

The experimental program is deliberately broader than a single QAOA run. It includes exact classical optimization on controlled instances, multiple classical feature-selection baselines, cross-validation, nested evaluation, seed stability, exploratory paired statistics, structured perturbation analysis, and pairwise robustness sensitivity.

The final locked evaluation reduces the predictor space from **42 features to 4 features**, a **90.48% reduction**. The held-out test results do not establish predictive superiority of QAOA over the evaluated classical selectors; the contribution is an experimentally validated quantum-optimization formulation for compact cybersecurity feature selection rather than a claim of quantum speedup.

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Contributions](#contributions)
- [Experimental Architecture](#experimental-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Dataset](#dataset)
- [Feature Graph](#feature-graph)
- [QUBO and QAOA](#qubo-and-qaoa)
- [Classical Baselines](#classical-baselines)
- [Validation Strategy](#validation-strategy)
- [Locked Test Results](#locked-test-results)
- [Robustness Analysis](#robustness-analysis)
- [Pairwise Robustness Ablation](#pairwise-robustness-ablation)
- [Selected Feature Sets](#selected-feature-sets)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Research Status](#research-status)
- [Citation](#citation)

---

## Overview

Q-CyberSelect is organized around one central research pipeline:

```text
UNSW-NB15
    │
    ▼
Preprocessing
    │
    ▼
Feature relationships
    │
    ▼
Graph representation
    │
    ▼
Graph-theoretic feature objective
    │
    ▼
QUBO
    │
    ▼
QAOA
    │
    ▼
Compact feature subset
    │
    ├───────────────┐
    ▼               ▼
IDS evaluation   Robustness analysis
    │               │
    └───────┬───────┘
            ▼
       Locked results
```

The project separates four distinct concerns:

1. **Optimization:** finding a compact subset under a graph-structured objective.
2. **Verification:** comparing QAOA solutions to exact classical optimization when feasible.
3. **Prediction:** measuring downstream intrusion-detection performance with a common classifier.
4. **Robustness:** evaluating the selected representations under controlled structured perturbations.

This separation is important because an optimizer can find a mathematically good solution without that subset necessarily producing the strongest classification metrics.

---

## Research Question

> **Can graph-theoretic feature selection formulated as a binary quadratic optimization problem produce a compact and reproducible feature representation for network intrusion detection when optimized with QAOA?**

The study treats QAOA as an optimization mechanism and evaluates the resulting subset through a conventional machine-learning pipeline.

The work does **not** assume that quantum optimization must outperform classical feature selection. Classical methods remain explicit baselines, and exact optimization is used where the search space permits it.

---

## Contributions

### Graph-structured feature selection

Cybersecurity predictors are represented through pairwise feature relationships rather than treated solely as independent scalar rankings.

### QUBO formulation

The subset-selection problem is expressed using binary variables and quadratic interactions, making the optimization problem compatible with QAOA.

### Quantum optimization with classical verification

QAOA is evaluated alongside exact enumeration for controlled problem instances so that optimization quality can be measured directly.

### Multi-stage validation

The repository contains development-stage cross-validation, nested evaluation, seed stability, selector comparisons, and an explicitly locked held-out test evaluation.

### Robustness analysis

Single-feature, structured, joint-pair, and pairwise interaction experiments examine sensitivity to controlled feature perturbations.

### Reproducible research artifact

The repository preserves experiment scripts, result tables, consolidated manifests, visual documentation, and the multi-language project structure.

---

## Experimental Architecture

<p align="center">
  <img src="docs/qcyberselect-architecture.svg" alt="Q-CyberSelect experimental architecture" width="100%">
</p>

### Stages

| Stage | Objective | Primary tooling |
|---|---|---|
| Data preparation | Load, clean, encode and split UNSW-NB15 | Python / scikit-learn |
| Feature modeling | Estimate relevance and feature relationships | Python |
| Graph construction | Represent predictors and their relationships | Python |
| Optimization | Encode subset selection as QUBO and solve with QAOA | Qiskit |
| Exact validation | Verify controlled instances exhaustively | Python / C++ |
| Classical comparison | Evaluate MI, ANOVA, L1 and RF selection | scikit-learn |
| Stability | Test seeds and folds | Python |
| Robustness | Evaluate structured perturbations | Python |
| Visualization | Present experiment outputs | TypeScript / frontend |
| Documentation | Preserve reproducibility and results | Markdown / Git |

---

## Mathematical Formulation

Let the feature set be:

\[
V = \{1,2,\ldots,n\}.
\]

For every candidate feature \(i\), introduce a binary variable:

\[
x_i \in \{0,1\},
\]

where \(x_i=1\) denotes selection.

A generic graph-structured objective is written as:

\[
\max_x
\left[
\sum_i r_i x_i
-
\lambda\sum_{i<j}R_{ij}x_ix_j
\right]
\]

subject to the fixed-cardinality constraint

\[
\sum_i x_i = k.
\]

Here:

- \(r_i\) is the relevance coefficient for feature \(i\);
- \(R_{ij}\) represents a pairwise feature relationship or redundancy term;
- \(\lambda\) controls the relative contribution of pairwise structure;
- \(k\) is the requested number of selected features.

The binary quadratic objective is mapped into a QUBO representation and subsequently into the cost Hamiltonian used by QAOA.

### Robustness extension

The project also tested single-feature and pairwise robustness terms. Pairwise interaction was measured using a directional non-additivity criterion of the form:

\[
I_{ij}^{(d)} = \max\left(0, D_{ij}^{(d)} - D_i^{(d)} - D_j^{(d)}\right),
\]

where \(D\) denotes F1 degradation under the specified perturbation direction \(d\).

The robustness-aware formulation is treated as an ablation because the evaluated pairwise penalty did not alter the final four-feature solution across the tested penalty range.

---

## Dataset

The experimental dataset is **UNSW-NB15**.

The repository uses the pre-split training and testing CSV files and excludes the following columns from the predictor matrix:

```text
id
attack_cat
label
```

Resulting predictor space:

| Quantity | Value |
|---|---:|
| Training rows | 175,341 |
| Testing rows | 82,332 |
| Predictors | 42 |
| Target | binary `label` |

The dataset files themselves are not redistributed by this repository.

---

## Feature Graph

The cybersecurity feature graph models relationships between candidate predictors. The graph is used to move beyond independent feature ranking and expose interactions that can be represented through quadratic terms.

The candidate modeling process includes:

```text
Raw predictors
      │
      ▼
Type-aware preprocessing
      │
      ▼
Relevance estimation
      │
      ▼
Relationship estimation
      │
      ▼
Weighted feature graph
      │
      ▼
Combinatorial subset objective
```

The repository preserves the generated graph and associated metadata under `results/`.

---

## QUBO and QAOA

### Why QUBO?

Feature selection with an explicit subset-size requirement naturally leads to binary decisions. Quadratic terms allow feature-to-feature interactions to enter the objective directly.

### Why QAOA?

QAOA is a hybrid variational algorithm designed for discrete optimization problems. Q-CyberSelect uses QAOA to sample candidate binary assignments and recover a selected feature subset.

The implementation uses Qiskit and keeps optimization outputs separate from downstream classifier metrics.

### Exact classical verification

For small controlled instances, exhaustive enumeration provides the true optimum:

```text
All feasible k-feature subsets
            │
            ▼
     Evaluate objective
            │
            ▼
       Exact optimum
            │
            ├─────────────┐
            ▼             ▼
          QAOA        Comparison
```

An approximation ratio of 1.0 means that the QAOA solution reached the exact objective value for that evaluated instance.

No quantum speedup claim is made from these experiments.

---

## Classical Baselines

QAOA is evaluated against four 4-feature classical selectors:

### Mutual Information

Ranks predictors according to their measured dependence with the binary target.

### ANOVA

Ranks predictors using the univariate F-statistic.

### L1-Regularized Logistic Regression

Uses coefficient sparsity to identify a compact subset.

### Random Forest

Uses tree-ensemble feature importance as a nonlinear selection baseline.

An all-feature model using all 42 predictors provides the reference configuration.

All four compact baselines use the same target cardinality:

\[
k=4.
\]

---

## Validation Strategy

The project uses a staged evaluation strategy.

### Development-stage analysis

The development experiments include:

- QAOA seed stability;
- 3-fold cross-validation;
- nested feature-selection evaluation;
- selector comparison;
- exploratory paired statistics;
- robustness calibration;
- pairwise robustness sensitivity.

### Locked evaluation

After the development configuration was frozen, the held-out UNSW-NB15 testing split was used for the final benchmark.

The final QAOA representation was fixed as:

```text
sttl
ct_state_ttl
dload
rate
```

The testing split was not used to select this feature set.

---

## Locked Test Results

<p align="center">
  <img src="docs/qcyberselect-results.svg" alt="Locked test results" width="100%">
</p>

| Method | Features | Reduction | Precision | Recall | F1 | Balanced Accuracy | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-42 | 42 | 0.00% | 0.816646 | 0.987625 | 0.894034 | 0.857974 | 0.980430 | 0.984265 | 0.755034 |
| QAOA-4 | 4 | **90.48%** | 0.772241 | 0.895593 | 0.829355 | 0.785985 | 0.850748 | 0.817297 | 0.592225 |
| MI-4 | 4 | 90.48% | 0.874278 | 0.945425 | 0.908461 | 0.889429 | 0.969565 | 0.969680 | 0.789362 |
| ANOVA-4 | 4 | 90.48% | 0.770608 | 0.943065 | 0.848158 | 0.799559 | 0.889986 | 0.872568 | 0.635692 |
| L1-4 | 4 | 90.48% | 0.754419 | 0.944388 | 0.838782 | 0.783870 | 0.814286 | 0.795371 | 0.610222 |
| RF-4 | 4 | 90.48% | 0.810377 | 0.959234 | 0.878545 | 0.842117 | 0.957890 | 0.963400 | 0.714414 |

### Interpretation

The locked test shows that all four-feature selectors substantially reduce dimensionality, but predictive performance differs across selection methods.

For QAOA specifically, the result demonstrates that a four-feature representation can be obtained through the graph-structured optimization pipeline, but the experiment does **not** establish predictive superiority over the classical baselines or the all-feature reference.

That distinction is central to interpreting the study correctly.

---

## Selected Feature Sets

| Selector | Features |
|---|---|
| **QAOA-4** | `sttl`, `ct_state_ttl`, `dload`, `rate` |
| **MI-4** | `dttl`, `dbytes`, `sttl`, `sbytes` |
| **ANOVA-4** | `ct_dst_sport_ltm`, `dload`, `ct_state_ttl`, `sttl` |
| **L1-4** | `dload`, `swin`, `proto`, `dttl` |
| **RF-4** | `rate`, `sload`, `ct_state_ttl`, `sttl` |

The selection overlap is itself informative: several methods repeatedly identify traffic-level and state-related variables, while their exact four-feature subsets differ.

---

## QAOA Stability

The seed-stability study evaluated five QAOA seeds.

Observed selection frequencies included:

| Feature | Selection frequency |
|---|---:|
| `ct_state_ttl` | 100% |
| `dttl` | 100% |
| `sttl` | 100% |
| `dload` | 80% |
| `state` | 20% |

Mean pairwise Jaccard similarity:

```text
0.84
```

Observed range:

```text
0.60 → 1.00
```

This measures selection consistency across QAOA runs; it is not a measure of quantum advantage.

---

## Cross-Validation and Nested Evaluation

The development-stage experiments include repeated feature selection inside validation folds.

Representative QAOA development metrics included:

```text
QAOA-4 CV mean F1:        0.939797
QAOA-4 CV F1 std:         0.000886
QAOA-4 CV mean balanced:  0.902120
QAOA-4 CV mean ROC-AUC:   0.960443
```

The nested evaluation was used to reduce the risk of reporting an optimistic result arising from selecting features with information from the evaluation fold.

These development-stage metrics should not be conflated with the final locked test metrics.

---

## Robustness Analysis

Robustness was evaluated using structured perturbations of predefined numeric traffic features at:

```text
5%
10%
20%
```

The final locked test robustness table is:

| Method | Perturbation | Clean F1 | Perturbed Mean F1 | Mean Degradation | Worst Degradation |
|---|---:|---:|---:|---:|---:|
| All-42 | 5% | 0.894034 | 0.885811 | 0.008223 | 0.009772 |
| All-42 | 10% | 0.894034 | 0.879632 | 0.014402 | 0.015028 |
| All-42 | 20% | 0.894034 | 0.868483 | 0.025552 | 0.025920 |
| QAOA-4 | 5% | 0.829355 | 0.822000 | 0.019670 | 0.039341 |
| QAOA-4 | 10% | 0.829355 | 0.803612 | 0.035042 | 0.070085 |
| QAOA-4 | 20% | 0.829355 | 0.820117 | 0.019541 | 0.039082 |
| MI-4 | 5% | 0.908461 | 0.456295 | 0.452166 | 0.576777 |
| MI-4 | 10% | 0.908461 | 0.603756 | 0.304705 | 0.475062 |
| MI-4 | 20% | 0.908461 | 0.674891 | 0.233570 | 0.333949 |
| ANOVA-4 | 5% | 0.848158 | 0.836138 | 0.012020 | 0.023732 |
| ANOVA-4 | 10% | 0.848158 | 0.836138 | 0.012020 | 0.023732 |
| ANOVA-4 | 20% | 0.848158 | 0.836101 | 0.012057 | 0.023806 |
| L1-4 | 5% | 0.838782 | 0.827725 | 0.011057 | 0.022113 |
| L1-4 | 10% | 0.838782 | 0.827725 | 0.011057 | 0.022113 |
| L1-4 | 20% | 0.838782 | 0.827725 | 0.011057 | 0.022113 |
| RF-4 | 5% | 0.878545 | 0.827418 | 0.051127 | 0.080588 |
| RF-4 | 10% | 0.878545 | 0.831002 | 0.047542 | 0.081497 |
| RF-4 | 20% | 0.878545 | 0.835603 | 0.042941 | 0.066371 |

### Important scope note

These are **controlled synthetic perturbation experiments**. They test sensitivity under a defined perturbation protocol and should not be interpreted as a comprehensive model of adaptive real-world attackers.

---

## Pairwise Robustness Ablation

The joint perturbation study evaluated all 28 pairs among eight predefined robustness features.

Only three pairs produced positive mean non-additive interaction under the final interaction criterion. The strongest interaction was associated with:

```text
sbytes + sload
```

A pairwise robustness-aware QUBO was then evaluated over:

```text
δ = 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0
```

Across the entire sweep:

```text
QAOA approximation ratio: 1.0
Exact subset: unchanged
QAOA subset: unchanged
```

Therefore the pairwise term is retained as a **robustness ablation**, not presented as an improved principal method.

---

## Why the Negative Results Matter

A research repository should preserve negative findings when they constrain the interpretation of the method.

The robustness-aware experiments showed that adding the tested robustness penalty did not automatically improve the selected representation. Rather than selecting a penalty solely because it produced a desired outcome, Q-CyberSelect records the observed behavior and keeps the robustness extension separate from the principal QAOA formulation.

This makes the reported conclusion narrower and more reproducible:

> QAOA is evaluated as a graph-structured combinatorial feature-selection mechanism; the current experiments do not establish a general predictive or robustness advantage over classical alternatives.

---

## Repository Structure

```text
Q-CyberSelect/
├── cpp/
│   └── baselines/
├── data/
│   └── unsw_nb15/
├── docs/
│   ├── qcyberselect-hero.svg
│   ├── qcyberselect-architecture.svg
│   └── qcyberselect-results.svg
├── frontend/
│   └── src/
├── notebooks/
├── paper/
│   ├── FINAL_EXPERIMENTAL_SUMMARY.md
│   └── ...
├── qiskit/
│   └── experiments/
├── results/
│   ├── FINAL_RESULTS_MASTER.json
│   ├── FINAL_TABLE_SELECTED_FEATURES.csv
│   ├── FINAL_TABLE_TEST_BENCHMARK.csv
│   ├── FINAL_TABLE_TEST_ROBUSTNESS.csv
│   └── ...
├── rust/
│   └── security_pipeline/
├── scripts/
├── PROJECT.md
└── requirements.txt
```

### Language responsibilities

| Language | Responsibility |
|---|---|
| **Python** | Data preparation, feature modeling, QAOA, validation and ML evaluation |
| **C++** | Exact classical optimization and benchmarking |
| **Rust** | Security-oriented preprocessing and network-flow components |
| **TypeScript** | Results presentation and visualization layer |

Each language is used for a distinct project role rather than as a decorative addition.

---

## Reproducibility

### Requirements

The main experimental environment uses:

- Python 3.13
- Qiskit 2.5.2
- Qiskit Machine Learning 0.9.1
- Qiskit Optimization
- Qiskit Aer
- qiskit-algorithms 0.4.0
- NumPy
- pandas
- SciPy
- scikit-learn

The repository also contains C++, Rust and TypeScript components.

### Environment setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset placement

Place the pre-split UNSW-NB15 files at:

```text
data\unsw_nb15\UNSW_NB15_training-set.csv
data\unsw_nb15\UNSW_NB15_testing-set.csv
```

### Core graph experiment

```powershell
python qiskit\experiments\build_cyber_feature_graph.py
```

### Development validation

```powershell
python qiskit\experiments\qaoa_cv_stability.py
python qiskit\experiments\qaoa_nested_cv.py
python qiskit\experiments\nested_selector_comparison.py
python qiskit\experiments\qaoa_paired_statistics.py
```

### Robustness experiments

```powershell
python qiskit\experiments\structured_feature_perturbation.py
python qiskit\experiments\joint_pairwise_robustness_complete.py
python qiskit\experiments\pairwise_robust_qaoa_sweep.py
```

### Final locked evaluation

```powershell
python qiskit\experiments\final_locked_test_benchmark.py
python qiskit\experiments\final_locked_test_robustness.py
```

### Consolidated result manifest

```powershell
python qiskit\experiments\build_final_results_pack.py
```

The consolidated files are:

```text
results/FINAL_RESULTS_MASTER.json
results/FINAL_TABLE_SELECTED_FEATURES.csv
results/FINAL_TABLE_TEST_BENCHMARK.csv
results/FINAL_TABLE_TEST_ROBUSTNESS.csv
paper/FINAL_EXPERIMENTAL_SUMMARY.md
```

---

## Reproducibility Principles

### Frozen evaluation

Feature-selection configurations used for the held-out benchmark are frozen before reading the test labels.

### Exact verification

Small optimization instances are checked against exhaustive classical enumeration whenever feasible.

### Shared classifier

Feature selectors are compared using the same Random Forest evaluation family so that the comparison focuses primarily on the representation.

### Explicit negative findings

Failed or non-improving robustness extensions are retained as ablations rather than removed from the research record.

### Artifact preservation

Raw experiment outputs, summary JSON files, final tables and methodological summaries are versioned under `results/` and `paper/`.

---

## Limitations

### No quantum speedup claim

The evaluated optimization instances are small enough for exact classical verification in the controlled experiments. Consequently, the repository does not establish a computational quantum advantage.

### Single-dataset evaluation

The locked benchmark is based on UNSW-NB15. Additional datasets are required to establish cross-dataset generalization.

### Robustness protocol

The perturbations are synthetic and structured. They provide controlled sensitivity measurements but do not represent the full space of adversarial traffic manipulation.

### Predictive performance

The locked test results do not show predictive superiority for QAOA over all classical selectors. The project should therefore be interpreted as a study of quantum-optimization-based feature selection rather than as a demonstrated replacement for classical selection methods.

### Scale

Exact verification is practical only for relatively small optimization instances. Larger candidate graphs require different validation strategies and may expose the scaling limitations of classical enumeration.

---

## Research Status

| Component | Status |
|---|---|
| Literature-grounded formulation | Complete |
| UNSW-NB15 pipeline | Complete |
| Feature graph construction | Complete |
| QAOA optimization experiments | Complete |
| Exact optimization validation | Complete for tractable instances |
| Classical selector baselines | Complete |
| Seed stability | Complete |
| Cross-validation | Complete |
| Nested evaluation | Complete |
| Statistical analysis | Complete / exploratory |
| Structured robustness | Complete |
| Pairwise robustness ablation | Complete |
| Locked test benchmark | Complete |
| Locked test robustness | Complete |
| Final result manifest | Complete |
| Publication manuscript | In progress |

---

## Citation

### Graph-theoretic QAOA formulation

Y. Li et al., “Implementing Graph-Theoretic Feature Selection by Quantum Approximate Optimization Algorithm,” *IEEE Transactions on Neural Networks and Learning Systems*, 2024.  
DOI: https://doi.org/10.1109/TNNLS.2022.3190042

### Dataset

N. Moustafa and J. Slay, **UNSW-NB15: a comprehensive data set for network intrusion detection systems**, University of New South Wales.

---

## Acknowledgment of Scope

Q-CyberSelect is an experimental research implementation. The repository is intended to make the formulation, validation strategy, results and limitations inspectable and reproducible.

The project makes no claim that the evaluated QAOA configuration is universally optimal, that four features are sufficient for production intrusion detection, or that the present implementation demonstrates quantum computational advantage.

---

<div align="center">

**Q-CyberSelect**  
*Graph-structured optimization for compact network intrusion detection.*

</div>
