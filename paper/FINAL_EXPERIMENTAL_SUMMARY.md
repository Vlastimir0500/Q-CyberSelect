# Q-CyberSelect — Final Experimental Summary

## 1. Study Overview

**Project:** Q-CyberSelect

**Study title:** Quantum Graph-Theoretic Feature Selection for Robust Network Intrusion Detection

**Dataset:** UNSW-NB15

**Predictor space:** 42 predictors after excluding `id`, `attack_cat`, and `label`.

The study evaluates graph-theoretic feature selection formulated as a binary quadratic optimization problem and solved using the Quantum Approximate Optimization Algorithm (QAOA). The experimental program includes classical feature-selection baselines, exact optimization checks, cross-validation, selection stability analysis, statistical comparison, and structured perturbation robustness analysis.

## 2. Frozen QAOA Configuration

The QAOA feature set used for the locked held-out test evaluation was:

`sttl`, `ct_state_ttl`, `dload`, `rate`

The selected representation contains four predictors, corresponding to **90.47619% feature reduction** relative to the 42-feature predictor space.

## 3. Locked Held-Out Test Evaluation

The held-out UNSW-NB15 testing split was evaluated after the feature-selection configurations had been frozen during development.

| Method | Features | Reduction (%) | Precision | Recall | F1 | Balanced Accuracy | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-42 | 42 | 0.000000 | 0.816646 | 0.987625 | 0.894034 | 0.857974 | 0.980430 | 0.984265 | 0.755034 |
| QAOA-4 | 4 | 90.476190 | 0.772241 | 0.895593 | 0.829355 | 0.785985 | 0.850748 | 0.817297 | 0.592225 |
| MI-4 | 4 | 90.476190 | 0.874278 | 0.945425 | 0.908461 | 0.889429 | 0.969565 | 0.969680 | 0.789362 |
| ANOVA-4 | 4 | 90.476190 | 0.770608 | 0.943065 | 0.848158 | 0.799559 | 0.889986 | 0.872568 | 0.635692 |
| L1-4 | 4 | 90.476190 | 0.754419 | 0.944388 | 0.838782 | 0.783870 | 0.814286 | 0.795371 | 0.610222 |
| RF-4 | 4 | 90.476190 | 0.810377 | 0.959234 | 0.878545 | 0.842117 | 0.957890 | 0.963400 | 0.714414 |

The locked test results do not support a claim of predictive superiority for QAOA relative to the evaluated classical feature-selection baselines. The principal empirical outcome is dimensionality reduction through a quantum-optimization-based feature-selection formulation.

## 4. Selected Feature Sets

**All-42:** dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports
**QAOA-4:** sttl, ct_state_ttl, dload, rate
**MI-4:** dttl, dbytes, sttl, sbytes
**ANOVA-4:** ct_dst_sport_ltm, dload, ct_state_ttl, sttl
**L1-4:** dload, swin, proto, dttl
**RF-4:** rate, sload, ct_state_ttl, sttl

## 5. Robustness Evaluation

Structured perturbations were applied to the predefined robustness-feature set. Perturbations were evaluated at 5%, 10%, and 20% levels where the corresponding features were present in each model.

| Method | Perturbation | Clean F1 | Perturbed Mean F1 | Mean Degradation | Worst Degradation |
|---|---:|---:|---:|---:|---:|
| All-42 | 0.050000 | 0.894034 | 0.885811 | 0.008223 | 0.009772 |
| All-42 | 0.100000 | 0.894034 | 0.879632 | 0.014402 | 0.015028 |
| All-42 | 0.200000 | 0.894034 | 0.868483 | 0.025552 | 0.025920 |
| QAOA-4 | 0.050000 | 0.829355 | 0.822000 | 0.019670 | 0.039341 |
| QAOA-4 | 0.100000 | 0.829355 | 0.803612 | 0.035042 | 0.070085 |
| QAOA-4 | 0.200000 | 0.829355 | 0.820117 | 0.019541 | 0.039082 |
| MI-4 | 0.050000 | 0.908461 | 0.456295 | 0.452166 | 0.576777 |
| MI-4 | 0.100000 | 0.908461 | 0.603756 | 0.304705 | 0.475062 |
| MI-4 | 0.200000 | 0.908461 | 0.674891 | 0.233570 | 0.333949 |
| ANOVA-4 | 0.050000 | 0.848158 | 0.836138 | 0.012020 | 0.023732 |
| ANOVA-4 | 0.100000 | 0.848158 | 0.836138 | 0.012020 | 0.023732 |
| ANOVA-4 | 0.200000 | 0.848158 | 0.836101 | 0.012057 | 0.023806 |
| L1-4 | 0.050000 | 0.838782 | 0.827725 | 0.011057 | 0.022113 |
| L1-4 | 0.100000 | 0.838782 | 0.827725 | 0.011057 | 0.022113 |
| L1-4 | 0.200000 | 0.838782 | 0.827725 | 0.011057 | 0.022113 |
| RF-4 | 0.050000 | 0.878545 | 0.827418 | 0.051127 | 0.080588 |
| RF-4 | 0.100000 | 0.878545 | 0.831002 | 0.047542 | 0.081497 |
| RF-4 | 0.200000 | 0.878545 | 0.835603 | 0.042941 | 0.066371 |

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
