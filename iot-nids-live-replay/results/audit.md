# Experiment 1: evaluation-protocol audit

## Results, mean ± std over 5 seed(s): 42, 43, 44, 45, 46

| dataset | protocol | paper accuracy | accuracy | f1_weighted | f1_macro | test rows identical to a train row (%) | DoS recall |
|---|---|---|---|---|---|---|---|
| WSN-DS | A | 99.71 | 99.70 ± 0.02 | 99.70 ± 0.02 | 98.05 ± 0.10 | 3.70 ± 0.05 |  |
| WSN-DS | B | 99.71 | 99.97 ± 0.00 | 99.97 ± 0.00 | 99.97 ± 0.00 | 80.74 ± 0.01 |  |
| WSN-DS | C | 99.71 | 99.71 ± 0.02 | 99.70 ± 0.02 | 98.16 ± 0.14 | 0.00 ± 0.00 |  |
| WSN-DS | D | 99.71 | 99.71 ± 0.02 | 99.70 ± 0.02 | 98.16 ± 0.14 | 0.00 ± 0.00 |  |
| WSN-DS | D+OS | 99.71 | 99.69 ± 0.02 | 99.69 ± 0.02 | 98.06 ± 0.13 | 0.00 ± 0.00 |  |
| UNSW-NB15 | A | 90.17 | 84.37 ± 0.15 | 83.88 ± 0.16 | 74.30 ± 0.23 | 45.09 ± 0.14 | 25.04 ± 0.94 |
| UNSW-NB15 | B | 90.17 | 87.19 ± 0.10 | 87.49 ± 0.10 | 87.49 ± 0.10 | 82.24 ± 0.09 | 95.62 ± 0.27 |
| UNSW-NB15 | C | 90.17 | 83.56 ± 0.17 | 83.03 ± 0.17 | 71.05 ± 0.28 | 5.07 ± 0.11 | 20.75 ± 0.86 |
| UNSW-NB15 | D | 90.17 | 83.56 ± 0.17 | 83.03 ± 0.17 | 71.06 ± 0.28 | 5.07 ± 0.11 | 20.78 ± 0.90 |
| UNSW-NB15 | D+OS | 90.17 | 83.18 ± 0.12 | 83.40 ± 0.13 | 71.35 ± 0.25 | 5.07 ± 0.11 | 24.93 ± 1.44 |
| UNSW-NB15 | E | 90.17 | 76.71 ± 0.11 | 79.07 ± 0.10 | 68.45 ± 0.15 | 2.22 ± 0.00 | 25.79 ± 0.32 |
| CIC-IDS2017 (DDoS) | A | 99.99 | 99.99 ± 0.00 | 99.99 ± 0.00 | 99.99 ± 0.00 | 1.61 ± 0.07 |  |
| CIC-IDS2017 (DDoS) | B | 99.99 | 99.99 ± 0.00 | 99.99 ± 0.00 | 99.99 ± 0.00 | 19.63 ± 0.08 |  |
| CIC-IDS2017 (DDoS) | C | 99.99 | 99.99 ± 0.00 | 99.99 ± 0.00 | 99.99 ± 0.00 | 0.05 ± 0.01 |  |
| CIC-IDS2017 (DDoS) | D | 99.99 | 99.99 ± 0.00 | 99.99 ± 0.00 | 99.99 ± 0.00 | 0.05 ± 0.01 |  |
| CIC-IDS2017 (DDoS) | D+OS | 99.99 | 99.99 ± 0.00 | 99.99 ± 0.00 | 99.99 ± 0.00 | 0.05 ± 0.01 |  |

## Protocols

| Protocol | Meaning |
|---|---|
| A | No dedupe; scaler + PCC filter fitted on all data; then stratified 80/20 split |
| B | Protocol A plus random oversampling of every class to the majority size BEFORE the split |
| C | Dedupe; scaler + PCC filter fitted on all data; then split |
| D | Dedupe; split; scaler + PCC filter fitted on the training split only (our pipeline) |
| D+OS | Protocol D plus random oversampling of the training split only |
| E | UNSW-NB15's official training and testing files (UNSW-NB15 only) |

## Reference

Paper's UNSW-NB15 accuracy after oversampling: 98.8%.
