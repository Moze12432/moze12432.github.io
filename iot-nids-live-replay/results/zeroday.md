# Experiment 3a: detecting attack types the model never saw

## Share of each attack's test flows that raised an alarm (%), mean over 5 seed(s): 42, 43, 44, 45, 46

'seen' = normal training with all classes; 'unseen_*' = that attack removed from training (multi-class RF as in the paper, or a binary normal-vs-attack RF).

| dataset | attack | seen | seen_std | unseen_multiclass | unseen_multiclass_std | unseen_binary | unseen_binary_std | false_alarm_multiclass | false_alarm_binary |
|---|---|---|---|---|---|---|---|---|---|
| WSN-DS | Blackhole | 100.00 | 0.00 | 83.00 | 1.25 | 89.72 | 4.19 | 0.10 | 0.11 |
| WSN-DS | Flooding | 99.08 | 0.26 | 0.38 | 0.36 | 0.54 | 0.53 | 0.04 | 0.06 |
| WSN-DS | Grayhole | 99.23 | 0.17 | 40.58 | 0.73 | 40.75 | 0.80 | 0.06 | 0.06 |
| WSN-DS | TDMA | 93.02 | 0.51 | 32.90 | 2.94 | 52.34 | 1.78 | 0.09 | 0.10 |
| UNSW-NB15 | DoS | 97.49 | 0.56 | 96.62 | 0.65 | 98.47 | 0.24 | 6.22 | 7.15 |
| UNSW-NB15 | Exploits | 97.47 | 0.24 | 93.07 | 0.98 | 96.05 | 0.35 | 5.87 | 6.51 |
| UNSW-NB15 | Fuzzers | 69.75 | 0.60 | 16.75 | 0.38 | 20.09 | 0.51 | 0.93 | 1.41 |
| UNSW-NB15 | Generic | 99.45 | 0.12 | 77.59 | 10.32 | 98.50 | 1.27 | 6.21 | 7.49 |
| UNSW-NB15 | Reconnaissance | 99.62 | 0.10 | 68.80 | 3.50 | 83.69 | 3.24 | 6.26 | 7.14 |

## Where the unseen flows went (multi-class RF, seed 42)

| dataset | unseen | predicted_as | pct |
|---|---|---|---|
| WSN-DS | Blackhole | Grayhole | 84.18 |
| WSN-DS | Blackhole | Normal | 15.27 |
| WSN-DS | Blackhole | TDMA | 0.55 |
| WSN-DS | Flooding | Normal | 99.05 |
| WSN-DS | Flooding | Grayhole | 0.79 |
| WSN-DS | Flooding | Blackhole | 0.16 |
| WSN-DS | Grayhole | Normal | 59.92 |
| WSN-DS | Grayhole | Blackhole | 38.89 |
| WSN-DS | Grayhole | TDMA | 1.19 |
| WSN-DS | TDMA | Normal | 66.24 |
| WSN-DS | TDMA | Blackhole | 28.71 |
| WSN-DS | TDMA | Grayhole | 5.05 |
| UNSW-NB15 | DoS | Exploits | 69.00 |
| UNSW-NB15 | DoS | Fuzzers | 15.64 |
| UNSW-NB15 | DoS | Reconnaissance | 9.00 |
| UNSW-NB15 | DoS | Generic | 3.64 |
| UNSW-NB15 | DoS | Normal | 2.73 |
| UNSW-NB15 | Exploits | DoS | 67.20 |
| UNSW-NB15 | Exploits | Reconnaissance | 13.12 |
| UNSW-NB15 | Exploits | Normal | 7.64 |
| UNSW-NB15 | Exploits | Fuzzers | 6.36 |
| UNSW-NB15 | Exploits | Generic | 5.69 |
| UNSW-NB15 | Fuzzers | Normal | 83.02 |
| UNSW-NB15 | Fuzzers | Exploits | 10.78 |
| UNSW-NB15 | Fuzzers | DoS | 3.32 |
| UNSW-NB15 | Fuzzers | Reconnaissance | 2.41 |
| UNSW-NB15 | Fuzzers | Generic | 0.48 |
| UNSW-NB15 | Generic | Exploits | 40.26 |
| UNSW-NB15 | Generic | Fuzzers | 39.28 |
| UNSW-NB15 | Generic | Normal | 15.39 |
| UNSW-NB15 | Generic | DoS | 3.75 |
| UNSW-NB15 | Generic | Reconnaissance | 1.32 |
| UNSW-NB15 | Reconnaissance | Exploits | 31.53 |
| UNSW-NB15 | Reconnaissance | Fuzzers | 29.38 |
| UNSW-NB15 | Reconnaissance | Normal | 28.18 |
| UNSW-NB15 | Reconnaissance | DoS | 10.06 |
| UNSW-NB15 | Reconnaissance | Generic | 0.85 |
