# Experiment 2: label-conflict ceiling and minority-attack fixes

## Label conflicts (identical features, different labels)

Rows that share their exact feature vector with a row of another class. 'accuracy_ceiling' is the best accuracy any classifier could reach on that data; 'recall_ceiling' is the best possible recall per class (before dedupe only, where it is well defined).

| dataset | stage | cls | rows | conflicting_pct | accuracy_ceiling | recall_ceiling | shares_vectors_with |
|---|---|---|---|---|---|---|---|
| UNSW-NB15 | before dedupe | ALL | 250982 | 14.46 | 92.78 |  |  |
| UNSW-NB15 | before dedupe | DoS | 16353 | 77.40 |  | 57.92 | Exploits (77.0% of rows), Fuzzers (76.1% of rows) |
| UNSW-NB15 | before dedupe | Exploits | 44525 | 37.04 |  | 88.26 | DoS (36.6% of rows), Fuzzers (36.4% of rows) |
| UNSW-NB15 | before dedupe | Fuzzers | 24246 | 14.91 |  | 87.63 | Exploits (11.4% of rows), DoS (11.2% of rows) |
| UNSW-NB15 | before dedupe | Generic | 58871 | 0.63 |  | 99.37 | Exploits (0.6% of rows), DoS (0.6% of rows) |
| UNSW-NB15 | before dedupe | Normal | 93000 | 1.00 |  | 99.54 | Fuzzers (1.0% of rows) |
| UNSW-NB15 | before dedupe | Reconnaissance | 13987 | 16.04 |  | 84.12 | Exploits (15.7% of rows), DoS (15.4% of rows) |
| UNSW-NB15 | after dedupe | ALL | 157206 | 5.31 | 96.23 |  |  |
| UNSW-NB15 | after dedupe | DoS | 5500 | 34.02 |  |  | Exploits (33.5% of rows), Fuzzers (31.7% of rows) |
| UNSW-NB15 | after dedupe | Exploits | 27434 | 7.18 |  |  | DoS (6.7% of rows), Fuzzers (6.5% of rows) |
| UNSW-NB15 | after dedupe | Fuzzers | 20960 | 10.53 |  |  | Exploits (8.5% of rows), DoS (8.3% of rows) |
| UNSW-NB15 | after dedupe | Generic | 7599 | 4.15 |  |  | Exploits (4.0% of rows), DoS (3.8% of rows) |
| UNSW-NB15 | after dedupe | Normal | 85722 | 0.48 |  |  | Fuzzers (0.5% of rows) |
| UNSW-NB15 | after dedupe | Reconnaissance | 9991 | 15.71 |  |  | Exploits (15.5% of rows), DoS (15.0% of rows) |
| WSN-DS | before dedupe | ALL | 374661 | 0.00 | 100.00 |  |  |
| WSN-DS | before dedupe | Blackhole | 10049 | 0.00 |  | 100.00 |  |
| WSN-DS | before dedupe | Flooding | 3312 | 0.00 |  | 100.00 |  |
| WSN-DS | before dedupe | Grayhole | 14596 | 0.00 |  | 100.00 |  |
| WSN-DS | before dedupe | Normal | 340066 | 0.00 |  | 100.00 | TDMA (0.0% of rows) |
| WSN-DS | before dedupe | TDMA | 6638 | 0.05 |  | 99.95 | Normal (0.0% of rows) |
| WSN-DS | after dedupe | ALL | 365788 | 0.00 | 100.00 |  |  |
| WSN-DS | after dedupe | Blackhole | 10049 | 0.00 |  |  |  |
| WSN-DS | after dedupe | Flooding | 3157 | 0.00 |  |  |  |
| WSN-DS | after dedupe | Grayhole | 13909 | 0.00 |  |  |  |
| WSN-DS | after dedupe | Normal | 332040 | 0.00 |  |  | TDMA (0.0% of rows) |
| WSN-DS | after dedupe | TDMA | 6633 | 0.05 |  |  | Normal (0.0% of rows) |

## UNSW-NB15: methods, mean ± std over 5 seed(s): 42, 43, 44, 45, 46

| method | accuracy | f1_weighted | f1_macro | detection_rate | false_alarm_rate | recall[DoS] | recall[Exploits] | recall[Fuzzers] | recall[Generic] | recall[Normal] | recall[Reconnaissance] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF (paper setting) | 83.56 ± 0.17 | 83.03 ± 0.17 | 71.06 ± 0.28 | 89.85 ± 0.21 | 6.29 ± 0.28 | 20.78 ± 0.90 | 85.69 ± 0.47 | 59.66 ± 0.62 | 84.96 ± 0.86 | 93.71 ± 0.28 | 74.25 ± 0.75 |
| RF + class weights | 83.50 ± 0.18 | 83.20 ± 0.18 | 70.41 ± 0.29 | 89.06 ± 0.16 | 5.84 ± 0.26 | 20.78 ± 1.45 | 85.77 ± 0.52 | 57.45 ± 0.36 | 84.91 ± 0.76 | 94.16 ± 0.26 | 73.96 ± 1.26 |
| RF + random oversampling | 83.18 ± 0.12 | 83.40 ± 0.13 | 71.35 ± 0.25 | 92.35 ± 0.19 | 9.07 ± 0.10 | 24.93 ± 1.44 | 85.62 ± 0.35 | 66.50 ± 0.58 | 85.92 ± 0.87 | 90.93 ± 0.10 | 74.92 ± 0.89 |
| RF + SMOTE | 82.56 ± 0.17 | 82.85 ± 0.18 | 71.80 ± 0.29 | 92.47 ± 0.14 | 9.34 ± 0.34 | 31.65 ± 1.44 | 80.80 ± 0.28 | 66.13 ± 0.53 | 86.50 ± 0.67 | 90.66 ± 0.34 | 77.42 ± 0.95 |
| HistGradientBoosting | 84.60 ± 0.20 | 83.91 ± 0.21 | 73.14 ± 0.43 | 88.82 ± 0.57 | 5.18 ± 0.39 | 22.93 ± 1.38 | 86.72 ± 0.50 | 59.58 ± 1.40 | 86.95 ± 1.07 | 94.82 ± 0.39 | 75.79 ± 0.52 |
| HGB + class weights | 79.73 ± 0.58 | 81.67 ± 0.51 | 74.52 ± 0.50 | 98.38 ± 0.18 | 19.99 ± 0.92 | 71.42 ± 1.78 | 73.29 ± 0.80 | 85.19 ± 0.64 | 88.80 ± 0.60 | 80.01 ± 0.92 | 81.27 ± 0.46 |
| Two-stage RF | 83.39 ± 0.12 | 83.25 ± 0.13 | 70.48 ± 0.27 | 90.57 ± 0.21 | 6.88 ± 0.10 | 20.91 ± 1.15 | 86.28 ± 0.23 | 60.27 ± 0.58 | 85.16 ± 0.68 | 93.12 ± 0.10 | 73.54 ± 0.87 |

## WSN-DS: methods, mean ± std over 5 seed(s): 42, 43, 44, 45, 46

| method | accuracy | f1_weighted | f1_macro | detection_rate | false_alarm_rate | recall[Normal] | recall[Blackhole] | recall[Flooding] | recall[Grayhole] | recall[TDMA] |
|---|---|---|---|---|---|---|---|---|---|---|
| RF (paper setting) | 99.71 ± 0.02 | 99.70 ± 0.02 | 98.16 ± 0.14 | 98.23 ± 0.15 | 0.10 ± 0.01 | 99.90 ± 0.01 | 99.62 ± 0.10 | 99.08 ± 0.26 | 98.51 ± 0.21 | 92.87 ± 0.58 |
| RF + class weights | 99.70 ± 0.01 | 99.70 ± 0.01 | 98.13 ± 0.13 | 98.16 ± 0.11 | 0.09 ± 0.01 | 99.91 ± 0.01 | 99.59 ± 0.06 | 98.92 ± 0.56 | 98.38 ± 0.12 | 92.81 ± 0.56 |
| RF + random oversampling | 99.69 ± 0.02 | 99.69 ± 0.02 | 98.06 ± 0.13 | 98.40 ± 0.15 | 0.14 ± 0.01 | 99.86 ± 0.01 | 99.59 ± 0.13 | 99.52 ± 0.25 | 98.83 ± 0.16 | 93.05 ± 0.67 |
| RF + SMOTE | 99.64 ± 0.02 | 99.64 ± 0.02 | 97.77 ± 0.16 | 98.60 ± 0.10 | 0.21 ± 0.02 | 99.79 ± 0.02 | 99.65 ± 0.17 | 99.78 ± 0.21 | 99.08 ± 0.13 | 93.29 ± 0.64 |
| HistGradientBoosting | 99.59 ± 0.03 | 99.59 ± 0.03 | 97.20 ± 0.27 | 97.71 ± 0.18 | 0.14 ± 0.03 | 99.86 ± 0.03 | 99.30 ± 0.10 | 93.47 ± 1.53 | 98.30 ± 0.22 | 92.52 ± 0.68 |
| HGB + class weights | 99.44 ± 0.03 | 99.44 ± 0.03 | 96.78 ± 0.14 | 98.80 ± 0.11 | 0.45 ± 0.03 | 99.55 ± 0.03 | 99.65 ± 0.09 | 99.68 ± 0.27 | 99.02 ± 0.10 | 94.02 ± 0.68 |
| Two-stage RF | 99.70 ± 0.02 | 99.70 ± 0.02 | 98.13 ± 0.15 | 98.24 ± 0.16 | 0.10 ± 0.01 | 99.90 ± 0.01 | 99.58 ± 0.10 | 99.11 ± 0.24 | 98.50 ± 0.25 | 92.84 ± 0.61 |
