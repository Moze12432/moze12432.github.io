# Experiment 3b: CIC-IDS2017 DDoS detector vs. other days and attack types

## Flows that raised an alarm (%), mean over 5 seed(s): 42, 43, 44, 45, 46

For BENIGN rows this is the false-alarm rate; for attack rows the detection rate.

| model | test_set | label | n | flagged_pct | flagged_std | meaning |
|---|---|---|---|---|---|---|
| Friday DDoS only (paper setup) | Friday PM (DDoS), same day | BENIGN | 19014.00 | 0.01 | 0.01 | false-alarm rate |
| Friday DDoS only (paper setup) | Friday PM (DDoS), same day | DDoS | 25603.00 | 99.99 | 0.01 | detection rate |
| Friday DDoS only (paper setup) | Wednesday (DoS), held-out 20% | BENIGN | 83348.00 | 0.02 | 0.01 | false-alarm rate |
| Friday DDoS only (paper setup) | Wednesday (DoS), held-out 20% | DoS GoldenEye | 2057.00 | 0.00 | 0.00 | detection rate |
| Friday DDoS only (paper setup) | Wednesday (DoS), held-out 20% | DoS Hulk | 34569.00 | 0.12 | 0.02 | detection rate |
| Friday DDoS only (paper setup) | Wednesday (DoS), held-out 20% | DoS Slowhttptest | 1046.00 | 0.02 | 0.04 | detection rate |
| Friday DDoS only (paper setup) | Wednesday (DoS), held-out 20% | DoS slowloris | 1077.00 | 0.00 | 0.00 | detection rate |
| Friday DDoS only (paper setup) | Wednesday (DoS), held-out 20% | Heartbleed | 2.00 | 0.00 | 0.00 | detection rate |
| Friday DDoS only (paper setup) | Friday PM (PortScan) | BENIGN | 123083.00 | 0.01 | 0.00 | false-alarm rate |
| Friday DDoS only (paper setup) | Friday PM (PortScan) | PortScan | 90694.00 | 0.05 | 0.02 | detection rate |
| Friday DDoS only (paper setup) | Tuesday (brute force) | BENIGN | 412476.00 | 0.00 | 0.00 | false-alarm rate |
| Friday DDoS only (paper setup) | Tuesday (brute force) | FTP-Patator | 5931.00 | 0.00 | 0.00 | detection rate |
| Friday DDoS only (paper setup) | Tuesday (brute force) | SSH-Patator | 3219.00 | 0.00 | 0.00 | detection rate |
| Friday DDoS + Wednesday DoS | Friday PM (DDoS), same day | BENIGN | 19014.00 | 0.02 | 0.02 | false-alarm rate |
| Friday DDoS + Wednesday DoS | Friday PM (DDoS), same day | DDoS | 25603.00 | 99.99 | 0.00 | detection rate |
| Friday DDoS + Wednesday DoS | Wednesday (DoS), held-out 20% | BENIGN | 83348.00 | 0.04 | 0.01 | false-alarm rate |
| Friday DDoS + Wednesday DoS | Wednesday (DoS), held-out 20% | DoS GoldenEye | 2057.00 | 99.88 | 0.07 | detection rate |
| Friday DDoS + Wednesday DoS | Wednesday (DoS), held-out 20% | DoS Hulk | 34569.00 | 99.91 | 0.01 | detection rate |
| Friday DDoS + Wednesday DoS | Wednesday (DoS), held-out 20% | DoS Slowhttptest | 1046.00 | 99.85 | 0.09 | detection rate |
| Friday DDoS + Wednesday DoS | Wednesday (DoS), held-out 20% | DoS slowloris | 1077.00 | 99.70 | 0.12 | detection rate |
| Friday DDoS + Wednesday DoS | Wednesday (DoS), held-out 20% | Heartbleed | 2.00 | 100.00 | 0.00 | detection rate |
| Friday DDoS + Wednesday DoS | Friday PM (PortScan) | BENIGN | 123083.00 | 0.04 | 0.00 | false-alarm rate |
| Friday DDoS + Wednesday DoS | Friday PM (PortScan) | PortScan | 90694.00 | 0.16 | 0.00 | detection rate |
| Friday DDoS + Wednesday DoS | Tuesday (brute force) | BENIGN | 412476.00 | 0.28 | 0.01 | false-alarm rate |
| Friday DDoS + Wednesday DoS | Tuesday (brute force) | FTP-Patator | 5931.00 | 0.14 | 0.27 | detection rate |
| Friday DDoS + Wednesday DoS | Tuesday (brute force) | SSH-Patator | 3219.00 | 0.35 | 0.22 | detection rate |
