# IoT NIDS Live Replay

A simulation of H. Hakami, M. Faheem, M. B. Ahmad, “Machine Learning Techniques for Enhanced Intrusion Detection in IoT Security,” *IEEE Access*, vol. 13, pp. 31140–31158, 2025 (doi:10.1109/ACCESS.2025.3542227).

**Live simulator:** [link added after deployment]

The paper’s best model, a Random Forest (50 trees, no depth limit), is retrained on its three datasets. The held-out test flows are then streamed through it as if they were live IoT traffic, in a Streamlit dashboard with a traffic feed (caught / missed / false alarm), live KPIs, a running confusion matrix, per-class detection rates and a “paper vs. our run” table. A zero-day model (WSN-DS trained without Flooding) shows what happens to an attack the model never saw.

## Results: the paper vs. this simulation

| Dataset | Accuracy, paper (Table 6) | Accuracy, ours (5 seeds) |
|---|---|---|
| WSN-DS | 99.71 | 99.71 ± 0.02 |
| UNSW-NB15 | 90.17 | 83.56 ± 0.17 |
| CIC-IDS2017 (Friday DDoS) | 99.99 | 99.99 ± 0.00 |

Extensions (all in `experiments.py`, results in `results/`):

1. **Leakage audit:** keeping duplicates and oversampling before the split add up to 3.6 pp on UNSW-NB15, but no protocol reaches the paper’s 90.17%.
2. **Label-conflict ceiling and better models:** 77% of UNSW-NB15 DoS flows have an identical Exploits/Fuzzers twin (DoS recall can be at most 57.9%); HistGradientBoosting beats the Random Forest (+1.0 pp accuracy, +2.1 pp macro-F1).
3. **Practical tests:** an unseen attack can vanish (WSN-DS Flooding: 99.1% → 0.4% detected); the CIC-IDS2017 DDoS model detects at most 0.12% of other days’ DoS attacks; 5-tree models are 14–28× smaller and lose at most 0.26 pp macro-F1.

## Pipeline

Clean + remove exact duplicates → stratified 80/20 split → label encoder, min-max scaler and Pearson-correlation filter (|r| > 0.97) fitted on the training split only → Random Forest → evaluate on the test split.

## Run it

```bash
pip install -r requirements.txt
python -m streamlit run app.py          # simulator, http://localhost:8501 (uses models/*.joblib)
```

Retrain and rerun the experiments (needs the datasets, see `data/README.md`):

```bash
python train.py --dataset wsnds --csv data/WSN-DS.csv
python train.py --dataset unsw --csv data/UNSW_NB15_training-set.csv data/UNSW_NB15_testing-set.csv
python train.py --dataset cic --csv data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
python train.py --dataset wsnds --csv data/WSN-DS.csv --holdout Flooding   # zero-day model
python experiments.py all     # all extensions, about 45 min on 8 CPU cores; tables and figures in results/
python experiments.py figs    # redraw the figures only
```

Tested with Python 3.10 and the package versions in `requirements.txt`. The models in `models/` must be loaded with scikit-learn 1.7.2.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit live-replay simulator |
| `nids_core.py` | Shared preprocessing (dataset presets, cleaning, transform) |
| `train.py` | Trains the paper’s Random Forest and saves a model bundle (`--holdout` for a zero-day model) |
| `experiments.py` | All extension experiments; writes CSV/Markdown tables and figures to `results/` |
| `models/` | Trained model bundles used by the simulator (they include the held-out test flows that are replayed) |
| `results/` | Tables and figures from the experiment run |

## References

- Hakami, Faheem, Ahmad, IEEE Access, 2025 (the simulated paper).
- I. Almomani, B. Al-Kasasbeh, M. Al-Akhras, “WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks,” *Journal of Sensors*, 2016.
- N. Moustafa, J. Slay, “UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems,” MilCIS, 2015.
- I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP, 2018.
- N. V. Chawla et al., “SMOTE: Synthetic Minority Over-sampling Technique,” *JAIR*, vol. 16, 2002 (implemented in `experiments.py`).

All code in this repository is our own.
