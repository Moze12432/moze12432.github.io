# IoT NIDS Live Replay

A reproduction and live demo of the Random Forest intrusion-detection pipeline from:

> H. Hakami, M. Faheem, M. B. Ahmad, "Machine Learning Techniques for Enhanced Intrusion
> Detection in IoT Security," *IEEE Access*, vol. 13, pp. 31140–31158, 2025.
> DOI: 10.1109/ACCESS.2025.3542227

Held-out test flows are streamed through the trained model as if they were arriving at an IoT
edge gateway. A Streamlit dashboard shows each flow's prediction, raised alarms, a running
confusion matrix, per-class detection rates, and the paper's reported numbers next to ours.

## Pipeline

1. Clean: strip headers, drop inf/NaN rows, **remove duplicates**
2. Stratified 80/20 train/test split
3. Fit on the **train split only**: label encoding, min-max scaling, Pearson-correlation
   filter (|r| > 0.97)
4. Random Forest, 50 trees, no depth limit (the paper's best setting)

This order differs from the paper, which does not state whether preprocessing came before or
after the split. Fitting on the training split only avoids test-set leakage, so results may come
out somewhat below the reported ones.

## Setup

```bash
git clone https://github.com/<your-username>/iot-nids-live-replay.git
cd iot-nids-live-replay
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Datasets are not included. Place the CSVs in `data/`:

| Dataset | File(s) | Task |
|---|---|---|
| WSN-DS | `WSN-DS.csv` | 5-class |
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | 6-class (as in the paper) |
| CIC-IDS2017 | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | DDoS vs. benign |

## Train

```bash
python -u train.py --dataset wsnds --csv data/WSN-DS.csv 2>&1 | tee logs/train_wsnds.log
python -u train.py --dataset unsw --csv data/UNSW_NB15_training-set.csv data/UNSW_NB15_testing-set.csv 2>&1 | tee logs/train_unsw.log
python -u train.py --dataset cic --csv data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv 2>&1 | tee logs/train_cic.log
```

Options: `--pcc` (correlation threshold, default 0.97), `--trees` (default 50), `--seed` (default 42).
Each run saves `models/<dataset>.joblib` and `models/<dataset>_metrics.json`.

## Run the demo

```bash
streamlit run app.py
```

On a remote server: `streamlit run app.py --server.headless true --server.port 8501`, then
`ssh -L 8501:localhost:8501 <Moze12432>@<server>` and open http://localhost:8501.

## Files

| File | Purpose |
|---|---|
| `nids_core.py` | Dataset presets and the shared preprocessing/transform |
| `train.py` | Trains the pipeline and saves a model bundle |
| `app.py` | Streamlit live-replay dashboard |
