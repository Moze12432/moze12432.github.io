# data/

The datasets are not included in this repository. Download them from these public Kaggle copies and put the CSV files here:

| Dataset | Kaggle | Files used |
|---|---|---|
| WSN-DS | https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds | `WSN-DS.csv` |
| UNSW-NB15 | https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| CIC-IDS2017 | https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`; for the cross-day experiment also `Wednesday-workingHours.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`, `Tuesday-WorkingHours.pcap_ISCX.csv` |

The simulator (`app.py`) does not need these files: it replays the held-out test flows stored in `models/*.joblib`.
