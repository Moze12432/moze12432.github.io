"""Shared preprocessing for train.py and app.py (paper: Hakami et al., IEEE Access 2025)."""
import numpy as np
import pandas as pd

# label column, columns to drop, categorical columns, classes kept, paper's reported RF numbers (Table 6)
PRESETS = {
    "wsnds": dict(title="WSN-DS", label="Attack type", drop=[], cat=[], keep=None,
                  normal="Normal", paper_acc=99.71, paper_f1=99.71),
    "unsw": dict(title="UNSW-NB15", label="attack_cat", drop=["id", "label"], cat=["proto", "service", "state"],
                 keep=["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"],
                 normal="Normal", paper_acc=90.17, paper_f1=90.14),
    "cic": dict(title="CIC-IDS2017 (DDoS)", label="Label", drop=[], cat=[], keep=None,
                normal="BENIGN", paper_acc=99.99, paper_f1=99.99),
}


def clean_frame(df, preset, dedupe=True):
    """Strip header/label whitespace, drop unused columns, remove inf/NaN rows and (by default) duplicates.

    dedupe=False keeps exact duplicate rows; experiments.py uses it to imitate the paper's protocol.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    lab = preset["label"]
    df[lab] = df[lab].astype(str).str.strip()
    df = df.drop(columns=[c for c in preset["drop"] if c in df.columns])
    feats = [c for c in df.columns if c != lab]
    for c in feats:
        if c not in preset["cat"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if preset["keep"]:
        df = df[df[lab].isin(preset["keep"])]
    if dedupe:
        df = df.drop_duplicates()
    return df.reset_index(drop=True), feats


def transform(bundle, raw):
    """Raw feature rows -> encoded, min-max scaled, PCC-filtered matrix the RF expects."""
    X = raw[bundle["features"]].copy()
    if bundle["cat"]:
        X[bundle["cat"]] = bundle["encoder"].transform(X[bundle["cat"]].astype(str))
    X = pd.DataFrame(bundle["scaler"].transform(X.astype(float)), columns=bundle["features"])
    return X[bundle["kept"]].to_numpy()
