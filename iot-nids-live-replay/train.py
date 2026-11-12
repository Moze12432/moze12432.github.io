#!/usr/bin/env python3
"""Train the paper's Random Forest pipeline and save a bundle for the Streamlit replay demo.

Leakage-safe order: clean + dedupe -> stratified 80/20 split -> fit encoder, min-max scaler
and PCC filter on the TRAIN split only -> RF (50 trees, no depth limit) -> evaluate on test.

Zero-day mode (--holdout CLASS): that attack class is removed from the training split only, so the
saved model has never seen it, while the replay still streams it (models/<dataset>_zeroday_<class>.joblib).
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from nids_core import PRESETS, clean_frame, transform


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pcc_filter(X, thr):
    """Drop one feature of every pair with |Pearson r| > thr (computed on train only)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    return [c for c in X.columns if c not in drop], drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=PRESETS)
    ap.add_argument("--csv", required=True, nargs="+", help="one or more CSV files (concatenated)")
    ap.add_argument("--pcc", type=float, default=0.97)
    ap.add_argument("--trees", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models")
    ap.add_argument("--holdout", help="attack class removed from the TRAINING split only (zero-day demo)")
    a = ap.parse_args()
    p = PRESETS[a.dataset]
    name = a.dataset if not a.holdout else f"{a.dataset}_zeroday_{a.holdout.lower().replace(' ', '_')}"
    title = p["title"] if not a.holdout else f"{p['title']} (zero-day: {a.holdout} unseen)"
    t_start = time.time()

    log(f"Loading {len(a.csv)} file(s) for {p['title']}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in a.csv], ignore_index=True)
    log(f"Raw rows: {len(df):,}")
    df, feats = clean_frame(df, p)
    log(f"After cleaning + dedupe: {len(df):,} rows, {len(feats)} features")
    log("Class counts:\n" + df[p["label"]].value_counts().to_string())

    X, y = df[feats], df[p["label"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)
    log(f"Split: train {len(Xtr):,} / test {len(Xte):,}")
    if a.holdout:
        attacks = sorted(set(ytr) - {p["normal"]})
        if a.holdout not in attacks:
            ap.error(f"--holdout must be one of the attack classes: {attacks}")
        keep = (ytr != a.holdout).to_numpy()
        Xtr, ytr = Xtr[keep], ytr[keep]
        log(f"Zero-day mode: removed {(~keep).sum():,} '{a.holdout}' flows from training "
            f"(they stay in the test split); train is now {len(Xtr):,}")

    enc = None
    Xtr_enc = Xtr.copy()
    if p["cat"]:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr_enc[p["cat"]] = enc.fit_transform(Xtr[p["cat"]].astype(str))
    scaler = MinMaxScaler().fit(Xtr_enc.astype(float))
    Xtr_s = pd.DataFrame(scaler.transform(Xtr_enc.astype(float)), columns=feats)

    kept, dropped = pcc_filter(Xtr_s, a.pcc)
    log(f"PCC filter (|r| > {a.pcc}) on train: kept {len(kept)}, dropped {len(dropped)}: {dropped}")

    log(f"Training Random Forest ({a.trees} trees, no depth limit)")
    rf = RandomForestClassifier(n_estimators=a.trees, max_depth=None, n_jobs=-1,
                                random_state=a.seed, verbose=2)
    t0 = time.time()
    rf.fit(Xtr_s[kept].to_numpy(), ytr)
    train_s = time.time() - t0
    log(f"Training done in {train_s:.1f} s")

    unseen = [a.holdout] if a.holdout else []
    bundle = dict(dataset=a.dataset, title=title, label=p["label"], normal=p["normal"],
                  features=feats, cat=p["cat"], encoder=enc, scaler=scaler, kept=kept,
                  dropped=dropped, model=rf, classes=list(rf.classes_) + unseen, unseen=unseen)

    rf.set_params(verbose=0)
    t0 = time.time()
    pred = rf.predict(transform(bundle, Xte))
    infer_ms = (time.time() - t0) / len(Xte) * 1000
    metrics = dict(
        accuracy=100 * accuracy_score(yte, pred),
        f1_weighted=100 * f1_score(yte, pred, average="weighted"),
        f1_macro=100 * f1_score(yte, pred, average="macro"),
        train_seconds=train_s, infer_ms_per_flow=infer_ms,
        paper_acc=p["paper_acc"], paper_f1=p["paper_f1"],
        n_train=len(Xtr), n_test=len(Xte), pcc=a.pcc, kept=len(kept), total_features=len(feats),
        per_class=classification_report(yte, pred, output_dict=True, zero_division=0),
        confusion=confusion_matrix(yte, pred, labels=bundle["classes"]).tolist(),
    )
    if a.holdout:
        unseen_rows = (yte == a.holdout).to_numpy()
        metrics["unseen_flagged_pct"] = 100 * float(np.mean(pred[unseen_rows] != p["normal"]))
        log(f"Unseen class '{a.holdout}': {metrics['unseen_flagged_pct']:.2f}% of its test flows raised an alarm")
    log(f"TEST  accuracy {metrics['accuracy']:.2f}% (paper {p['paper_acc']}%)  "
        f"weighted-F1 {metrics['f1_weighted']:.2f}% (paper {p['paper_f1']}%)  macro-F1 {metrics['f1_macro']:.2f}%")
    log(f"Batch inference: {infer_ms:.4f} ms per flow")
    log("Per-class report:\n" + classification_report(yte, pred, digits=4, zero_division=0))

    bundle.update(metrics=metrics, test_raw=Xte.reset_index(drop=True), test_y=yte.reset_index(drop=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / f"{name}.joblib", compress=3)
    (out / f"{name}_metrics.json").write_text(json.dumps(
        {k: v for k, v in metrics.items() if k not in ("per_class", "confusion")}, indent=2))
    log(f"Saved {out / (name + '.joblib')}  (total {time.time() - t_start:.1f} s)")


if __name__ == "__main__":
    main()
