#!/usr/bin/env python3
"""Extension experiments for the IoT NIDS reproduction of Hakami et al., IEEE Access 2025.

Run from the project folder. Tables go to results/ (CSV + Markdown), figures to results/figs/.

  /usr/bin/python3 experiments.py env       hardware + software description for the report
  /usr/bin/python3 experiments.py audit     Exp 1: evaluation-protocol (leakage) audit
  /usr/bin/python3 experiments.py improve   Exp 2: label-conflict ceiling + minority-attack fixes
  /usr/bin/python3 experiments.py zeroday   Exp 3a: unseen-attack (leave-one-attack-out) detection
  /usr/bin/python3 experiments.py crossday  Exp 3b: CIC-IDS2017 DDoS model vs. attacks from other days
  /usr/bin/python3 experiments.py edge      Exp 3c: trees x depth vs. accuracy, latency and model size
  /usr/bin/python3 experiments.py all       everything above, in this order (about 45 min on 8 cores)
  /usr/bin/python3 experiments.py figs      only redraw the figures from the CSVs in results/

Options: --seeds 42 43 44 45 46 (default)
         --sample N   use N random rows per dataset (quick smoke test; writes to results_smoke/)
"""
import os

# HistGradientBoosting uses OpenMP: match its thread count to the CPUs this process may use (taskset)
os.environ.setdefault("OMP_NUM_THREADS", str(len(os.sched_getaffinity(0))))

import argparse  # noqa: E402
import io  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from functools import lru_cache  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

from nids_core import PRESETS, clean_frame, transform  # noqa: E402
from train import pcc_filter  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
CSV = {
    "wsnds": ["WSN-DS.csv"],
    "unsw": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
    "cic": ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
}
CIC_OTHER_DAYS = {  # same Kaggle dataset as the Friday DDoS file (chethuhn/network-intrusion-dataset)
    "Wednesday (DoS)": "Wednesday-workingHours.pcap_ISCX.csv",
    "Friday PM (PortScan)": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Tuesday (brute force)": "Tuesday-WorkingHours.pcap_ISCX.csv",
}
PAPER_UNSW_OVERSAMPLED = 98.8  # the paper's UNSW-NB15 accuracy after oversampling (see report notes)
PCC = 0.97
INK, MAROON, ROSE, GRAY, LIGHT = "#222222", "#800400", "#D98C8C", "#595959", "#BFBFBF"
SEQ = LinearSegmentedColormap.from_list("maroon", ["#FFFFFF", ROSE, MAROON])

PROTOCOLS = {  # key: (short label for figures, description for tables)
    "A": ("no dedupe, prep on all, then split",
          "No dedupe; scaler + PCC filter fitted on all data; then stratified 80/20 split"),
    "B": ("A + oversample before split",
          "Protocol A plus random oversampling of every class to the majority size BEFORE the split"),
    "C": ("dedupe, prep on all, then split",
          "Dedupe; scaler + PCC filter fitted on all data; then split"),
    "D": ("dedupe, split, prep on train (ours)",
          "Dedupe; split; scaler + PCC filter fitted on the training split only (our pipeline)"),
    "D+OS": ("D + oversample train only",
             "Protocol D plus random oversampling of the training split only"),
    "E": ("official train/test files",
          "UNSW-NB15's official training and testing files (UNSW-NB15 only)"),
}
METHODS = ["RF (paper setting)", "RF + class weights", "RF + random oversampling", "RF + SMOTE",
           "HistGradientBoosting", "HGB + class weights", "Two-stage RF"]
ARGS = None

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": GRAY,
    "xtick.color": GRAY, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "savefig.dpi": 200,
})


# ================================================================ helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def maybe_sample(df):
    if ARGS.sample and len(df) > ARGS.sample:
        return df.sample(ARGS.sample, random_state=0).reset_index(drop=True)
    return df


@lru_cache(None)
def raw(ds):
    return pd.concat([pd.read_csv(DATA / f, low_memory=False) for f in CSV[ds]], ignore_index=True)


@lru_cache(None)
def cleaned(ds, dedupe):
    df, feats = clean_frame(raw(ds), PRESETS[ds], dedupe=dedupe)
    return maybe_sample(df), feats


@lru_cache(None)
def cic_day(fname):
    return maybe_sample(clean_frame(pd.read_csv(DATA / fname, low_memory=False), PRESETS["cic"])[0])


@lru_cache(None)
def unsw_official():
    """Official UNSW-NB15 split. The larger file is the official 175,341-row training set
    (the two file names are swapped in this release)."""
    files = sorted(CSV["unsw"], key=lambda f: (DATA / f).stat().st_size, reverse=True)
    (tr, feats), (te, _) = (clean_frame(pd.read_csv(DATA / f, low_memory=False), PRESETS["unsw"]) for f in files)
    return maybe_sample(tr), maybe_sample(te), feats


def strat(y):
    y = np.asarray(y)
    return y if pd.Series(y).value_counts().min() >= 2 else None


def split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, stratify=strat(y), random_state=seed)


def fit_prep(X, p, pcc=PCC):
    """Fit ordinal encoder, min-max scaler and PCC filter on X (same steps as train.py)."""
    enc, Xe = None, X.copy()
    if p["cat"]:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xe[p["cat"]] = enc.fit_transform(X[p["cat"]].astype(str))
    scaler = MinMaxScaler().fit(Xe.astype(float))
    kept, dropped = pcc_filter(pd.DataFrame(scaler.transform(Xe.astype(float)), columns=X.columns), pcc)
    return dict(features=list(X.columns), cat=p["cat"], encoder=enc, scaler=scaler, kept=kept, dropped=dropped)


def split_fit(X, y, p, seed):
    """Our leakage-safe protocol: split first, then fit preprocessing on the training split only."""
    Xtr, Xte, ytr, yte = split(X, np.asarray(y), seed)
    prep = fit_prep(Xtr, p)
    return prep, transform(prep, Xtr), transform(prep, Xte), np.asarray(ytr), np.asarray(yte), Xtr, Xte


def rf(seed, trees=50, depth=None, **kw):
    return RandomForestClassifier(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=seed, **kw)


def scores(yt, yp, normal):
    yt, yp = np.asarray(yt), np.asarray(yp)
    att = yt != normal
    r = dict(accuracy=100 * accuracy_score(yt, yp),
             f1_weighted=100 * f1_score(yt, yp, average="weighted", zero_division=0),
             f1_macro=100 * f1_score(yt, yp, average="macro", zero_division=0),
             detection_rate=100 * np.mean(yp[att] != normal) if att.any() else np.nan,
             false_alarm_rate=100 * np.mean(yp[~att] != normal) if (~att).any() else np.nan)
    for c in np.unique(yt):
        r[f"recall[{c}]"] = 100 * np.mean(yp[yt == c] == c)
    return r


def row_hash(M):
    return pd.util.hash_pandas_object(pd.DataFrame(M), index=False).to_numpy()


def test_in_train(Mtr, Mte):
    """% of test rows whose model input is identical to some training row (memorisable)."""
    return 100 * np.isin(row_hash(Mte), row_hash(Mtr)).mean()


def random_oversample(M, y, seed):
    """Duplicate random rows of every class until all classes have the majority-class size."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    idx = [np.where(y == c)[0] for c in classes]
    idx = np.concatenate([np.concatenate([i, rng.choice(i, counts.max() - len(i))]) for i in idx])
    return M[idx], y[idx]


def smote(M, y, seed, k=5, cat_idx=()):
    """SMOTE (Chawla et al., 2002), own implementation: new minority rows are interpolated between a
    row and one of its k nearest same-class neighbours. Categorical columns are copied, not interpolated."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    Ms, ys = [M], [y]
    for c, n in zip(classes, counts):
        need = counts.max() - n
        if need <= 0 or n < 2:
            continue
        Xc = M[y == c]
        nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(Xc)
        base = rng.integers(0, n, need)
        neigh = nn.kneighbors(Xc[base], return_distance=False)[:, 1:]
        pick = neigh[np.arange(need), rng.integers(0, neigh.shape[1], need)]
        new = Xc[base] + rng.random((need, 1)) * (Xc[pick] - Xc[base])
        if len(cat_idx):
            new[:, cat_idx] = Xc[base][:, cat_idx]
        Ms.append(new)
        ys.append(np.full(need, c, dtype=y.dtype))
    return np.vstack(Ms), np.concatenate(ys)


def save_csv(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    log(f"wrote {OUT.name}/{name}")


def save_fig(fig, name):
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figs" / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {OUT.name}/figs/{name}")


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.2f}"
    return str(v)


def md_table(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_md(name, title, sections):
    OUT.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n" + "\n\n".join(f"## {h}\n\n{body}" for h, body in sections) + "\n"
    (OUT / name).write_text(text, encoding="utf-8")
    log(f"wrote {OUT.name}/{name}")


def mean_pm_std(runs, by, cols):
    """Group means, stds, and a 'mean ± std' text table."""
    g = runs.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std()
    txt = m.copy().astype(object)
    for c in cols:
        txt[c] = ["" if np.isnan(a) else f"{a:.2f}" if np.isnan(b) else f"{a:.2f} ± {b:.2f}"
                  for a, b in zip(m[c], s[c])]
    return m.reset_index(), s.reset_index(), txt.reset_index()


def seeds_text():
    return f"{len(ARGS.seeds)} seed(s): {', '.join(map(str, ARGS.seeds))}"


# ================================================================ env
def env():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"n/a ({e})"

    lscpu = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in sh("lscpu").splitlines() if ":" in ln)}
    mem_kb = next((int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines()
                   if ln.startswith("MemTotal")), 0)
    osrel = dict(ln.split("=", 1) for ln in Path("/etc/os-release").read_text().splitlines() if "=" in ln)
    gpus = Counter(g.strip() for g in sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines() if g.strip())
    rows = [
        ("Machine", platform.node()),
        ("CPU", lscpu.get("Model name", "n/a")),
        ("Sockets x cores per socket x threads per core",
         f"{lscpu.get('Socket(s)', '?')} x {lscpu.get('Core(s) per socket', '?')} x {lscpu.get('Thread(s) per core', '?')}"),
        ("Logical CPUs: total / used by these runs (taskset)", f"{os.cpu_count()} / {len(os.sched_getaffinity(0))}"),
        ("RAM", f"{mem_kb / 1024 ** 2:.0f} GB"),
        ("GPU", (", ".join(f"{n} x {g}" for g, n in gpus.items()) or "none") + " (not used; all models run on CPU)"),
        ("OS", f"{osrel.get('PRETTY_NAME', 'Linux').strip(chr(34))}, kernel {platform.release()}"),
        ("Server load average (1/5/15 min) at run time", " / ".join(f"{x:.1f}" for x in os.getloadavg())),
        ("Python", f"{sys.version.split()[0]} ({sys.executable})"),
    ] + [(lib, version(lib)) for lib in ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib",
                                          "streamlit", "jinja2")]
    write_md("environment.md", "Simulation environment",
             [("Hardware and software", md_table(pd.DataFrame(rows, columns=["Item", "Value"])))])


# ================================================================ Exp 1: protocol audit
def run_protocol(ds, proto, seed):
    p = PRESETS[ds]
    lab = p["label"]
    if proto == "E":
        tr, te, feats = unsw_official()
        prep = fit_prep(tr[feats], p)
        Mtr, Mte = transform(prep, tr[feats]), transform(prep, te[feats])
        ytr, yte = tr[lab].to_numpy(), te[lab].to_numpy()
    else:
        df, feats = cleaned(ds, proto not in ("A", "B"))
        X, y = df[feats], df[lab].to_numpy()
        if proto in ("A", "B", "C"):  # preprocessing sees all data, split afterwards
            M = transform(fit_prep(X, p), X)
            if proto == "B":
                M, y = random_oversample(M, y, seed)
            Mtr, Mte, ytr, yte = split(M, y, seed)
        else:  # D, D+OS: split first, preprocessing fitted on train only
            _, Mtr, Mte, ytr, yte, _, _ = split_fit(X, y, p, seed)
            if proto == "D+OS":
                Mtr, ytr = random_oversample(Mtr, ytr, seed)
    t0 = time.time()
    pred = rf(seed).fit(Mtr, ytr).predict(Mte)
    r = dict(ds=ds, dataset=p["title"], protocol=proto, seed=seed, n_train=len(ytr), n_test=len(yte),
             test_in_train_pct=test_in_train(Mtr, Mte), seconds=time.time() - t0)
    r.update(scores(yte, pred, p["normal"]))
    return r


def audit():
    rows = []
    for ds in ("wsnds", "unsw", "cic"):
        for proto in PROTOCOLS:
            if proto == "E" and ds != "unsw":
                continue
            for seed in ARGS.seeds:
                r = run_protocol(ds, proto, seed)
                rows.append(r)
                log(f"audit {ds:5s} {proto:4s} seed {seed}: acc {r['accuracy']:.2f}  macro-F1 {r['f1_macro']:.2f}  "
                    f"test rows identical to a train row {r['test_in_train_pct']:.1f}%  ({r['seconds']:.0f} s)")
    runs = pd.DataFrame(rows)
    save_csv(runs, "audit_runs.csv")
    cols = [c for c in ["accuracy", "f1_weighted", "f1_macro", "test_in_train_pct", "recall[DoS]"] if c in runs]
    m, s, txt = mean_pm_std(runs, ["ds", "dataset", "protocol"], cols)
    txt.insert(3, "paper accuracy", [PRESETS[d]["paper_acc"] for d in txt["ds"]])
    txt = txt.drop(columns="ds").rename(columns={"test_in_train_pct": "test rows identical to a train row (%)",
                                                 "recall[DoS]": "DoS recall"})
    save_csv(txt, "audit_summary.csv")
    write_md("audit.md", "Experiment 1: evaluation-protocol audit", [
        (f"Results, mean ± std over {seeds_text()}", md_table(txt)),
        ("Protocols", md_table(pd.DataFrame([(k, v[1]) for k, v in PROTOCOLS.items()],
                                            columns=["Protocol", "Meaning"]))),
        ("Reference", f"Paper's UNSW-NB15 accuracy after oversampling: {PAPER_UNSW_OVERSAMPLED}%."),
    ])
    fig_audit(m, s)


def fig_audit(m, s):
    dss = list(dict.fromkeys(m["ds"]))
    counts = [int((m.ds == d).sum()) for d in dss]
    fig, axes = plt.subplots(len(dss), 2, figsize=(10.5, 0.42 * sum(counts) + 1.1 * len(dss)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.7, 1], height_ratios=counts))
    for i, ds in enumerate(dss):
        mi, si = m[m.ds == ds].set_index("protocol"), s[s.ds == ds].set_index("protocol")
        order = [k for k in PROTOCOLS if k in mi.index]
        y = np.arange(len(order))
        col = [MAROON if k == "D" else GRAY for k in order]
        acc = mi.loc[order, "accuracy"].to_numpy()
        err = si.loc[order, "accuracy"].fillna(0).to_numpy()
        refs = [(PRESETS[ds]["paper_acc"], "--", f"paper {PRESETS[ds]['paper_acc']:g}%")]
        if ds == "unsw":
            refs.append((PAPER_UNSW_OVERSAMPLED, ":", f"paper, oversampled {PAPER_UNSW_OVERSAMPLED:g}%"))
        lo = min((acc - err).min(), *(r[0] for r in refs))
        hi = max((acc + err).max(), *(r[0] for r in refs))
        pad = max((hi - lo) * 0.08, 0.02)
        ax = axes[i, 0]
        ax.grid(axis="x", color="#EEEEEE", lw=0.8)
        ax.set_axisbelow(True)
        for x, ls, label in refs:
            ax.axvline(x, ls=ls, lw=1, color=INK, zorder=1)
            ax.text(x, -0.95, label, ha="center", va="center", fontsize=8)
        ax.errorbar(acc, y, xerr=err, fmt="none", ecolor=LIGHT, elinewidth=2.5, zorder=2)
        ax.scatter(acc, y, c=col, s=36, zorder=3)
        for xi, yi in zip(acc, y):
            ax.text(xi, yi - 0.36, f"{xi:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xlim(lo - pad, min(hi + pad, 100.0))  # accuracy cannot exceed 100%
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f" if hi - lo < 1 else "%g"))
        ax.set_ylim(len(order) - 0.5, -1.3)
        ax.set_yticks(y, [f"{k}  {PROTOCOLS[k][0]}" for k in order])
        ax.set_title(f"{PRESETS[ds]['title']}: test accuracy (%), mean ± std")
        ov = mi.loc[order, "test_in_train_pct"].to_numpy()
        ax2 = axes[i, 1]
        ax2.barh(y, ov, color=col, height=0.5)
        for yi, v in zip(y, ov):
            ax2.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=8)
        ax2.set_xlim(0, 112)
        ax2.set_ylim(len(order) - 0.5, -1.3)
        ax2.set_yticks(y, [""] * len(y))
        ax2.set_title("Test rows identical to a training row (%)")
    fig.tight_layout()
    save_fig(fig, "audit.png")


# ================================================================ Exp 2: label conflicts + minority fixes
def label_conflicts(ds):
    """Rows whose exact feature vector also appears with a different label. A classifier must give
    identical inputs the same answer, so these rows put a hard ceiling on accuracy."""
    p = PRESETS[ds]
    out = []
    for stage, dedupe in (("before dedupe", False), ("after dedupe", True)):
        df, feats = cleaned(ds, dedupe)
        g = pd.DataFrame({"key": pd.util.hash_pandas_object(df[feats], index=False).to_numpy(),
                          "y": df[p["label"]].to_numpy()})
        counts = g.groupby(["key", "y"]).size().rename("n").reset_index()
        nlab = counts.groupby("key")["y"].transform("nunique")
        bad_keys = counts.loc[nlab > 1, "key"].unique()
        g["conflict"] = np.isin(g["key"], bad_keys)
        best = counts.groupby("key")["n"].max().sum()  # best case: every vector gets its majority label
        maj = counts.loc[counts.groupby("key")["n"].idxmax()].set_index("key")["y"]
        counts["is_maj"] = counts["y"].to_numpy() == counts["key"].map(maj).to_numpy()
        rec_ceiling = 100 * counts[counts.is_maj].groupby("y")["n"].sum() / counts.groupby("y")["n"].sum()
        labsets = counts[counts.key.isin(bad_keys)].groupby("key")["y"].agg(frozenset)
        out.append(dict(dataset=p["title"], stage=stage, cls="ALL", rows=len(g),
                        conflicting_pct=100 * g["conflict"].mean(), accuracy_ceiling=100 * best / len(g),
                        recall_ceiling=np.nan, shares_vectors_with=""))
        for c in sorted(g["y"].unique()):
            mc = (g["y"] == c).to_numpy()
            keys = g.loc[mc & g["conflict"].to_numpy(), "key"].to_numpy()
            other = Counter(o for ls in labsets.reindex(keys) for o in ls if o != c)
            top = ", ".join(f"{o} ({100 * n / mc.sum():.1f}% of rows)" for o, n in other.most_common(2))
            out.append(dict(dataset=p["title"], stage=stage, cls=c, rows=int(mc.sum()),
                            conflicting_pct=100 * g.loc[mc, "conflict"].mean(), accuracy_ceiling=np.nan,
                            recall_ceiling=rec_ceiling.get(c, np.nan) if not dedupe else np.nan,
                            shares_vectors_with=top))
    return pd.DataFrame(out)


def fit_predict(method, Mtr, ytr, Mte, seed, cat_idx, normal):
    if method == "RF (paper setting)":
        return rf(seed).fit(Mtr, ytr).predict(Mte)
    if method == "RF + class weights":
        return rf(seed, class_weight="balanced").fit(Mtr, ytr).predict(Mte)
    if method == "RF + random oversampling":
        X, y = random_oversample(Mtr, ytr, seed)
        return rf(seed).fit(X, y).predict(Mte)
    if method == "RF + SMOTE":
        X, y = smote(Mtr, ytr, seed, cat_idx=cat_idx)
        return rf(seed).fit(X, y).predict(Mte)
    if method in ("HistGradientBoosting", "HGB + class weights"):
        cw = "balanced" if method == "HGB + class weights" else None
        return HistGradientBoostingClassifier(max_iter=300, random_state=seed, class_weight=cw).fit(Mtr, ytr).predict(Mte)
    if method == "Two-stage RF":  # stage 1: normal vs. attack; stage 2: attack type (class-weighted)
        is_att = ytr != normal
        gate = rf(seed).fit(Mtr, is_att)
        typ = rf(seed, class_weight="balanced").fit(Mtr[is_att], ytr[is_att])
        pred = np.full(len(Mte), normal, dtype=object)
        flag = gate.predict(Mte).astype(bool)
        if flag.any():
            pred[flag] = typ.predict(Mte[flag])
        return pred
    raise ValueError(method)


def improve():
    conf = pd.concat([label_conflicts(ds) for ds in ("unsw", "wsnds")], ignore_index=True)
    save_csv(conf, "improve_conflicts.csv")
    rows, cms = [], {}
    for ds in ("unsw", "wsnds"):
        p = PRESETS[ds]
        df, feats = cleaned(ds, True)
        for seed in ARGS.seeds:
            prep, Mtr, Mte, ytr, yte, _, _ = split_fit(df[feats], df[p["label"]], p, seed)
            cat_idx = [prep["kept"].index(c) for c in p["cat"] if c in prep["kept"]]
            for meth in METHODS:
                t0 = time.time()
                pred = fit_predict(meth, Mtr, ytr, Mte, seed, cat_idx, p["normal"])
                r = dict(ds=ds, dataset=p["title"], method=meth, seed=seed, seconds=time.time() - t0)
                r.update(scores(yte, pred, p["normal"]))
                rows.append(r)
                if seed == ARGS.seeds[0]:
                    cms[(ds, meth)] = (yte, pred)
                minority = "recall[DoS]" if ds == "unsw" else "recall[TDMA]"
                log(f"improve {ds:5s} seed {seed} {meth:26s}: acc {r['accuracy']:.2f}  macro-F1 {r['f1_macro']:.2f}  "
                    f"{minority} {r.get(minority, np.nan):.1f}  ({r['seconds']:.0f} s)")
    runs = pd.DataFrame(rows)
    save_csv(runs, "improve_runs.csv")
    sections = [("Label conflicts (identical features, different labels)",
                 "Rows that share their exact feature vector with a row of another class. 'accuracy_ceiling' is "
                 "the best accuracy any classifier could reach on that data; 'recall_ceiling' is the best "
                 "possible recall per class (before dedupe only, where it is well defined).\n\n" + md_table(conf))]
    for ds in ("unsw", "wsnds"):
        r = runs[runs.ds == ds].dropna(axis=1, how="all")
        rcols = [c for c in r.columns if c.startswith("recall[")]
        _, _, txt = mean_pm_std(r, ["method"], ["accuracy", "f1_weighted", "f1_macro", "detection_rate",
                                                "false_alarm_rate", *rcols])
        save_csv(txt, f"improve_summary_{ds}.csv")
        sections.append((f"{PRESETS[ds]['title']}: methods, mean ± std over {seeds_text()}", md_table(txt)))
        fig_improve(runs, ds)
        best = r.groupby("method")["f1_macro"].mean().idxmax()
        fig_confusion(*cms[(ds, METHODS[0])], ds, f"confusion_{ds}_baseline.png",
                      f"{PRESETS[ds]['title']}: {METHODS[0]}, row % (seed {ARGS.seeds[0]})")
        if best != METHODS[0]:
            fig_confusion(*cms[(ds, best)], ds, f"confusion_{ds}_best.png",
                          f"{PRESETS[ds]['title']}: {best}, row % (seed {ARGS.seeds[0]})")
    write_md("improve.md", "Experiment 2: label-conflict ceiling and minority-attack fixes", sections)


def class_order(ds, labels):
    normal = PRESETS[ds]["normal"]
    return [normal] + sorted(c for c in set(labels) if c != normal)


def fig_improve(runs, ds):
    r = runs[runs.ds == ds]
    classes = class_order(ds, [c[7:-1] for c in r.columns if c.startswith("recall[") and r[c].notna().any()])
    rec = r.groupby("method")[[f"recall[{c}]" for c in classes]].mean().loc[METHODS].to_numpy()
    f1 = r.groupby("method")["f1_macro"].agg(["mean", "std"]).loc[METHODS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 0.48 * len(METHODS) + 1.9),
                               gridspec_kw=dict(width_ratios=[2.3, 1]))
    a.imshow(rec, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            a.text(j, i, f"{rec[i, j]:.1f}", ha="center", va="center", fontsize=8,
                   color="white" if rec[i, j] > 60 else INK)
    a.set_xticks(range(len(classes)), classes, rotation=25, ha="right")
    a.set_yticks(range(len(METHODS)), METHODS)
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title(f"{PRESETS[ds]['title']}: recall per class (%), mean over seeds")
    y = np.arange(len(METHODS))
    b.grid(axis="x", color="#EEEEEE", lw=0.8)
    b.set_axisbelow(True)
    b.axvline(f1["mean"].iloc[0], ls=":", lw=1, color=GRAY)
    b.errorbar(f1["mean"], y, xerr=f1["std"].fillna(0), fmt="none", ecolor=LIGHT, elinewidth=2.5)
    b.scatter(f1["mean"], y, c=[GRAY] + [MAROON] * (len(METHODS) - 1), s=36, zorder=3)
    for yi, v in zip(y, f1["mean"]):
        b.text(v, yi - 0.34, f"{v:.2f}", ha="center", va="center", fontsize=8)
    lo, hi = (f1["mean"] - f1["std"].fillna(0)).min(), (f1["mean"] + f1["std"].fillna(0)).max()
    pad = max((hi - lo) * 0.12, 0.1)
    b.set_xlim(lo - pad, hi + pad)
    b.set_ylim(len(METHODS) - 0.5, -0.8)
    b.set_yticks(y, [""] * len(y))
    b.set_title("Macro-F1 (%), mean ± std")
    fig.tight_layout()
    save_fig(fig, f"improve_{ds}.png")


def fig_confusion(yt, yp, ds, name, title):
    classes = class_order(ds, set(yt) | set(yp))
    cm = pd.crosstab(pd.Series(yt, name="true"), pd.Series(yp, name="pred")).reindex(
        index=classes, columns=classes, fill_value=0).to_numpy()
    pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(pct, cmap=SEQ, vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 60 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, name)


# ================================================================ Exp 3a: zero-day (unseen attack)
def zeroday():
    rows, where = [], []
    for ds in ("wsnds", "unsw"):
        p = PRESETS[ds]
        normal = p["normal"]
        df, feats = cleaned(ds, True)
        attacks = sorted(c for c in df[p["label"]].unique() if c != normal)
        for seed in ARGS.seeds:
            Xtr, Xte, ytr, yte = split(df[feats], df[p["label"]].to_numpy(), seed)
            for held in ["(none)", *attacks]:
                keep = ytr != held
                prep = fit_prep(Xtr[keep], p)  # the unseen class never reaches preprocessing either
                Mtr, Mte = transform(prep, Xtr[keep]), transform(prep, Xte)
                for form in ("multi-class", "binary"):
                    target = ytr[keep] if form == "multi-class" else np.where(ytr[keep] == normal, normal, "Attack")
                    pred = rf(seed).fit(Mtr, target).predict(Mte)
                    flagged = pred != normal
                    r = dict(ds=ds, dataset=p["title"], held_out=held, formulation=form, seed=seed,
                             false_alarm_pct=100 * flagged[yte == normal].mean())
                    for c in attacks:
                        r[f"flagged[{c}]"] = 100 * flagged[yte == c].mean()
                    rows.append(r)
                    if held != "(none)":
                        log(f"zeroday {ds:5s} seed {seed} unseen {held:10s} {form:11s}: "
                            f"{r[f'flagged[{held}]']:.1f}% of its flows raised an alarm "
                            f"(false alarms {r['false_alarm_pct']:.2f}%)")
                        if seed == ARGS.seeds[0] and form == "multi-class":
                            vc = pd.Series(pred[yte == held]).value_counts(normalize=True) * 100
                            where += [dict(dataset=p["title"], unseen=held, predicted_as=k, pct=v) for k, v in vc.items()]
    runs = pd.DataFrame(rows)
    save_csv(runs, "zeroday_runs.csv")
    save_csv(pd.DataFrame(where), "zeroday_where.csv")
    summ = []
    for (ds, title), r in runs.groupby(["ds", "dataset"], sort=False):
        for c in [k[8:-1] for k in r.columns if k.startswith("flagged[") and r[k].notna().any()]:
            seen = r[(r.held_out == "(none)") & (r.formulation == "multi-class")]
            um = r[(r.held_out == c) & (r.formulation == "multi-class")]
            ub = r[(r.held_out == c) & (r.formulation == "binary")]
            summ.append(dict(ds=ds, dataset=title, attack=c,
                             seen=seen[f"flagged[{c}]"].mean(), seen_std=seen[f"flagged[{c}]"].std(),
                             unseen_multiclass=um[f"flagged[{c}]"].mean(), unseen_multiclass_std=um[f"flagged[{c}]"].std(),
                             unseen_binary=ub[f"flagged[{c}]"].mean(), unseen_binary_std=ub[f"flagged[{c}]"].std(),
                             false_alarm_multiclass=um["false_alarm_pct"].mean(),
                             false_alarm_binary=ub["false_alarm_pct"].mean()))
    summ = pd.DataFrame(summ)
    save_csv(summ, "zeroday_summary.csv")
    write_md("zeroday.md", "Experiment 3a: detecting attack types the model never saw", [
        (f"Share of each attack's test flows that raised an alarm (%), mean over {seeds_text()}",
         "'seen' = normal training with all classes; 'unseen_*' = that attack removed from training "
         "(multi-class RF as in the paper, or a binary normal-vs-attack RF).\n\n" + md_table(summ.drop(columns="ds"))),
        (f"Where the unseen flows went (multi-class RF, seed {ARGS.seeds[0]})", md_table(pd.DataFrame(where))),
    ])
    fig_zeroday(summ)


def fig_zeroday(summ):
    dss = list(dict.fromkeys(summ["ds"]))
    n = [int((summ.ds == d).sum()) for d in dss]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.4 * len(dss), 0.75 * max(n) + 1.8), squeeze=False)
    series = [("seen", "Seen in training", LIGHT), ("unseen_multiclass", "Unseen, multi-class RF (paper)", MAROON),
              ("unseen_binary", "Unseen, binary normal-vs-attack RF", ROSE)]
    for ax, ds in zip(axes[0], dss):
        s = summ[summ.ds == ds].reset_index(drop=True)
        y = np.arange(len(s))
        h = 0.26
        for k, (col, label, color) in enumerate(series):
            yy = y + (k - 1) * h
            ax.barh(yy, s[col], height=h * 0.92, color=color, label=label)
            if col != "seen":
                for yi, v in zip(yy, s[col]):
                    ax.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=7.5)
        ax.set_yticks(y, s["attack"])
        ax.set_ylim(len(s) - 0.5, -0.6)
        ax.set_xlim(0, 112)
        ax.set_xlabel("Flows of this attack that raised an alarm (%)")
        ax.set_title(PRESETS[ds]["title"])
        ax.grid(axis="x", color="#EEEEEE", lw=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save_fig(fig, "zeroday.png")


# ================================================================ Exp 3b: CIC-IDS2017 cross-day
def crossday():
    p = PRESETS["cic"]
    lab, normal = p["label"], p["normal"]
    days = {name: cic_day(f) for name, f in CIC_OTHER_DAYS.items() if (DATA / f).exists()}
    if not days:
        log("crossday: none of the other CIC-IDS2017 day files are in data/; skipped")
        return
    fri, feats = cleaned("cic", True)
    rows = []
    for seed in ARGS.seeds:
        Xtr, Xte, ytr, yte = split(fri[feats], fri[lab].to_numpy(), seed)
        tests = {"Friday PM (DDoS), same day": (Xte, yte)}
        variants = {"Friday DDoS only (paper setup)": (Xtr, ytr)}
        wed = days.get("Wednesday (DoS)")
        for name, d in days.items():
            if d is wed:
                wtr, wte = train_test_split(d, test_size=0.2, stratify=strat(d[lab]), random_state=seed)
                tests[name + ", held-out 20%"] = (wte[feats], wte[lab].to_numpy())
                variants["Friday DDoS + Wednesday DoS"] = (pd.concat([Xtr, wtr[feats]]),
                                                            np.concatenate([ytr, wtr[lab].to_numpy()]))
            else:
                tests[name] = (d[feats], d[lab].to_numpy())
        for vname, (XT, yT) in variants.items():
            prep = fit_prep(XT, p)
            model = rf(seed).fit(transform(prep, XT), np.where(yT == normal, normal, "ATTACK"))
            for tname, (XE, yE) in tests.items():
                flagged = model.predict(transform(prep, XE)) != normal
                for c in np.unique(yE):
                    mc = yE == c
                    rows.append(dict(model=vname, test_set=tname, label=c, seed=seed, n=int(mc.sum()),
                                     flagged_pct=100 * flagged[mc].mean()))
            log(f"crossday seed {seed} {vname}: " + ", ".join(
                f"{r['label']} {r['flagged_pct']:.1f}%" for r in rows if r["seed"] == seed and r["model"] == vname))
    runs = pd.DataFrame(rows)
    save_csv(runs, "crossday_runs.csv")
    summ = runs.groupby(["model", "test_set", "label"], sort=False).agg(
        n=("n", "mean"), flagged_pct=("flagged_pct", "mean"), flagged_std=("flagged_pct", "std")).reset_index()
    summ["meaning"] = np.where(summ["label"] == normal, "false-alarm rate", "detection rate")
    save_csv(summ, "crossday_summary.csv")
    write_md("crossday.md", "Experiment 3b: CIC-IDS2017 DDoS detector vs. other days and attack types", [
        (f"Flows that raised an alarm (%), mean over {seeds_text()}",
         "For BENIGN rows this is the false-alarm rate; for attack rows the detection rate.\n\n" + md_table(summ)),
    ])
    fig_crossday(summ, normal)


def fig_crossday(summ, normal):
    models = list(dict.fromkeys(summ["model"]))
    colors = dict(zip(models, [MAROON, ROSE]))
    att = summ[summ.label != normal]
    ben = summ[summ.label == normal]
    cats_a = list(dict.fromkeys(zip(att.test_set, att.label)))
    cats_b = list(dict.fromkeys(ben.test_set))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 0.42 * len(cats_a) + 1.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    h = 0.8 / len(models)
    for k, mname in enumerate(models):
        sa = att[att.model == mname].set_index(["test_set", "label"])["flagged_pct"]
        sb = ben[ben.model == mname].set_index("test_set")["flagged_pct"]
        ya = np.arange(len(cats_a)) + (k - (len(models) - 1) / 2) * h
        va = [sa.get(c, np.nan) for c in cats_a]
        a.barh(ya, va, height=h * 0.92, color=colors[mname], label=mname)
        for yi, v in zip(ya, va):
            if not np.isnan(v):
                a.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=7.5)
        yb = np.arange(len(cats_b)) + (k - (len(models) - 1) / 2) * h
        vb = [sb.get(c, np.nan) for c in cats_b]
        b.barh(yb, vb, height=h * 0.92, color=colors[mname])
        for yi, v in zip(yb, vb):
            if not np.isnan(v):
                b.text(v + 0.3, yi, f"{v:.2f}", va="center", fontsize=7.5)
    a.set_yticks(range(len(cats_a)), [f"{lbl}  [{ts}]" for ts, lbl in cats_a])
    a.set_ylim(len(cats_a) - 0.5, -0.6)
    a.set_xlim(0, 112)
    a.set_xlabel("Attack flows that raised an alarm (%)")
    a.set_title("Detection rate per attack type")
    b.set_yticks(range(len(cats_b)), cats_b)
    b.set_ylim(len(cats_b) - 0.5, -0.6)
    b.set_xlim(0, max(5, np.nanmax(ben.flagged_pct.to_numpy()) * 1.25))
    b.set_xlabel("BENIGN flows that raised an alarm (%)")
    b.set_title("False-alarm rate per day")
    for ax in (a, b):
        ax.grid(axis="x", color="#EEEEEE", lw=0.8)
        ax.set_axisbelow(True)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save_fig(fig, "crossday.png")


# ================================================================ Exp 3c: gateway cost (trees x depth)
def edge():
    rows = []
    seed = ARGS.seeds[0]
    for ds in ("wsnds", "unsw", "cic"):
        p = PRESETS[ds]
        df, feats = cleaned(ds, True)
        prep, Mtr, Mte, ytr, yte, _, Xte = split_fit(df[feats], df[p["label"]], p, seed)
        probe = np.random.default_rng(seed).choice(len(Mte), min(200, len(Mte)), replace=False)
        for trees in (5, 10, 25, 50, 100):
            for depth in (8, 12, 16, 20, None):
                model = rf(seed, trees, depth)
                t0 = time.time()
                model.fit(Mtr, ytr)
                fit_s = time.time() - t0
                r = dict(ds=ds, dataset=p["title"], trees=trees, max_depth=str(depth or "none"), fit_s=fit_s)
                r.update(scores(yte, model.predict(Mte), p["normal"]))
                model.set_params(n_jobs=1)  # a gateway classifies one flow at a time on one core
                lat_m, lat_p = [], []
                for i in probe:
                    t0 = time.perf_counter()
                    model.predict(Mte[i:i + 1])
                    lat_m.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    model.predict(transform(prep, Xte.iloc[i:i + 1]))
                    lat_p.append(time.perf_counter() - t0)
                nb = min(10000, len(Mte))
                t0 = time.perf_counter()
                model.predict(Mte[:nb])
                thr = nb / (time.perf_counter() - t0)
                buf = io.BytesIO()
                joblib.dump(model, buf, compress=3)
                r.update(latency_model_ms_median=1000 * np.median(lat_m),
                         latency_pipeline_ms_median=1000 * np.median(lat_p),
                         latency_pipeline_ms_p95=1000 * np.percentile(lat_p, 95),
                         throughput_flows_per_s_1core=thr, size_mb=len(pickle.dumps(model)) / 1e6,
                         size_compressed_mb=buf.tell() / 1e6,
                         nodes=int(sum(e.tree_.node_count for e in model.estimators_)))
                rows.append(r)
                log(f"edge {ds:5s} trees {trees:3d} depth {r['max_depth']:>4s}: macro-F1 {r['f1_macro']:.2f}  "
                    f"latency {r['latency_pipeline_ms_median']:.2f} ms  size {r['size_compressed_mb']:.2f} MB")
    runs = pd.DataFrame(rows)
    save_csv(runs, "edge_runs.csv")
    picks = edge_picks(runs)
    cols = ["dataset", "choice", "trees", "max_depth", "accuracy", "f1_macro", "latency_pipeline_ms_median",
            "latency_pipeline_ms_p95", "throughput_flows_per_s_1core", "size_compressed_mb", "nodes"]
    save_csv(picks[cols], "edge_summary.csv")
    write_md("edge.md", "Experiment 3c: gateway cost (trees x depth)", [
        ("Paper setting vs. the smallest model within 0.5 pp macro-F1", md_table(picks[cols])),
        ("All configurations (seed %d)" % seed, md_table(runs[cols[:1] + cols[2:]])),
    ])
    fig_edge(runs, picks)


def edge_picks(runs):
    """Per dataset: the paper setting and the smallest model within 0.5 pp macro-F1 of it."""
    picks = []
    for _, r in runs.groupby("ds", sort=False):
        paper = r[(r.trees == 50) & (r.max_depth == "none")].iloc[0]
        ok = r[r.f1_macro >= paper.f1_macro - 0.5].sort_values(["size_compressed_mb", "latency_pipeline_ms_median"])
        picks.append(paper.to_dict() | {"choice": "paper setting"})
        picks.append(ok.iloc[0].to_dict() | {"choice": "smallest within 0.5 pp macro-F1"})
    return pd.DataFrame(picks)


def fig_edge(runs, picks):
    dss = list(dict.fromkeys(runs["ds"]))
    fig, axes = plt.subplots(2, len(dss), figsize=(4.3 * len(dss), 7), squeeze=False)
    panels = [("latency_pipeline_ms_median", "Per-flow latency, median (ms, 1 core)", False),
              ("size_compressed_mb", "Model size on disk (MB, log scale)", True)]
    for j, ds in enumerate(dss):
        r = runs[runs.ds == ds]
        pk = picks[picks.ds == ds]
        for i, (xcol, xlabel, logx) in enumerate(panels):
            ax = axes[i, j]
            ax.scatter(r[xcol], r["f1_macro"], s=16, color=LIGHT, zorder=2, label="other configurations")
            for (_, row), (color, face, label) in zip(pk.iterrows(), [(MAROON, MAROON, "paper setting (50 trees, no depth limit)"),
                                                                     (MAROON, "white", "smallest within 0.5 pp macro-F1")]):
                ax.scatter(row[xcol], row["f1_macro"], s=52, color=face, edgecolor=color, lw=1.8, zorder=3, label=label)
                ax.annotate(f"{row['trees']} trees, depth {row['max_depth']}", (row[xcol], row["f1_macro"]),
                            xytext=(6, -12 if face == "white" else 6), textcoords="offset points", fontsize=7.5)
            if logx:
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Macro-F1 (%)")
            ax.grid(color="#EEEEEE", lw=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(PRESETS[ds]["title"])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save_fig(fig, "edge.png")


# ================================================================ redraw figures only
def figs():
    """Redraw the figures from the CSVs already in results/ (no training; confusion matrices need a full run)."""
    def load(name):
        return pd.read_csv(OUT / name) if (OUT / name).exists() else None

    runs = load("audit_runs.csv")
    if runs is not None:
        cols = [c for c in ["accuracy", "f1_weighted", "f1_macro", "test_in_train_pct", "recall[DoS]"] if c in runs]
        m, s, _ = mean_pm_std(runs, ["ds", "dataset", "protocol"], cols)
        fig_audit(m, s)
    runs = load("improve_runs.csv")
    if runs is not None:
        for ds in dict.fromkeys(runs["ds"]):
            fig_improve(runs, ds)
    summ = load("zeroday_summary.csv")
    if summ is not None:
        fig_zeroday(summ)
    summ = load("crossday_summary.csv")
    if summ is not None:
        fig_crossday(summ, PRESETS["cic"]["normal"])
    runs = load("edge_runs.csv")
    if runs is not None:
        runs["max_depth"] = runs["max_depth"].astype(str)
        fig_edge(runs, edge_picks(runs))


# ================================================================ main
EXPERIMENTS = {"env": env, "audit": audit, "improve": improve, "zeroday": zeroday, "crossday": crossday,
               "edge": edge}
TOOLS = {"figs": figs}  # not part of "all"


def main():
    global ARGS, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=[*EXPERIMENTS, "all", *TOOLS])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--sample", type=int, default=0, help="use N random rows per dataset (smoke test)")
    ARGS = ap.parse_args()
    if ARGS.sample:
        OUT = ROOT / "results_smoke"
    todo = list(EXPERIMENTS) if ARGS.experiment == "all" else [ARGS.experiment]
    for name in todo:
        t0 = time.time()
        log(f"==== {name} ====")
        {**EXPERIMENTS, **TOOLS}[name]()
        log(f"==== {name} done in {time.time() - t0:.0f} s ====")


if __name__ == "__main__":
    main()
