"""Live IoT NIDS replay: streams held-out flows through the trained RF as if they were live traffic."""
import time
from collections import deque
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nids_core import transform  # noqa: E402

MAROON, ROSE, GRAY = "#800400", "#D98C8C", "#595959"
MODELS = Path(__file__).parent / "models"
st.set_page_config(page_title="IoT NIDS Live Replay", layout="wide")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


def build_order(bundle, mode, seed=0):
    rng = np.random.default_rng(seed)
    y = bundle["test_y"].to_numpy()
    if mode.startswith("Natural"):
        return rng.permutation(len(y))
    per = min(2000, min((y == c).sum() for c in bundle["classes"]))
    idx = np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in bundle["classes"]])
    return rng.permutation(idx)


def reset(key, bundle, mode):
    st.session_state.update(
        key=key, order=build_order(bundle, mode), pos=0, y_true=[], y_pred=[],
        feed=deque(maxlen=14), alarms=deque(maxlen=8), lat=[], running=False)


# ---------------------------------------------------------------- sidebar
files = sorted(MODELS.glob("*.joblib"))
if not files:
    st.error("No trained models found in models/. Run train.py first.")
    st.stop()
st.sidebar.title("Replay controls")
choice = st.sidebar.selectbox("Dataset / model", files, format_func=lambda f: load_bundle(f)["title"])
bundle = load_bundle(choice)
mode = st.sidebar.radio("Traffic mix", ["Natural mix (as in test set)", "Balanced (equal per class)"])
per_tick = st.sidebar.slider("Flows per tick", 1, 200, 20)
tick = st.sidebar.slider("Seconds per tick", 0.2, 2.0, 0.5, 0.1)

key = f"{choice}|{mode}"
if st.session_state.get("key") != key:
    reset(key, bundle, mode)
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Start", type="primary", width="stretch"):
    st.session_state.running = True
if c2.button("Pause", width="stretch"):
    st.session_state.running = False
if c3.button("Reset", width="stretch"):
    reset(key, bundle, mode)

m = bundle["metrics"]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** Random Forest, {bundle['model'].n_estimators} trees  \n"
                    f"**Features:** {m['kept']} of {m['total_features']} kept by PCC (|r| > {m['pcc']})  \n"
                    f"**Train / test:** {m['n_train']:,} / {m['n_test']:,} flows  \n"
                    f"**Training time:** {m['train_seconds']:.1f} s")
UNSEEN = set(bundle.get("unseen", []))
if UNSEEN:
    st.sidebar.markdown(f"**Unseen in training:** {', '.join(sorted(UNSEEN))} (zero-day test)")

st.title("IoT Network Intrusion Detection: Live Replay")
st.caption(f"{bundle['title']}: held-out test flows are streamed through the trained Random Forest "
           "as if they were arriving at an IoT edge gateway.")


# ---------------------------------------------------------------- live view
def step():
    s = st.session_state
    if not s.running or s.pos >= len(s.order):
        s.running = s.running and s.pos < len(s.order)
        return
    idx = s.order[s.pos:s.pos + per_tick]
    s.pos += len(idx)
    raw = bundle["test_raw"].iloc[idx]
    t0 = time.perf_counter()
    pred = bundle["model"].predict(transform(bundle, raw))
    s.lat.append((time.perf_counter() - t0) / len(idx) * 1000)
    truth = bundle["test_y"].iloc[idx].to_numpy()
    normal = bundle["normal"]
    for i, (t, p) in enumerate(zip(truth, pred)):
        n = s.pos - len(idx) + i + 1
        if t == p:
            status = "ALARM: attack caught" if p != normal else "ok"
        elif p == normal:
            status = "MISSED attack"
        elif t == normal:
            status = "FALSE alarm"
        else:
            status = "ALARM: wrong attack type"
        s.feed.appendleft(dict(flow=n, true=t, predicted=p, status=status))
        if p != normal:
            s.alarms.appendleft(f"#{n}: **{p}** detected" + ("" if t == p else f" (actually {t})"))
    s.y_true.extend(truth)
    s.y_pred.extend(pred)


def color_status(v):
    if v.startswith("MISSED"):
        return "background-color:#F8D7DA;color:#800400;font-weight:bold"
    if v.startswith("FALSE") or "wrong" in v:
        return "background-color:#FFF3CD"
    if v.startswith("ALARM"):
        return "background-color:#F7E6E4;color:#800400"
    return "color:#595959"


def confusion_fig(yt, yp, classes):
    cm = pd.crosstab(pd.Series(yt, name="true"), pd.Series(yp, name="pred")).reindex(
        index=classes, columns=classes, fill_value=0).to_numpy()
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.imshow(pct, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("m", ["#FFFFFF", ROSE, MAROON]),
              vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if pct[i, j] > 60 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Running confusion matrix (row %)", loc="left", fontsize=11, color=MAROON)
    fig.tight_layout()
    return fig


def recall_fig(yt, yp, classes):
    yt, yp = np.array(yt), np.array(yp)
    normal = bundle["normal"]
    # an unseen (zero-day) class can never get its exact label, so show how often it raised any alarm
    rec = [np.mean(yp[yt == c] != normal) * 100 if c in UNSEEN and (yt == c).any()
           else ((yp == c) & (yt == c)).sum() / max((yt == c).sum(), 1) * 100 for c in classes]
    labels = [f"{c} (unseen: any alarm)" if c in UNSEEN else c for c in classes]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    bars = ax.barh(labels, rec, color=[ROSE if c in UNSEEN else GRAY if c == normal else MAROON for c in classes])
    for b, r in zip(bars, rec):
        inside = r > 85
        ax.text(r - 1 if inside else r + 1, b.get_y() + b.get_height() / 2, f"{r:.1f}%", va="center",
                ha="right" if inside else "left", fontsize=9, color="white" if inside else "black")
    ax.set_xlim(0, 105); ax.invert_yaxis(); ax.set_xlabel("Recall (%)")
    ax.set_title("Per-class detection rate so far", loc="left", fontsize=11, color=MAROON)
    fig.tight_layout()
    return fig


@st.fragment(run_every=tick)
def live():
    step()
    s = st.session_state
    yt, yp, normal = np.array(s.y_true), np.array(s.y_pred), bundle["normal"]
    n = len(yt)
    att = yt != normal
    k = st.columns(6)
    k[0].metric("Flows processed", f"{n:,} / {len(s.order):,}")
    k[1].metric("Alarms raised", f"{int((yp != normal).sum()):,}" if n else "0")
    k[2].metric("Accuracy so far", f"{(yt == yp).mean() * 100:.2f}%" if n else "–")
    k[3].metric("Attack detection rate", f"{(yp[att] != normal).mean() * 100:.2f}%" if att.any() else "–")
    k[4].metric("False-alarm rate", f"{(yp[~att] != normal).mean() * 100:.2f}%" if (~att).any() else "–")
    k[5].metric("Latency per flow", f"{np.mean(s.lat[-20:]):.3f} ms" if s.lat else "–")
    st.progress(s.pos / len(s.order), text="Running…" if s.running else ("Finished" if s.pos else "Press Start"))

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Incoming traffic")
        if s.feed:
            st.dataframe(pd.DataFrame(s.feed).style.map(color_status, subset=["status"]),
                         hide_index=True, width="stretch", height=520)
        else:
            st.info("Waiting for traffic. Press Start in the sidebar.")
    with right:
        if n:
            st.pyplot(confusion_fig(yt, yp, bundle["classes"]), clear_figure=True)
            st.pyplot(recall_fig(yt, yp, bundle["classes"]), clear_figure=True)
            plt.close("all")

    a, b = st.columns([1, 1.4])
    with a:
        st.subheader("Latest alarms")
        st.markdown("\n".join(f"- {x}" for x in s.alarms) if s.alarms else "_No alarms yet._")
    with b:
        st.subheader("Paper vs. our run")
        st.dataframe(pd.DataFrame({
            "": ["Accuracy (%)", "Weighted F1 (%)"],
            "Paper (Table 6)": [m["paper_acc"], m["paper_f1"]],
            "Our offline test (full split)": [round(m["accuracy"], 2), round(m["f1_weighted"], 2)],
            "Live replay so far": [round((yt == yp).mean() * 100, 2) if n else None,
                                   round(f1_score(yt, yp, average="weighted", zero_division=0) * 100, 2) if n else None],
        }), hide_index=True, width="stretch")
        if mode.startswith("Balanced"):
            st.caption("Balanced mix: attacks are over-represented on purpose, so live accuracy "
                       "is not comparable with the paper's natural-mix numbers.")
        if UNSEEN:
            st.caption(f"Zero-day model: {', '.join(sorted(UNSEEN))} was removed from training, so its flows "
                       "can only be caught as 'some other attack'. Accuracy is not comparable with the paper.")


live()
