# app.py — Player Comparison (Classic Radar)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io

st.set_page_config(page_title="Player Comparison — Radar", layout="wide")

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_df():
    p = Path(__file__).with_name("WORLDJUNE25.csv")
    if p.exists():
        return pd.read_csv(p)
    return None

df = load_df()
if df is None:
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not up:
        st.warning("Upload dataset to continue.")
        st.stop()
    df = pd.read_csv(up)

# sanity
need = {"Player","League","Team","Position","Minutes played","Age"}
missing = [c for c in need if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ---------- Defaults (13 metrics) ----------
DEFAULT_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90","Shots on target, %",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90","Key passes per 90",
]

SHORT = {
    "Non-penalty goals per 90":"NP goals/90",
    "Shots on target, %":"SoT %",
    "Successful dribbles, %":"Dribble %",
    "Accurate passes, %":"Pass %",
    "Touches in box per 90":"Box touches/90",
    "Aerial duels won, %":"Aerials won %",
}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")

    pos_scope = st.text_input("Position startswith", "CF")

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider("Minutes filter", 0, int(df["Minutes played"].max() or 99999), (500, 99999))
    min_age, max_age = st.slider("Age filter", int(df["Age"].min() or 14), int(df["Age"].max() or 40), (16, 33))

    picker_pool = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))].copy()
    players = sorted(picker_pool["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players for this filter.")
        st.stop()

    pA = st.selectbox("Player A (red)", players, index=0)
    pB = st.selectbox("Player B (blue)", players, index=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    default_metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics (13 recommended)", [c for c in df.columns if c in numeric_cols], default_metrics)
    if len(metrics) < 3:
        st.warning("Pick at least 3 metrics.")
        st.stop()

    show_values = st.checkbox("Annotate percentile values", True)
    rings_step = st.select_slider("Ring step (%)", options=[5,10,20], value=10)

# ---------- Percentile pool: union of the two players’ leagues ----------
try:
    rowA = df[df["Player"] == pA].iloc[0]
    rowB = df[df["Player"] == pB].iloc[0]
except IndexError:
    st.error("Selected player not found.")
    st.stop()

union_leagues = {rowA["League"], rowB["League"]}

pool = df[
    (df["League"].isin(union_leagues)) &
    (df["Position"].astype(str).str.startswith(tuple([pos_scope]))) &
    (df["Minutes played"].between(min_minutes, max_minutes)) &
    (df["Age"].between(min_age, max_age))
].copy()

# numeric + dropna on selected metrics
missing_m = [m for m in metrics if m not in pool.columns]
if missing_m:
    st.error(f"Missing metric columns: {missing_m}")
    st.stop()
for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")
pool = pool.dropna(subset=metrics)
if pool.empty:
    st.warning("No players left in pool after filters.")
    st.stop()

# Percentiles 0..100 within pool
ranks = pool[metrics].rank(pct=True) * 100
pool_pct = pd.concat([pool[["Player"]], ranks], axis=1)

def pct_for(name: str):
    sub = pool_pct[pool_pct["Player"] == name][metrics]
    if sub.empty:
        return np.full(len(metrics), np.nan)
    return sub.mean().values

A_pct = pct_for(pA)
B_pct = pct_for(pB)

# also grab team/league for title line
teamA, leagueA = rowA["Team"], rowA["League"]
teamB, leagueB = rowB["Team"], rowB["League"]

# ---------- Radar plotting (matplotlib) ----------
def plot_radar(ax, labels, A, B, colors=("crimson","royalblue"),
               ring_step=10, show_vals=True):
    """
    Classic radar: grey rings, two filled polygons, tidy labels.
    A, B are 0..100 percentiles (len == len(labels)).
    """
    # angles
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # close the values loop
    A = np.asarray(A, dtype=float).tolist()
    B = np.asarray(B, dtype=float).tolist()
    A += A[:1]; B += B[:1]

    # style
    ax.set_theta_offset(np.pi / 2)   # start at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_xticks(np.linspace(0, 2*np.pi, N, endpoint=False))
    ax.set_xticklabels(labels, fontsize=11, fontweight=600, color="#333")

    # rings
    max_r = 100
    ax.set_rlabel_position(0)
    ring_vals = list(range(ring_step, max_r+ring_step, ring_step))
    ax.set_yticks(ring_vals)
    ax.set_yticklabels([str(v) for v in ring_vals], fontsize=9, color="#555")
    ax.set_ylim(0, max_r)

    # grid styling
    ax.grid(color="lightgrey", linestyle="-", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # fill polygons
    colA, colB = colors
    ax.plot(angles, A, color=colA, linewidth=2.5)
    ax.fill(angles, A, color=colA, alpha=0.25, zorder=2)
    ax.plot(angles, B, color=colB, linewidth=2.5)
    ax.fill(angles, B, color=colB, alpha=0.25, zorder=2)

    # value annotations
    if show_vals:
        for ang, val in zip(angles[:-1], A[:-1]):
            if val >= 8:
                ax.text(ang, val+2, f"{val:.0f}", color=colA, fontsize=9, ha="center", va="center")
        for ang, val in zip(angles[:-1], B[:-1]):
            if val >= 8:
                ax.text(ang, max(val-7, 0)+2, f"{val:.0f}", color=colB, fontsize=9, ha="center", va="center")

# labels (shorten where mapped)
labels = [SHORT.get(m, m) for m in metrics]

# order = as-chosen; if you want biggest-gap first, uncomment:
# gap = np.abs(A_pct - B_pct); order = np.argsort(-gap)
# labels = [labels[i] for i in order]; A_pct = A_pct[order]; B_pct = B_pct[order]

fig = plt.figure(figsize=(9.5, 10), dpi=180)
ax = plt.subplot(111, polar=True)
plot_radar(ax, labels, A_pct, B_pct, colors=("#E64B3C","#1F77B4"),
           ring_step=rings_step, show_vals=show_values)

# header titles like your example
fig.suptitle("", y=0.98)
ax.set_title("", pad=20)

# Left (A) and right (B) title blocks
fig.text(0.12, 0.96, pA, color="#E64B3C", fontsize=22, fontweight="bold", ha="left")
fig.text(0.12, 0.93, f"{teamA} — {leagueA}", color="#E64B3C", fontsize=12, ha="left")
fig.text(0.88, 0.96, pB, color="#1F77B4", fontsize=22, fontweight="bold", ha="right")
fig.text(0.88, 0.93, f"{teamB} — {leagueB}", color="#1F77B4", fontsize=12, ha="right")

st.pyplot(fig, use_container_width=True)

# download as PNG
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button("⬇️ Download radar (PNG)", data=buf.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar.png",
                   mime="image/png")

