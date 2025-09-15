# app.py — StatsBomb-style radar ONLY (no numbers, dark red/blue)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from pathlib import Path
import io

st.set_page_config(page_title="Player Comparison — Radar", layout="wide")

# ---------------- Theme ----------------
COL_A = "#DF3B37"          # dark SB-like red
COL_B = "#2D6DB7"          # dark SB-like blue
FILL_A = (223/255, 59/255, 55/255, 0.22)
FILL_B = (45/255, 109/255, 183/255, 0.22)

PAGE_BG   = "#FFFFFF"
DISC_BG   = "#E5E9EF"      # chart disc
RAY_COLOR = "#D4D9E0"      # spoke lines
RING_COLOR= "#C8CDD3"      # ring lines
RING_LW   = 1.0

LABEL_COLOR = "#111827"
TITLE_FS    = 26
SUB_FS      = 12
AXIS_FS     = 11

# -------------- Data ---------------
@st.cache_data(show_spinner=False)
def load_df():
    p = Path(__file__).with_name("WORLDJUNE25.csv")
    return pd.read_csv(p) if p.exists() else None

df = load_df()
if df is None:
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not up:
        st.warning("Upload dataset to continue.")
        st.stop()
    df = pd.read_csv(up)

required = {"Player","League","Team","Position","Minutes played","Age"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# No “Key passes per 90” by default
DEFAULT_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90","Shots on target, %",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90"
]
SHORT = {
    "Non-penalty goals per 90":"NP goals/90",
    "Shots on target, %":"SoT %",
    "Successful dribbles, %":"Dribble %",
    "Accurate passes, %":"Pass %",
    "Touches in box per 90":"Box touches/90",
    "Aerial duels won, %":"Aerials won %",
}

# -------------- Sidebar --------------
with st.sidebar:
    st.header("Controls")

    pos_scope = st.text_input("Position startswith", "CF")
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"]            = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider("Minutes filter", 0, 5000, (500, 5000))
    min_age, max_age         = st.slider("Age filter",
                                         int(np.nanmin(df["Age"]) if pd.notna(df["Age"]).any() else 14),
                                         int(np.nanmax(df["Age"]) if pd.notna(df["Age"]).any() else 40),
                                         (16, 33))

    picker_pool = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))].copy()
    players = sorted(picker_pool["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players for this filter.")
        st.stop()

    pA = st.selectbox("Player A (red)", players, index=0)
    pB = st.selectbox("Player B (blue)", players, index=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics_default = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics", [c for c in df.columns if c in numeric_cols], metrics_default)
    if len(metrics) < 5:
        st.warning("Pick at least 5 metrics.")
        st.stop()

    sort_by_gap = st.checkbox("Sort axes by biggest gap", False)

# -------------- Pool & arrays --------------
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

missing_m = [m for m in metrics if m not in pool.columns]
if missing_m:
    st.error(f"Missing metric columns: {missing_m}")
    st.stop()

for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")
pool = pool.dropna(subset=metrics)
if pool.empty:
    st.warning("No players remain in pool after filters.")
    st.stop()

def vals_for(name: str) -> np.ndarray:
    sub = pool[pool["Player"] == name][metrics]
    return sub.mean().values if not sub.empty else np.full(len(metrics), np.nan)

A_val = vals_for(pA)
B_val = vals_for(pB)

# per-axis min/max -> normalize to [0..100] radius
axis_min = pool[metrics].min().values
axis_max = pool[metrics].max().values
pad = (axis_max - axis_min) * 0.07
axis_min = axis_min - pad
axis_max = axis_max + pad

def normalize(vals, mn, mx):
    rng = (mx - mn)
    rng[rng == 0] = 1.0
    return np.clip((vals - mn)/rng, 0, 1)

A_r = normalize(A_val, axis_min, axis_max) * 100
B_r = normalize(B_val, axis_min, axis_max) * 100

labels = [SHORT.get(m, m) for m in metrics]
if sort_by_gap:
    order = np.argsort(-np.abs(A_r - B_r))
    labels = [labels[i] for i in order]
    A_r    = A_r[order]
    B_r    = B_r[order]

# -------------- Radar drawer --------------
def draw_radar(labels, A_r, B_r, headerA, subA, headerB, subB):
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    Ar = np.concatenate([A_r, A_r[:1]])
    Br = np.concatenate([B_r, B_r[:1]])

    fig = plt.figure(figsize=(13.2, 8.0), dpi=260)
    fig.patch.set_facecolor(PAGE_BG)

    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(PAGE_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # axis labels only (no radial ticks)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
    ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # disc background
    disc = Circle((0,0), radius=102, transform=ax.transData._b, color=DISC_BG, zorder=0)
    ax.add_artist(disc)

    # subtle alternating bands (like SB separation, but no numbers)
    bands = [(25, "#FFFFFF"), (50, "#F2F5F9"), (75, "#E9EDF3"), (100, "#E1E6EE")]
    inner = 0
    for r, color in bands:
        ax.add_artist(Wedge((0,0), r, 0, 360, width=r-inner, facecolor=color, edgecolor="none", zorder=1))
        inner = r

    # spoke rays
    for ang in theta:
        ax.plot([ang, ang], [0, 100], color=RAY_COLOR, lw=1.0, zorder=2)

    # many light rings (no numbers)
    ring_t = np.linspace(0, 2*np.pi, 361)
    for r in np.linspace(20, 100, 5):  # 5 rings
        ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=2)

    # polygons (double-stroke + fill)
    ax.plot(theta_closed, Ar, color="white", lw=5.0, zorder=5)
    ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=6)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=4)

    ax.plot(theta_closed, Br, color="white", lw=5.0, zorder=5)
    ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=6)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=4)

    ax.set_rlim(0, 105)

    # titles (left & right)
    fig.text(0.12, 0.96, headerA, color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
    fig.text(0.12, 0.935, subA,    color=COL_A, fontsize=SUB_FS,      ha="left")
    fig.text(0.88, 0.96, headerB, color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
    fig.text(0.88, 0.935, subB,   color=COL_B, fontsize=SUB_FS,      ha="right")

    return fig

headerA = f"{pA}"
subA    = f"{rowA['Team']} — {rowA['League']}"
headerB = f"{pB}"
subB    = f"{rowB['Team']} — {rowB['League']}"

fig = draw_radar(labels, A_r, B_r, headerA, subA, headerB, subB)
st.pyplot(fig, use_container_width=True)

# -------------- Exports --------------
buf_png = io.BytesIO()
fig.savefig(buf_png, format="png", dpi=340, bbox_inches="tight")
st.download_button("⬇️ Download PNG", data=buf_png.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar.png",
                   mime="image/png")

buf_svg = io.BytesIO()
fig.savefig(buf_svg, format="svg", bbox_inches="tight")
st.download_button("⬇️ Download SVG", data=buf_svg.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar.svg",
                   mime="image/svg+xml")





