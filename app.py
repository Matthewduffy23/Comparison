# app.py — StatsBomb-like radar: white/grey rings • dark red/blue • 10 rings • tidy ticks
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from pathlib import Path
import io
import re

st.set_page_config(page_title="Player Comparison — SB Radar", layout="wide")

# ---------------- Theme ----------------
# Darker, TV-safe colours
COL_A = "#C62828"          # deep red
COL_B = "#1F4E8C"          # deep blue
FILL_A = (198/255, 40/255, 40/255, 0.23)
FILL_B = (31/255, 78/255, 140/255, 0.23)

PAGE_BG   = "#FFFFFF"
DISC_BG   = "#FFFFFF"      # centre disc base (kept white)
RING_FILL_A = "#FFFFFF"    # alternating ring fills
RING_FILL_B = "#EEF1F4"    # light grey (like SB)
RAY_COLOR = "#C9CED6"      # spoke lines
RING_EDGE = "#C9CED6"      # ring borders
RING_LW   = 1.0

LABEL_COLOR = "#0F172A"
TITLE_FS    = 26
SUB_FS      = 12
AXIS_FS     = 10           # compact axis labels
TICK_FS     = 8            # compact tick labels

NUM_RINGS   = 10
INNER_HOLE  = 10           # small central hole radius (keeps middle clean)

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

DEFAULT_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90","Shots on target, %",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90"
]

def clean_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "NP goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Successful dribbles, %", "Dribble %")
    s = s.replace("Accurate passes, %", "Pass %")
    s = s.replace("Shots on target, %", "SoT %")
    s = re.sub(r"\s*per\s*90", "", s, flags=re.I)
    return s

# -------------- Sidebar --------------
with st.sidebar:
    st.header("Controls")

    pos_scope = st.text_input("Position startswith", "CF")
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"]            = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider("Minutes filter", 0, 5000, (500, 5000))
    min_age, max_age         = st.slider(
        "Age filter",
        int(np.nanmin(df["Age"]) if pd.notna(df["Age"]).any() else 14),
        int(np.nanmax(df["Age"]) if pd.notna(df["Age"]).any() else 40),
        (16, 33)
    )

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

labels = [clean_label(m) for m in metrics]

if sort_by_gap:
    order = np.argsort(-np.abs(A_r - B_r))
    labels  = [labels[i] for i in order]
    A_r     = A_r[order]
    B_r     = B_r[order]
    A_val   = A_val[order]
    B_val   = B_val[order]
    axis_min = axis_min[order]; axis_max = axis_max[order]

# ring radii and per-spoke tick values
ring_radii = np.linspace(INNER_HOLE, 100, NUM_RINGS)
axis_ticks = [np.linspace(axis_min[i], axis_max[i], NUM_RINGS) for i in range(len(labels))]

# -------------- Radar drawer --------------
def draw_radar(labels, A_r, B_r, ticks, headerA, subA, headerB, subB):
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

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
    ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # --- alternating white/grey rings (like SB) ---
    ring_edges = np.linspace(INNER_HOLE, 100, NUM_RINGS)
    last = 0.0
    for i, r in enumerate(ring_edges, start=1):
        fill = RING_FILL_A if i % 2 else RING_FILL_B
        ax.add_artist(Wedge((0,0), r, 0, 360, width=r-last,
                            facecolor=fill, edgecolor=RING_EDGE, lw=RING_LW, zorder=1))
        last = r

    # spoke rays on top
    for ang in theta:
        ax.plot([ang, ang], [INNER_HOLE, 100], color=RAY_COLOR, lw=1.0, zorder=2)

    # per-spoke values (1 decimal) from 3rd ring outward
    start_idx = 2
    for i, ang in enumerate(theta):
        vals = ticks[i][start_idx:]
        for rr, v in zip(ring_radii[start_idx:], vals):
            ax.text(ang, rr-2.0, f"{v:.1f}", ha="center", va="center",
                    fontsize=TICK_FS, color="#5B6470", rotation=0, zorder=3)

    # polygons (white under-stroke + dark coloured edge + translucent fill)
    ax.plot(theta_closed, Ar, color="white", lw=5.0, zorder=5)
    ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=6)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=4)

    ax.plot(theta_closed, Br, color="white", lw=5.0, zorder=5)
    ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=6)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=4)

    ax.set_rlim(0, 105)

    # headers
    fig.text(0.12, 0.96, headerA, color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
    fig.text(0.12, 0.935, subA,    color=COL_A, fontsize=SUB_FS,      ha="left")
    fig.text(0.88, 0.96, headerB, color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
    fig.text(0.88, 0.935, subB,   color=COL_B, fontsize=SUB_FS,      ha="right")

    return fig

headerA = f"{pA}"
subA    = f"{rowA['Team']} — {rowA['League']}"
headerB = f"{pB}"
subB    = f"{rowB['Team']} — {rowB['League']}"

fig = draw_radar(labels, A_r, B_r, axis_ticks, headerA, subA, headerB, subB)
st.pyplot(fig, use_container_width=True)

# -------------- Exports --------------
buf_png = io.BytesIO()
fig.savefig(buf_png, format="png", dpi=340, bbox_inches="tight")
st.download_button("⬇️ Download PNG", data=buf_png.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar_SB.png",
                   mime="image/png")

buf_svg = io.BytesIO()
fig.savefig(buf_svg, format="svg", bbox_inches="tight")
st.download_button("⬇️ Download SVG", data=buf_svg.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar_SB.svg",
                   mime="image/svg+xml")








