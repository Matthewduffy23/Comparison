# app.py — StatsBomb-like radar (alt. sector shading • 10 rings • inline ticks • dark red/blue)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from pathlib import Path
import io
import re

st.set_page_config(page_title="Player Comparison — SB-style Radar", layout="wide")

# ---------------- Theme ----------------
COL_A = "#DF3B37"          # SB dark red
COL_B = "#2D6DB7"          # SB dark blue
FILL_A = (223/255, 59/255, 55/255, 0.22)
FILL_B = (45/255, 109/255, 183/255, 0.22)

PAGE_BG   = "#FFFFFF"
DISC_BG   = "#E7EBF1"      # chart disc
SECTOR_A  = "#F7F8FA"      # alternating sector shades
SECTOR_B  = "#EFF2F6"
RAY_COLOR = "#D5D9E0"      # spoke lines
RING_COLOR= "#C4CAD2"      # ring lines
RING_LW   = 1.0

LABEL_COLOR = "#0F172A"
TITLE_FS    = 26
SUB_FS      = 12
AXIS_FS     = 11
TICK_FS     = 8.5

NUM_RINGS   = 10           # << 10 separation rings

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

# Defaults (no Key passes per 90)
DEFAULT_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90","Shots on target, %",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90"
]

# Label cleaner: remove " per 90", tidy names
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

# per-axis min/max for normalization, small padding
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
    labels = [labels[i] for i in order]
    A_r    = A_r[order]
    B_r    = B_r[order]
    A_val  = A_val[order]
    B_val  = B_val[order]
    axis_min = axis_min[order]; axis_max = axis_max[order]

# tick values for inline spoke labels (10 rings => 10 ticks)
ring_radii = np.linspace(10, 100, NUM_RINGS)   # keep small inner hole at 10
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

    # axis labels only (no radial ticks)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
    ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # disc
    disc = Circle((0,0), radius=102, transform=ax.transData._b, color=DISC_BG, zorder=0)
    ax.add_artist(disc)

    # --- alternating sector shading like SB ---
    # boundaries are midpoints between axes
    bounds = np.linspace(0, 2*np.pi, N, endpoint=False)
    bounds = np.concatenate([bounds, [bounds[0] + 2*np.pi/N]])
    starts = (bounds[:-1] - (np.pi/N)/2)
    ends   = (bounds[:-1] + (np.pi/N)/2)
    # normalize to degrees
    starts_deg = np.degrees(starts)
    ends_deg   = np.degrees(ends)
    for i in range(N):
        color = SECTOR_A if i % 2 == 0 else SECTOR_B
        ax.add_artist(Wedge((0,0), 100, starts_deg[i], ends_deg[i], width=100, facecolor=color,
                            edgecolor="none", zorder=1.2))

    # spoke rays
    for ang in theta:
        ax.plot([ang, ang], [10, 100], color=RAY_COLOR, lw=1.0, zorder=2)

    # 10 separation rings
    ring_t = np.linspace(0, 2*np.pi, 361)
    for r in ring_radii:
        ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=2)

    # per-spoke value ticks (small, neat numbers at each ring)
    for i, ang in enumerate(theta):
        vals = ticks[i]
        for rr, v in zip(ring_radii, vals):
            txt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
            ax.text(ang, rr-2.2, txt, ha="center", va="center",
                    fontsize=TICK_FS, color="#5B6470", rotation=0, zorder=3)

    # polygons (white under-stroke for crisp edge)
    ax.plot(theta_closed, Ar, color="white", lw=5.0, zorder=5)
    ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=6)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=4)

    ax.plot(theta_closed, Br, color="white", lw=5.0, zorder=5)
    ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=6)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=4)

    ax.set_rlim(0, 105)

    # titles (left & right, SB vibe)
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
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar_SB_style.png",
                   mime="image/png")

buf_svg = io.BytesIO()
fig.savefig(buf_svg, format="svg", bbox_inches="tight")
st.download_button("⬇️ Download SVG", data=buf_svg.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar_SB_style.svg",
                   mime="image/svg+xml")






