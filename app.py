# app.py — SB-style radar (grey annulus bands • solid fills • no spokes • subtle tick labels • PERCENTILES 0–100 • 11 rings)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from pathlib import Path
import io
import re

st.set_page_config(page_title="Player Comparison — SB Radar", layout="wide")

# ---------------- Theme ----------------
COL_A = "#C81E1E"          # deep red
COL_B = "#1D4ED8"          # deep blue
FILL_A = (200/255, 30/255, 30/255, 0.60)   # solid-ish fill
FILL_B = (29/255, 78/255, 216/255, 0.60)

PAGE_BG   = "#FFFFFF"
AX_BG     = "#FFFFFF"

GRID_BAND_A = "#E5E7EB"    # light gray band
GRID_BAND_B = "#FFFFFF"    # white band
RING_COLOR  = "#D1D5DB"    # ring outlines
RING_LW     = 1.0

LABEL_COLOR = "#0F172A"
TITLE_FS    = 26
SUB_FS      = 12
AXIS_FS     = 10

# subtle numeric tick labels
TICK_FS     = 7
TICK_COLOR  = "#9CA3AF"    # gray-400

NUM_RINGS   = 11           # <-- Option A: rings at 0,10,20,...,100
INNER_HOLE  = 10

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
    "Non-penalty goals per 90","xG per 90","Shots per 90",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90"
]

def clean_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Successful dribbles, %", "Dribble %")
    s = s.replace("Shots on target, %", "SoT %")
    s = s.replace("Accurate passes, %", "Pass %")
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
        int(np.nanmin(df["Age"]) if pd.notna(df["Age']).any() else 14),
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
    show_avg    = st.checkbox("Show pool average (thin line)", True)

# -------------- Pool & arrays --------------
try:
    rowA = df[df["Player"] == pA].iloc[0]
    rowB = df[df["Player"] == pB].iloc[0]
except IndexError:
    st.error("Selected player not found.")
    st.stop()

# Pool limited to shared leagues + filters
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

# Numeric conversion + drop NaNs
for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")
pool = pool.dropna(subset=metrics)
if pool.empty:
    st.warning("No players remain in pool after filters.")
    st.stop()

labels = [clean_label(m) for m in metrics]

# ---- Percentile scaling (0–100) ----
pool_pct = pool[metrics].rank(pct=True) * 100.0  # 0..100 within the pool

def pct_for(player: str) -> np.ndarray:
    sub_idx = pool[pool["Player"] == player].index
    if len(sub_idx) == 0:
        return np.full(len(metrics), np.nan)
    # If multiple rows for a player, average their percentile rows
    return pool_pct.loc[sub_idx, :].mean(axis=0).values

A_r = pct_for(pA)
B_r = pct_for(pB)

# Pool "average" as 50th percentile line
AVG_r = np.full(len(metrics), 50.0)

# Sort axes by percentile gap if requested
if sort_by_gap:
    order = np.argsort(-np.abs(A_r - B_r))
    labels = [labels[i] for i in order]
    A_r    = A_r[order]
    B_r    = B_r[order]
    AVG_r  = AVG_r[order]

# Radii for 11 rings (0..100 by tens) and tick values in tens
ring_radii = np.linspace(INNER_HOLE, 100, NUM_RINGS)
axis_ticks = [np.linspace(0, 100, NUM_RINGS) for _ in range(len(labels))]

# -------------- Radar drawer --------------
def draw_radar(labels, A_r, B_r, ticks, headerA, subA, headerB, subB,
               show_avg=False, AVG_r=None):
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    Ar = np.concatenate([A_r, A_r[:1]])
    Br = np.concatenate([B_r, B_r[:1]])

    fig = plt.figure(figsize=(13.2, 8.0), dpi=260)
    fig.patch.set_facecolor(PAGE_BG)

    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(AX_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Perimeter labels; no polar grid/spokes
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    # Alternating annulus bands (SB look)
    for i in range(NUM_RINGS-1):
        r0, r1 = ring_radii[i], ring_radii[i+1]
        band = GRID_BAND_A if i % 2 == 0 else GRID_BAND_B
        ax.add_artist(Wedge((0,0), r1, 0, 360, width=(r1-r0),
                            transform=ax.transData._b, facecolor=band,
                            edgecolor="none", zorder=0.8))

    # Subtle ring outlines only
    ring_t = np.linspace(0, 2*np.pi, 361)
    for r in ring_radii:
        ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=0.9)

    # Tiny, light numeric labels from the 3rd ring outward (20..100)
    start_idx = 2  # 0-based; so rings 2..10 => 20..100
    for i, ang in enumerate(theta):
        vals = ticks[i][start_idx:]
        for rr, v in zip(ring_radii[start_idx:], vals):
            ax.text(ang, rr-1.8, f"{v:.0f}", ha="center", va="center",  # integers
                    fontsize=TICK_FS, color=TICK_COLOR, zorder=1.1)

    # Clean inner hole
    ax.add_artist(Circle((0,0), radius=INNER_HOLE-0.6, transform=ax.transData._b,
                         color=PAGE_BG, zorder=1.2, ec="none"))

    # Optional pool average (50th pct) as thin line
    if show_avg and AVG_r is not None:
        Avg = np.concatenate([AVG_r, AVG_r[:1]])
        ax.plot(theta_closed, Avg, lw=1.5, color="#94A3B8", ls="--", alpha=0.9, zorder=2.2)

    # Player polygons — solid fills, dark borders
    ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=3)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=2.5)

    ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=3)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=2.5)

    ax.set_rlim(0, 105)

    # Headers
    fig.text(0.12, 0.96, headerA, color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
    fig.text(0.12, 0.935, subA,    color=COL_A, fontsize=SUB_FS,      ha="left")
    fig.text(0.88, 0.96, headerB, color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
    fig.text(0.88, 0.935, subB,   color=COL_B, fontsize=SUB_FS,      ha="right")

    if show_avg and AVG_r is not None:
        fig.text(0.5, 0.09, "— Pool average (50th percentile)", color="#6B7280", fontsize=10, ha="center")

    return fig

headerA = f"{pA}"
subA    = f"{rowA['Team']} — {rowA['League']}"
headerB = f"{pB}"
subB    = f"{rowB['Team']} — {rowB['League']}"

fig = draw_radar(labels, A_r, B_r, axis_ticks, headerA, subA, headerB, subB,
                 show_avg=show_avg, AVG_r=AVG_r)
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




