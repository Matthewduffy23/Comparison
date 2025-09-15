# app.py — StatsBomb-style radar (two players, per-metric scales + table)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pathlib import Path
import io

st.set_page_config(page_title="Player Comparison — StatsBomb-style", layout="wide")

# ------------------------------- THEME ---------------------------------
COL_A = "#DF3B37"      # SB-ish red
COL_B = "#2D6DB7"      # SB-ish blue
FILL_A = (223/255, 59/255, 55/255, 0.20)
FILL_B = (45/255, 109/255, 183/255, 0.20)
EDGE_A = COL_A
EDGE_B = COL_B

RING_COLOR = "#C8CDD3"
RING_LW = 1.0
RAY_COLOR = "#D6DADF"
RAY_LW = 1.0
AXIS_TXT = "#1F2937"
TICK_TXT = "#4B5563"
TITLE_RED = COL_A
TITLE_BLUE = COL_B
PAGE_BG = "#FFFFFF"

# ------------------------------- DATA ----------------------------------
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

# defaults (no Key passes)
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

# ------------------------------- SIDEBAR -------------------------------
with st.sidebar:
    st.header("Controls")

    pos_scope = st.text_input("Position startswith", "CF")
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider("Minutes filter", 0, 5000, (500, 5000))
    min_age, max_age = st.slider("Age filter",
                                 int(np.nanmin(df["Age"]) if pd.notna(df["Age"]).any() else 14),
                                 int(np.nanmax(df["Age"]) if pd.notna(df["Age"]).any() else 40),
                                 (16, 33))

    pool_pick = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))].copy()
    players = sorted(pool_pick["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players for this filter.")
        st.stop()

    pA = st.selectbox("Player A (red)", players, index=0)
    pB = st.selectbox("Player B (blue)", players, index=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics_default = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics (StatsBomb-style)", [c for c in df.columns if c in numeric_cols], metrics_default)
    if len(metrics) < 5:
        st.warning("Pick at least 5 metrics for a readable radar.")
        st.stop()

    sort_by_gap = st.checkbox("Sort axes by biggest gap", False)

# ------------------------------- BUILD POOL ----------------------------
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

# percentiles for table; and raw values for scale
ranks = pool[metrics].rank(pct=True) * 100
pool_pct = pd.concat([pool[["Player"]], ranks], axis=1)

def pct_for(name: str) -> np.ndarray:
    sub = pool_pct[pool_pct["Player"] == name][metrics]
    return sub.mean().values if not sub.empty else np.full(len(metrics), np.nan)

def values_for(name: str) -> np.ndarray:
    sub = pool[pool["Player"] == name][metrics]
    return sub.mean().values if not sub.empty else np.full(len(metrics), np.nan)

A_pct = pct_for(pA)
B_pct = pct_for(pB)
A_val = values_for(pA)
B_val = values_for(pB)

# per-metric scales (min..max in pool) + ticks
axis_min = pool[metrics].min().values
axis_max = pool[metrics].max().values
# small padding so polygons don’t sit on ring
pad = (axis_max - axis_min) * 0.07
axis_min = axis_min - pad
axis_max = axis_max + pad
# ticks per axis (5 rings -> 5 tick labels) like SB
ticks = [np.linspace(axis_min[i], axis_max[i], 6) for i in range(len(metrics))]

labels = [SHORT.get(m, m) for m in metrics]

if sort_by_gap:
    order = np.argsort(-np.abs(A_pct - B_pct))
    labels  = [labels[i] for i in order]
    metrics = [metrics[i] for i in order]
    A_pct   = A_pct[order]
    B_pct   = B_pct[order]
    A_val   = A_val[order]
    B_val   = B_val[order]
    axis_min = axis_min[order]; axis_max = axis_max[order]
    ticks    = [ticks[i] for i in order]

# map raw value to radius [0..1]
def normalize(vals, mn, mx):
    rng = (mx - mn)
    rng[rng == 0] = 1.0
    return np.clip((vals - mn)/rng, 0, 1)

A_r = normalize(A_val, axis_min, axis_max) * 100
B_r = normalize(B_val, axis_min, axis_max) * 100

# ------------------------------- FIGURE --------------------------------
def statsbomb_like_radar(labels, A_r, B_r, ticks, axis_min, axis_max,
                         headerA, subA, headerB, subB):
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    # for closed polygon
    theta_closed = np.concatenate([theta, theta[:1]])
    Ar = np.concatenate([A_r, A_r[:1]])
    Br = np.concatenate([B_r, B_r[:1]])

    fig = plt.figure(figsize=(13.6, 8.2), dpi=250)
    fig.patch.set_facecolor(PAGE_BG)

    # left radar axis occupying left half
    ax = plt.subplot(121, polar=True)
    ax.set_facecolor(PAGE_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=11, color=AXIS_TXT, fontweight=600)
    ax.set_yticks([])

    # grid rays
    for ang in theta:
        ax.plot([ang, ang], [0, 100], color=RAY_COLOR, lw=RAY_LW, zorder=1)

    # concentric ring bands (light -> center out)
    for r in np.linspace(20, 100, 5):
        ax.plot(np.linspace(0, 2*np.pi, 361), np.full(361, r), color=RING_COLOR, lw=RING_LW, zorder=1)

    # per-axis tick labels (like SB radial numbers along each spoke)
    # place them on each spoke at the ring radii
    ring_radii = np.linspace(20, 100, 5)   # 5 labeled rings
    for i, ang in enumerate(theta):
        vals = np.linspace(axis_min[i], axis_max[i], len(ring_radii))
        for rr, v in zip(ring_radii, vals):
            # offset so labels don't overlap the line
            rtxt = rr - 3
            txt = f"{v:.2f}" if (np.abs(v) < 100) else f"{v:.0f}"
            ax.text(ang, rtxt, txt, ha="center", va="center",
                    fontsize=8.5, color=TICK_TXT, rotation=0, zorder=2)

    # polygons (double stroke + fill)
    ax.plot(theta_closed, Ar, color="white", lw=5.0, zorder=6)
    ax.plot(theta_closed, Ar, color=EDGE_A, lw=2.2, zorder=7)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=5)

    ax.plot(theta_closed, Br, color="white", lw=5.0, zorder=6)
    ax.plot(theta_closed, Br, color=EDGE_B, lw=2.2, zorder=7)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=5)

    ax.set_rlim(0, 105)

    # Titles above radar (left/right like SB)
    fig.text(0.08, 0.96, headerA, color=TITLE_RED, fontsize=20, fontweight="bold", ha="left")
    fig.text(0.08, 0.935, subA, color=TITLE_RED, fontsize=11, ha="left")
    fig.text(0.52, 0.96, headerB, color=TITLE_BLUE, fontsize=20, fontweight="bold", ha="left")
    fig.text(0.52, 0.935, subB, color=TITLE_BLUE, fontsize=11, ha="left")

    return fig, ax

headerA = f"{pA} ({int(rowA['Age']) if pd.notna(rowA['Age']) else ''})"
subA    = f"{rowA['Team']} — {rowA['League']}"
headerB = f"{pB} ({int(rowB['Age']) if pd.notna(rowB['Age']) else ''})"
subB    = f"{rowB['Team']} — {rowB['League']}"

fig, ax = statsbomb_like_radar(labels, A_r, B_r, ticks, axis_min, axis_max,
                               headerA, subA, headerB, subB)

# --------------------------- RIGHT TABLE ------------------------------
# table columns: metric | A value | A %ile | B value | B %ile
table_df = pd.DataFrame({
    "Metric": labels,
    f"{pA} value": [f"{v:.3g}" for v in A_val],
    f"{pA} %ile":  [f"{p:.0f}"  for p in A_pct],
    f"{pB} value": [f"{v:.3g}" for v in B_val],
    f"{pB} %ile":  [f"{p:.0f}"  for p in B_pct],
})

# place table in Streamlit next to figure (SB puts a full panel; we do a tidy table)
col_plot, col_table = st.columns([1.35, 1.0])

with col_plot:
    st.pyplot(fig, use_container_width=True)

with col_table:
    st.subheader("Striker Radar & Information")
    st.caption(f"Percentiles computed vs union of {rowA['League']} & {rowB['League']} "
               f"(position startswith '{pos_scope}', minutes {min_minutes}-{max_minutes}, age {min_age}-{max_age}).")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

# ---------------------------- EXPORTS --------------------------------
buf_png = io.BytesIO()
fig.savefig(buf_png, format="png", dpi=340, bbox_inches="tight")
st.download_button("⬇️ Download PNG", data=buf_png.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_statsbomb_style.png",
                   mime="image/png")

buf_svg = io.BytesIO()
fig.savefig(buf_svg, format="svg", bbox_inches="tight")
st.download_button("⬇️ Download SVG", data=buf_svg.getvalue(),
                   file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_statsbomb_style.svg",
                   mime="image/svg+xml")



