# app.py — Player Comparison (Director Radar — Elite)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from pathlib import Path
import io

st.set_page_config(page_title="Player Comparison — Director Radar", layout="wide")

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

required = {"Player","League","Team","Position","Minutes played","Age"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ---------- Defaults (13-ish metrics) ----------
# ✔ removed "Key passes per 90" from defaults by request
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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    pos_scope = st.text_input("Position startswith", "CF")

    # Numerics
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Premier-League friendly minutes control
    min_minutes, max_minutes = st.slider("Minutes filter", 0, 5000, (500, 5000))
    min_age, max_age = st.slider("Age filter", int(df["Age"].min() or 14), int(df["Age"].max() or 40), (16, 33))

    # Player pickers (by position scope)
    pool_pick = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))].copy()
    players = sorted(pool_pick["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players for this filter.")
        st.stop()
    pA = st.selectbox("Player A (red)", players, index=0)
    pB = st.selectbox("Player B (blue)", players, index=1)

    # Metric selection
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics_default = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics (recommended)", [c for c in df.columns if c in numeric_cols], metrics_default)
    if len(metrics) < 3:
        st.warning("Pick at least 3 metrics.")
        st.stop()

    sort_by_gap = st.checkbox("Sort axes by biggest gap", False)
    show_percent_numbers = st.checkbox("Show numbers on polygons", False)
    show_rays = st.checkbox("Show metric rays", False)  # default off for a cleaner elite look

# ---------- Build percentile pool: union of both players’ leagues ----------
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

# Num conversion + NA drop for chosen metrics
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

# Percentiles (0..100) within this pool
ranks = pool[metrics].rank(pct=True) * 100
pool_pct = pd.concat([pool[["Player"]], ranks], axis=1)

def pct_for(name: str):
    sub = pool_pct[pool_pct["Player"] == name][metrics]
    if sub.empty:
        return np.full(len(metrics), np.nan)
    return sub.mean().values

A_pct = pct_for(pA)
B_pct = pct_for(pB)

teamA, leagueA = rowA["Team"], rowA["League"]
teamB, leagueB = rowB["Team"], rowB["League"]

labels = [SHORT.get(m, m) for m in metrics]
if sort_by_gap:
    gap = np.abs(A_pct - B_pct)
    order = np.argsort(-gap)
    labels = [labels[i] for i in order]
    A_pct = A_pct[order]
    B_pct = B_pct[order]

# ---------- Radar Drawer (elite, from scratch) ----------
def draw_director_radar(
    labels, A, B,
    title_left, subtitle_left, title_right, subtitle_right,
    colA="#E64B3C", colB="#1F77B4",
    page_bg="#F6F7F9",   # page
    disc_bg="#D1D7E0",   # chart disc
    band_0_25="#FFFFFF",
    band_25_50="#EFF2F6",
    band_50_75="#E3E7ED",
    band_75_100="#D6DBE4",
    ring_line="#4B5563", # bold quartile lines
    ray_line="#CBD5E1",  # optional metric rays
    label_color="#0F172A",
    show_vals=False,
    show_rays=False
):
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    theta += theta[:1]

    A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0).tolist(); A += A[:1]
    B = np.nan_to_num(np.asarray(B, dtype=float), nan=0.0).tolist(); B += B[:1]

    fig = plt.figure(figsize=(12.5, 12), dpi=260)
    fig.patch.set_facecolor(page_bg)
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(page_bg)

    # Start at 12 o’clock, clockwise
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Axis labels only (no radial ticks)
    ax.set_xticks(np.linspace(0, 2*np.pi, N, endpoint=False))
    ax.set_xticklabels(labels, fontsize=12, fontweight=700, color=label_color)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Disc background
    disc = Circle((0,0), radius=102, transform=ax.transData._b, color=disc_bg, zorder=0)
    ax.add_artist(disc)

    # Quartile bands (no labels)
    bands = [(25, band_0_25),(50, band_25_50),(75, band_50_75),(100, band_75_100)]
    inner = 0
    for r, color in bands:
        ax.add_artist(Wedge((0,0), r, 0, 360, width=r-inner, facecolor=color, edgecolor="none", zorder=1))
        inner = r

    # Bold separations at 25/50/75/100 (no text)
    for r in (25, 50, 75, 100):
        ax.plot([0, 2*np.pi], [r, r], color=ring_line, lw=2.3, alpha=1.0, zorder=2)

    # Optional very light rays per metric
    if show_rays:
        for t in np.linspace(0, 2*np.pi, N, endpoint=False):
            ax.plot([t, t], [0, 100], color=ray_line, lw=1.0, alpha=0.6, zorder=2)

    # Polygons with double-stroke outline for razor crispness
    ax.plot(theta, A, color="white", lw=5.4, zorder=4)
    ax.plot(theta, A, color=colA,   lw=2.9, zorder=5)
    ax.fill(theta, A, color=colA, alpha=0.26, zorder=3)

    ax.plot(theta, B, color="white", lw=5.4, zorder=4)
    ax.plot(theta, B, color=colB,   lw=2.9, zorder=5)
    ax.fill(theta, B, color=colB, alpha=0.26, zorder=3)

    # Optional numbers
    if show_vals:
        for a, v in zip(theta[:-1], A[:-1]):
            if v >= 8:
                ax.text(a, min(v+3, 100), f"{v:.0f}", color=colA, fontsize=10,
                        ha="center", va="center", fontweight="bold", zorder=6)
        for a, v in zip(theta[:-1], B[:-1]):
            if v >= 8:
                ax.text(a, max(v-7, 0)+3, f"{v:.0f}", color=colB, fontsize=10,
                        ha="center", va="center", fontweight="bold", zorder=6)

    ax.set_rlim(0, 102)

    # Titles only (no on-chart “percentiles vs …” subtitle)
    fig.text(0.18, 0.965, title_left,  color=colA, fontsize=28, fontweight="bold", ha="left")
    fig.text(0.18, 0.937, f"{subtitle_left}",  color=colA, fontsize=13, ha="left")
    fig.text(0.82, 0.965, title_right, color=colB, fontsize=28, fontweight="bold", ha="right")
    fig.text(0.82, 0.937, f"{subtitle_right}", color=colB, fontsize=13, ha="right")

    return fig

labels_display = [SHORT.get(m, m) for m in metrics]
fig = draw_director_radar(
    labels=labels_display,
    A=A_pct, B=B_pct,
    title_left=pA, subtitle_left=f"{rowA['Team']} — {rowA['League']}",
    title_right=pB, subtitle_right=f"{rowB['Team']} — {rowB['League']}",
    show_vals=show_percent_numbers,
    show_rays=show_rays
)

st.pyplot(fig, use_container_width=True)

# ---------- Exports ----------
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



