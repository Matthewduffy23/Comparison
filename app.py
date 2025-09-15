# app.py — Player Comparison (Disc Radar Pro)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from pathlib import Path
import io

st.set_page_config(page_title="Player Comparison — Disc Radar Pro", layout="wide")

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
        st.error("Not enough players with current filter.")
        st.stop()

    pA = st.selectbox("Player A (red)", players, index=0)
    pB = st.selectbox("Player B (blue)", players, index=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics_default = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics (13 recommended)", [c for c in df.columns if c in numeric_cols], metrics_default)
    if len(metrics) < 3:
        st.warning("Pick at least 3 metrics.")
        st.stop()

    sort_by_gap = st.checkbox("Sort axes by biggest gap", False)
    show_percent_numbers = st.checkbox("Show percentile numbers on polygons", False)

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

# ---------- Disc Radar Drawer ----------
def draw_disc_radar(
    labels, A, B,
    title_left, subtitle_left, title_right, subtitle_right,
    colA="#E64B3C", colB="#1F77B4",
    page_bg="#F5F6F8",
    disc_bg="#D6DAE1",          # chart disc color (distinct from page)
    band_0_25="#FFFFFF",
    band_25_50="#EEF1F5",
    band_50_75="#E1E5EB",
    band_75_100="#D3D8E0",
    ring_line="#9BA4B0",        # strong 25/50/75/100 separators
    ray_line="#CCD2DB",         # metric rays
    label_color="#2B2B2B",
    show_vals=False
):
    N = len(labels)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    ang += ang[:1]

    A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0).tolist(); A += A[:1]
    B = np.nan_to_num(np.asarray(B, dtype=float), nan=0.0).tolist(); B += B[:1]

    fig = plt.figure(figsize=(12.5, 12.5), dpi=220)
    fig.patch.set_facecolor(page_bg)
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(page_bg)

    # start at top, clockwise
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Remove default stuff
    ax.set_xticks(np.linspace(0, 2*np.pi, N, endpoint=False))
    ax.set_xticklabels(labels, fontsize=12, fontweight=600, color=label_color)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # --- Chart disc (clear contrast with page) ---
    disc = Circle((0,0), radius=102, transform=ax.transData._b, color=disc_bg, zorder=0)
    ax.add_artist(disc)

    # --- Quartile bands (0–25, 25–50, 50–75, 75–100) ---
    bands = [
        (25, band_0_25),
        (50, band_25_50),
        (75, band_50_75),
        (100, band_75_100),
    ]
    inner = 0
    for r, color in bands:
        w = Wedge((0,0), r, 0, 360, width=r-inner, facecolor=color, edgecolor="none", zorder=1)
        ax.add_artist(w)
        inner = r

    # Strong separators at 25/50/75/100
    for r in (25, 50, 75, 100):
        ax.plot([0, 2*np.pi], [r, r], color=ring_line, lw=1.6, alpha=0.95, zorder=2)

    # Subtle metric rays
    for t in np.linspace(0, 2*np.pi, N, endpoint=False):
        ax.plot([t, t], [0, 100], color=ray_line, lw=1.0, alpha=0.7, zorder=2)

    # Polygons
    ax.plot(ang, A, color=colA, lw=3.2, zorder=4)
    ax.fill(ang, A, color=colA, alpha=0.28, zorder=3)
    ax.plot(ang, B, color=colB, lw=3.2, zorder=4)
    ax.fill(ang, B, color=colB, alpha=0.28, zorder=3)

    # Optional percentile numbers
    if show_vals:
        for a, v in zip(ang[:-1], A[:-1]):
            if v >= 8:
                ax.text(a, min(v+3, 100), f"{v:.0f}", color=colA, fontsize=10,
                        ha="center", va="center", fontweight="bold", zorder=5)
        for a, v in zip(ang[:-1], B[:-1]):
            if v >= 8:
                ax.text(a, max(v-7, 0)+3, f"{v:.0f}", color=colB, fontsize=10,
                        ha="center", va="center", fontweight="bold", zorder=5)

    ax.set_rlim(0, 102)

    # Four ring labels (25/50/75/100) on left side
    label_angle = np.deg2rad(180)
    for r, txt in zip((25, 50, 75, 100), ("25", "50", "75", "100")):
        ax.text(label_angle, r, txt, color="#606A76", fontsize=12,
                ha="center", va="center", zorder=6)

    # Title blocks
    fig.text(0.18, 0.96, title_left, color=colA, fontsize=28, fontweight="bold", ha="left")
    fig.text(0.18, 0.93, subtitle_left, color=colA, fontsize=13, ha="left")
    fig.text(0.82, 0.96, title_right, color=colB, fontsize=28, fontweight="bold", ha="right")
    fig.text(0.82, 0.93, subtitle_right, color=colB, fontsize=13, ha="right")

    return fig

fig = draw_disc_radar(
    labels=labels,
    A=A_pct, B=B_pct,
    title_left=pA, subtitle_left=f"{teamA} — {leagueA}",
    title_right=pB, subtitle_right=f"{teamB} — {leagueB}",
    show_vals=show_percent_numbers
)

st.pyplot(fig, use_container_width=True)

# PNG download
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    "⬇️ Download radar (PNG)",
    data=buf.getvalue(),
    file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_disc_radar.png",
    mime="image/png",
)
