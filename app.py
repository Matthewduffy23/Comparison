# app.py — Player Comparison (Coxcomb Pro)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
from pathlib import Path
import io

# ---------------- Page setup ----------------
st.set_page_config(page_title="Player Comparison — Pro Polar", layout="wide")
st.title("🧭 Player Comparison — Pro Polar")

# ---------------- Data loading ----------------
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
        st.warning("Upload the dataset to continue.")
        st.stop()
    df = pd.read_csv(up)

# Basic validation
base_cols = {"Player","League","Position","Minutes played","Age"}
missing = [c for c in base_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------------- Default metrics (13) ----------------
DEFAULT_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90","Shots on target, %",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90","Key passes per 90",
]

SHORT_LABEL = {
    "Non-penalty goals per 90":"NP goals/90",
    "Shots on target, %":"SoT %",
    "Successful dribbles, %":"Dribble success",
    "Accurate passes, %":"Pass accuracy %",
    "Passes to final third per 90":"Passes to 1/3",
    "Passes to penalty area per 90":"Passes to PA/90",
    "Aerial duels won, %":"Aerials won %",
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")
    pos_scope = st.text_input("Position startswith", "CF")
    # Safe ranges
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played filter", 0, int(df["Minutes played"].max() or 99999), (500, 99999))
    min_age, max_age = st.slider("Age filter", int(df["Age"].min() or 14), int(df["Age"].max() or 40), (16, 33))

    # Player pickers
    picker_pool = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))].copy()
    players = sorted(picker_pool["Player"].dropna().unique())
    if len(players) < 2:
        st.error("Not enough players for this position filter.")
        st.stop()
    p1 = st.selectbox("Player A (red)", players, index=0)
    p2 = st.selectbox("Player B (blue)", players, index=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    default_metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics (13 recommended)", [c for c in df.columns if c in numeric_cols], default_metrics)
    if not metrics:
        st.stop()

    show_values = st.checkbox("Show value labels on wedges", True)
    save_png = st.checkbox("Prepare PNG download", True)

# ---------------- Build percentile pool (union of the two players' leagues) ----------------
row1 = df[df["Player"] == p1].iloc[0]
row2 = df[df["Player"] == p2].iloc[0]
leagues_union = {row1["League"], row2["League"]}

pool = df[
    (df["League"].isin(leagues_union)) &
    (df["Position"].astype(str).str.startswith(tuple([pos_scope]))) &
    (df["Minutes played"].between(min_minutes, max_minutes)) &
    (df["Age"].between(min_age, max_age))
].copy()

# Ensure metrics numeric & available
miss_m = [m for m in metrics if m not in pool.columns]
if miss_m:
    st.error(f"Missing metric columns: {miss_m}")
    st.stop()

for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")
pool = pool.dropna(subset=metrics)
if pool.empty:
    st.warning("No players in pool after filters. Loosen filters.")
    st.stop()

# Percentiles (0..100) within this pool
ranks = pool[metrics].rank(pct=True) * 100
pool_pct = pd.concat([pool[["Player"]], ranks], axis=1)

def pct_for(name: str):
    sub = pool_pct[pool_pct["Player"] == name][metrics]
    return (sub.mean().values if not sub.empty else np.full(len(metrics), np.nan))

p1_pct = pct_for(p1)
p2_pct = pct_for(p2)

# Order metrics by absolute gap
gap = np.abs(p1_pct - p2_pct)
order = np.argsort(-gap)
metrics_ord = [metrics[i] for i in order]
labels_ord  = [SHORT_LABEL.get(m, m) for m in metrics_ord]
p1_ord = p1_pct[order]
p2_ord = p2_pct[order]

# ---------------- POLAR COXCOMB COMPARISON ----------------
def draw_coxcomb_compare(labels, A, B, nameA, nameB,
                         bg_color="#f7f7f8",
                         ring_colors=("white","#f1f1f1","#e8e8e8","#dfdfdf"),
                         colA="#E74C3C", colB="#1F77B4",
                         show_values=True):
    """
    Draws a polar coxcomb with quartile rings and two concentric wedges per metric (A outer, B inner).
    A, B are 0..100 percentiles.
    """
    n = len(labels)
    # angles
    theta = np.linspace(0, 2*np.pi, n+1)
    width = (2*np.pi) / n * 0.9   # wedge width (leave small gap)
    offsets = theta[:-1] + (2*np.pi/n - width)/2

    # Figure
    fig = plt.figure(figsize=(11, 9), dpi=200)
    ax = plt.subplot(111, polar=True)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Hide standard polar junk
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # Quartile ring bands (0–25, 25–50, 50–75, 75–100)
    # Draw as full 360° annuli
    radii = [25, 50, 75, 100]
    inner = 0
    for i, r in enumerate(radii):
        band = Wedge(center=(0,0), r=r, theta1=0, theta2=360, width=r-inner, facecolor=ring_colors[i], edgecolor="none", zorder=0)
        ax.add_patch(band)
        inner = r

    # Light ring outlines at 25/50/75/100
    for r in (25, 50, 75, 100):
        ax.add_line(Line2D([0, 2*np.pi], [r, r], color="silver", linewidth=0.6, alpha=0.7))

    # Metric divider lines
    for ang in theta:
        ax.add_line(Line2D([ang, ang], [0, 100], color="lightgrey", linewidth=0.6, alpha=0.8, zorder=1))

    # Wedges: A outer, B inner (slightly inset) so both visible
    inset = 6  # how much to inset B so it’s visibly concentric
    for i, ang in enumerate(offsets):
        # Clamp values to [0,100]
        va = float(np.nan_to_num(A[i], nan=0.0))
        vb = float(np.nan_to_num(B[i], nan=0.0))

        # A wedge (outer)
        ax.bar(x=ang, height=va, width=width, bottom=0,
               color=colA, edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)

        # B wedge (inner, inset)
        height_b = max(vb - inset, 0)
        ax.bar(x=ang, height=height_b, width=width*0.72, bottom=inset,  # slightly narrower
               color=colB, edgecolor="white", linewidth=0.8, alpha=0.95, zorder=4)

        if show_values:
            # Value labels near wedge tip; nudge for readability
            if va > 10:
                ax.text(ang, min(va, 98), f"{va:.0f}", color="white", ha="center", va="center",
                        fontsize=9, fontweight="bold", zorder=5)
            if vb > 10:
                ax.text(ang, min(vb, 92)-inset*0.6, f"{vb:.0f}", color="white", ha="center", va="center",
                        fontsize=8, fontweight="bold", zorder=6)

    # Metric labels around the rim
    for i, ang in enumerate(offsets):
        ax.text(ang, 108, labels[i], ha="center", va="center", fontsize=11, fontweight=600, color="#333")

    # Custom legend
    legend_elems = [
        Line2D([0],[0], color=colA, lw=8, label=nameA),
        Line2D([0],[0], color=colB, lw=8, label=nameB),
    ]
    leg = ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                    ncol=2, frameon=False, fontsize=12)
    for txt in leg.get_texts():
        txt.set_color("#222")

    ax.set_rlim(0, 110)  # little headroom for labels
    return fig

# Draw it
fig = draw_coxcomb_compare(labels_ord, p1_ord, p2_ord, p1, p2,
                           show_values=show_values)

st.pyplot(fig, use_container_width=True)

# PNG download
if save_png:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    st.download_button("⬇️ Download chart (PNG)", data=buf.getvalue(),
                       file_name=f"{p1}_vs_{p2}_polar.png", mime="image/png")
