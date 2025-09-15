# app.py — Player Comparison (pro edition)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(page_title="Player Comparison", layout="wide")
st.title("🆚 Player Comparison")

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_df_from_repo(csv_name: str = "WORLDJUNE25.csv"):
    p = Path(__file__).with_name(csv_name)
    if p.exists():
        return pd.read_csv(p)
    return None

df = load_df_from_repo()
if df is None:
    uploaded = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not uploaded:
        st.warning("Add WORLDJUNE25.csv to the repo root or upload it here to continue.")
        st.stop()
    df = pd.read_csv(uploaded)

# Basic validation
required_base = {"Player", "League", "Position", "Minutes played", "Age"}
missing_base = [c for c in required_base if c not in df.columns]
if missing_base:
    st.error(f"Dataset is missing required columns: {missing_base}")
    st.stop()

# ---------- Defaults (13 headline metrics) ----------
DEFAULT_METRICS = [
    "Non-penalty goals per 90",
    "xG per 90",
    "Shots per 90",
    "Shots on target, %",
    "Dribbles per 90",
    "Successful dribbles, %",
    "Touches in box per 90",
    "Aerial duels per 90",
    "Aerial duels won, %",
    "Passes per 90",
    "Accurate passes, %",
    "xA per 90",
    "Key passes per 90",
]

# Short labels for charts
LABEL_MAP = {
    "Non-penalty goals per 90": "NP goals/90",
    "Successful dribbles, %": "Dribble success",
    "Passes to final third per 90": "Passes to 1/3",
    "Passes to penalty area per 90": "Passes to PA/90",
    "Aerial duels won, %": "Aerials won %",
    "Accurate passes, %": "Pass accuracy %",
    "Shots on target, %": "SoT %",
}

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")

    # Position scope for pool + pickers
    pos_scope = st.text_input("Position startswith", "CF")

    # Hygiene filters for the percentile pool
    minutes_max = int(pd.to_numeric(df["Minutes played"], errors="coerce").fillna(0).max())
    age_min = int(pd.to_numeric(df["Age"], errors="coerce").fillna(0).min())
    age_max = int(pd.to_numeric(df["Age"], errors="coerce").fillna(0).max())

    min_minutes, max_minutes = st.slider("Minutes filter", 0, max(99_999, minutes_max), (500, 99_999))
    min_age, max_age = st.slider("Age filter", max(14, age_min), max(33, age_max), (16, min(33, age_max)))

    # Player pickers (filtered by position so choices stay relevant)
    pool_for_picker = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))]
    players = sorted(pool_for_picker["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players for the current position filter.")
        st.stop()
    p1 = st.selectbox("Player A", players, index=0)
    p2 = st.selectbox("Player B", players, index=1)

    # Metric selection (defaults to 13)
    numeric_cols = set(df.select_dtypes(include=["number"]).columns.tolist())
    default_existing = [m for m in DEFAULT_METRICS if m in df.columns]
    selectable = sorted(list(numeric_cols | set(default_existing)))  # ensure defaults are present if numeric
    metrics = st.multiselect("Metrics (13 recommended)", selectable, default=default_existing)
    if len(metrics) == 0:
        st.warning("Select at least one metric.")
        st.stop()

# ---------- Build comparison pool: union of both players' leagues ----------
try:
    row1 = df[df["Player"] == p1].iloc[0]
    row2 = df[df["Player"] == p2].iloc[0]
except IndexError:
    st.error("Selected player not found in dataset. Adjust filters or reload.")
    st.stop()

leagues_union = {row1["League"], row2["League"]}

# Coerce key numeric fields for filtering
df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

pool = df[
    (df["League"].isin(leagues_union))
    & (df["Position"].astype(str).str.startswith(tuple([pos_scope])))
    & (df["Minutes played"].between(min_minutes, max_minutes))
    & (df["Age"].between(min_age, max_age))
].copy()

# Validate metric columns and convert to numeric
missing_metrics = [m for m in metrics if m not in pool.columns]
if missing_metrics:
    st.error(f"Missing metric columns in data: {missing_metrics}")
    st.stop()

for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")

pool = pool.dropna(subset=metrics)
if pool.empty:
    st.warning("No players in the percentile pool after filters. Loosen Minutes/Age/Position.")
    st.stop()

# ---------- Percentiles across this pool (0..100) ----------
ranks = pool[metrics].rank(pct=True) * 100
pool_pct = pd.concat([pool[["Player"]], ranks], axis=1)

def percentiles_for(name: str) -> np.ndarray:
    sub = pool_pct[pool_pct["Player"] == name][metrics]
    if sub.empty:
        return np.array([np.nan] * len(metrics))
    return sub.mean().values

p1_pct = percentiles_for(p1)
p2_pct = percentiles_for(p2)

# Raw values, averaged if a player has multiple rows
raw = df[df["Player"].isin([p1, p2])][["Player"] + metrics].copy()
raw_grouped = raw.groupby("Player")[metrics].mean(numeric_only=True).reset_index()
p1_vals = raw_grouped.loc[raw_grouped["Player"] == p1, metrics].values[0]
p2_vals = raw_grouped.loc[raw_grouped["Player"] == p2, metrics].values[0]

# ---------- Order metrics by biggest percentile gap ----------
gap = np.abs(p1_pct - p2_pct)
order = np.argsort(-gap)  # descending by difference
metrics_ord   = [metrics[i] for i in order]
labels_ord    = [LABEL_MAP.get(m, m) for m in metrics_ord]
p1_pct_ord    = p1_pct[order]
p2_pct_ord    = p2_pct[order]
p1_vals_ord   = [p1_vals[i] for i in order]
p2_vals_ord   = [p2_vals[i] for i in order]

# ---------- Professional radar (percentiles) ----------
COL_A = "#E74C3C"  # red
COL_B = "#1F77B4"  # blue

st.subheader("Percentile radar (vs combined leagues)")
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=p1_pct_ord, theta=labels_ord, name=p1,
    mode="lines+markers", line=dict(color=COL_A, width=3),
    marker=dict(size=6), fill="toself", fillcolor="rgba(231,76,60,0.12)",
    hovertemplate="<b>%{theta}</b><br>%{r:.1f}th percentile<extra>"+p1+"</extra>",
))
fig.add_trace(go.Scatterpolar(
    r=p2_pct_ord, theta=labels_ord, name=p2,
    mode="lines+markers", line=dict(color=COL_B, width=3),
    marker=dict(size=6), fill="toself", fillcolor="rgba(31,119,180,0.12)",
    hovertemplate="<b>%{theta}</b><br>%{r:.1f}th percentile<extra>"+p2+"</extra>",
))
fig.update_layout(
    template="plotly_white", height=620, margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    polar=dict(
        radialaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(0,0,0,.08)", linecolor="rgba(0,0,0,.25)"),
        angularaxis=dict(gridcolor="rgba(0,0,0,.06)", tickfont=dict(size=12)),
    ),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Side-by-side percentile bars (clear read) ----------
st.subheader("Percentiles by metric (sorted by biggest gap)")
bars_df = pd.DataFrame({
    "Metric": labels_ord,
    p1: p1_pct_ord,
    p2: p2_pct_ord,
}).melt(id_vars="Metric", var_name="Player", value_name="Percentile")

bar_fig = px.bar(
    bars_df, y="Metric", x="Percentile", color="Player",
    barmode="group", range_x=[0, 100], orientation="h",
    color_discrete_map={p1: COL_A, p2: COL_B},
)
bar_fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10), legend_title=None)
bar_fig.update_yaxes(autorange="reversed")
st.plotly_chart(bar_fig, use_container_width=True)

# ---------- Details & differences table ----------
def fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return v

delta_table = pd.DataFrame({
    "Metric": metrics_ord,
    f"{p1} value": [fmt(v) for v in p1_vals_ord],
    f"{p1} %ile":  [f"{x:.1f}%" for x in p1_pct_ord],
    f"{p2} value": [fmt(v) for v in p2_vals_ord],
    f"{p2} %ile":  [f"{x:.1f}%" for x in p2_pct_ord],
    "Δ %ile (A - B)": [f"{(a - b):+,.1f}" for a, b in zip(p1_pct_ord, p2_pct_ord)],
})
st.subheader("Details & differences")
st.dataframe(delta_table, use_container_width=True)

st.download_button(
    "⬇️ Download comparison (CSV)",
    data=delta_table.to_csv(index=False).encode("utf-8"),
    file_name="player_comparison.csv",
    mime="text/csv",
)

# Friendly note if same player is selected twice
if p1 == p2:
    st.info("You selected the same player for A and B — the radar will overlap.")
