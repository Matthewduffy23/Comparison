import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Player Comparison", layout="wide")
st.title("🆚 Player Comparison")

# --- Load data (CSV in repo root) ---
@st.cache_data
def load_df():
    return pd.read_csv("WORLDJUNE25.csv")
df = load_df()

# --- Defaults (13 metrics) ---
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

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")

    # (Optional) restrict by position tag
    pos_scope = st.text_input("Position startswith", "CF")

    # Simple hygiene filters so pool is sensible
    min_minutes, max_minutes = st.slider("Minutes filter", 0, int(df["Minutes played"].max()), (500, 99999))
    min_age, max_age       = st.slider("Age filter", int(df["Age"].min()), int(df["Age"].max()), (16, 33))

    # Choose 2 players
    pool_for_picker = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))]
    players = sorted(pool_for_picker["Player"].dropna().unique().tolist())
    p1 = st.selectbox("Player A", players, index=0)
    p2 = st.selectbox("Player B", players, index=1)

    # Choose metrics (default 13)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    suggested = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect("Metrics (13 recommended)", [c for c in df.columns if c in numeric_cols], default=suggested)
    if len(metrics) == 0:
        st.stop()

# ---- Build the comparison pool: both players' leagues combined ----
row1 = df[df["Player"] == p1].iloc[0]
row2 = df[df["Player"] == p2].iloc[0]
leagues = {row1["League"], row2["League"]}

pool = df[
    (df["League"].isin(leagues)) &
    (df["Position"].astype(str).str.startswith(tuple([pos_scope]))) &
    (df["Minutes played"].between(min_minutes, max_minutes)) &
    (df["Age"].between(min_age, max_age))
].copy()

# Make sure all metrics are present and numeric
missing = [m for m in metrics if m not in pool.columns]
if missing:
    st.error(f"Missing metric columns in data: {missing}")
    st.stop()

for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")

pool = pool.dropna(subset=metrics)

if pool.empty:
    st.warning("No players in the pool after filters. Loosen Minutes/Age/Position.")
    st.stop()

# ---- Compute percentiles across the pool (between the two leagues) ----
ranks = pool[metrics].rank(pct=True) * 100  # 0..100 percentiles
pool = pd.concat([pool[["Player"]], ranks], axis=1)

def get_player_percentiles(name: str):
    # if multiple rows per player, take mean
    sub = pool[pool["Player"] == name][metrics]
    if sub.empty:
        return np.array([np.nan] * len(metrics))
    return sub.mean().values

p1_pct = get_player_percentiles(p1)
p2_pct = get_player_percentiles(p2)

# Also fetch raw values (averaged if duplicates)
raw = df[df["Player"].isin([p1, p2])][["Player"] + metrics].copy()
raw_grouped = raw.groupby("Player")[metrics].mean(numeric_only=True).reset_index()
p1_vals = raw_grouped[raw_grouped["Player"] == p1][metrics].values[0]
p2_vals = raw_grouped[raw_grouped["Player"] == p2][metrics].values[0]

# ---- Radar chart (percentiles) ----
st.subheader("Percentile radar (vs combined leagues)")
theta = metrics
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=p1_pct, theta=theta, fill="toself", name=p1))
fig.add_trace(go.Scatterpolar(r=p2_pct, theta=theta, fill="toself", name=p2))
fig.update_layout(
    polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")),
    showlegend=True,
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ---- Table: raw values + percentiles ----
def fmt(v):
    try:
        return f"{v:.2f}"
    except Exception:
        return v

table = pd.DataFrame({
    "Metric": metrics,
    f"{p1} value": [fmt(v) for v in p1_vals],
    f"{p1} %ile":  [f"{x:.1f}" for x in p1_pct],
    f"{p2} value": [fmt(v) for v in p2_vals],
    f"{p2} %ile":  [f"{x:.1f}" for x in p2_pct],
})

st.subheader("Details")
st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download comparison (CSV)", data=csv, file_name="player_comparison.csv", mime="text/csv")
