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

# Short labels for readability in the radar
LABEL_MAP = {
    "Non-penalty goals per 90": "NP goals/90",
    "Successful dribbles, %": "Dribble success",
    "Passes to final third per 90": "Passes to 1/3",
    "Passes to penalty area per 90": "Passes to PA/90",
    "Aerial duels won, %": "Aerials won %",
    "Accurate passes, %": "Pass accuracy %",
    "Shots on target, %": "SoT %",
}

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")

    # Position scope
    pos_scope = st.text_input("Position startswith", "CF")

    # Hygiene filters
    min_minutes, max_minutes = st.slider(
        "Minutes filter", 0, int(df["Minutes played"].max()), (500, 99_999)
    )
    min_age, max_age = st.slider(
        "Age filter", int(df["Age"].min()), int(df["Age"].max()), (16, 33)
    )

    # Pick players (filtered by position only so you can still compare cross-league)
    pool_for_picker = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))]
    players = sorted(pool_for_picker["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players found for the current position filter.")
        st.stop()

    p1 = st.selectbox("Player A", players, index=0)
    p2 = st.selectbox("Player B", players, index=1)

    # Pick metrics (defaults to 13)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    suggested = [m for m in DEFAULT_METRICS if m in df.columns]
    metrics = st.multiselect(
        "Metrics (13 recommended)",
        [c for c in df.columns if c in numeric_cols],
        default=suggested,
    )
    if len(metrics) == 0:
        st.warning("Select at least one metric.")
        st.stop()

# ---- Build the comparison pool: union of the two players' leagues ----
row1 = df[df["Player"] == p1].iloc[0]
row2 = df[df["Player"] == p2].iloc[0]
leagues = {row1["League"], row2["League"]}

pool = df[
    (df["League"].isin(leagues))
    & (df["Position"].astype(str).str.startswith(tuple([pos_scope])))
    & (df["Minutes played"].between(min_minutes, max_minutes))
    & (df["Age"].between(min_age, max_age))
].copy()

# Validate metrics present and numeric
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

# ---- Percentiles across this pool (0..100) ----
ranks = pool[metrics].rank(pct=True) * 100
pool_pct = pd.concat([pool[["Player"]], ranks], axis=1)

def get_player_percentiles(name: str):
    sub = pool_pct[pool_pct["Player"] == name][metrics]
    if sub.empty:
        return np.array([np.nan] * len(metrics))
    return sub.mean().values

p1_pct = get_player_percentiles(p1)
p2_pct = get_player_percentiles(p2)

# Raw values (averaged if dup rows exist for a player)
raw = df[df["Player"].isin([p1, p2])][["Player"] + metrics].copy()
raw_grouped = raw.groupby("Player")[metrics].mean(numeric_only=True).reset_index()
p1_vals = raw_grouped[raw_grouped["Player"] == p1][metrics].values[0]
p2_vals = raw_grouped[raw_grouped["Player"] == p2][metrics].values[0]

# ---- Radar chart (percentiles) ----
st.subheader("Percentile radar (vs combined leagues)")

theta = [LABEL_MAP.get(m, m) for m in metrics]
COL_A = "#E74C3C"  # red
COL_B = "#1F77B4"  # blue

fig = go.Figure()
fig.add_trace(
    go.Scatterpolar(
        r=p1_pct,
        theta=theta,
        name=p1,
        mode="lines+markers",
        line=dict(color=COL_A, width=3),
        marker=dict(size=6),
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.15)",
        hovertemplate="<b>%{theta}</b><br>%{r:.1f}th percentile<extra>"
        + p1
        + "</extra>",
    )
)
fig.add_trace(
    go.Scatterpolar(
        r=p2_pct,
        theta=theta,
        name=p2,
        mode="lines+markers",
        line=dict(color=COL_B, width=3),
        marker=dict(size=6),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.15)",
        hovertemplate="<b>%{theta}</b><br>%{r:.1f}th percentile<extra>"
        + p2
        + "</extra>",
    )
)
fig.update_layout(
    template="plotly_white",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    margin=dict(l=10, r=10, t=10, b=10),
    height=600,
    polar=dict(
        radialaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            tickfont=dict(size=12),
            gridcolor="rgba(0,0,0,0.08)",
            gridwidth=1,
            linecolor="rgba(0,0,0,0.25)",
        ),
        angularaxis=dict(
            tickfont=dict(size=12),
            gridcolor="rgba(0,0,0,0.06)",
        ),
    ),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Details table: raw values + percentiles ----
def fmt_num(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return v

table = pd.DataFrame(
    {
        "Metric": metrics,
        f"{p1} value": [fmt_num(v) for v in p1_vals],
        f"{p1} %ile": [f"{x:.1f}%" for x in p1_pct],
        f"{p2} value": [fmt_num(v) for v in p2_vals],
        f"{p2} %ile": [f"{x:.1f}%" for x in p2_pct],
    }
)

st.subheader("Details")
st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download comparison (CSV)",
    data=csv,
    file_name="player_comparison.csv",
    mime="text/csv",
)

# Friendly note if same player is selected twice
if p1 == p2:
    st.info("You selected the same player for A and B — the radar will overlap.")
