"""
Hyperliquid × Fear/Greed  —  Interactive Streamlit Dashboard
============================================================
Sections:
  1. Overview Dashboard   — key metrics + sentiment summary
  2. Behavioral Explorer  — filter by sentiment, compare metrics
  3. Trader Clustering    — KMeans archetypes with scatter
  4. Predictive Model     — predict next-day PnL direction
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyperliquid × Sentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme constants ────────────────────────────────────────────────────
FEAR_COLOR    = "#E74C3C"
GREED_COLOR   = "#2ECC71"
NEUTRAL_COLOR = "#F4A623"
DARK_BG       = "#080B14"
PANEL_BG      = "#0D1117"
ACCENT        = "#F4A623"
GRID_COL      = "#1E2D4A"
BUCKET_COLORS = {"Fear": FEAR_COLOR, "Neutral": NEUTRAL_COLOR, "Greed": GREED_COLOR}
ORDER         = ["Fear", "Neutral", "Greed"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COL,
    "axes.labelcolor":  "#C8D6E5",
    "xtick.color":      "#5A7A9A",
    "ytick.color":      "#5A7A9A",
    "text.color":       "#C8D6E5",
    "grid.color":       GRID_COL,
    "grid.linestyle":   "--",
    "grid.alpha":       0.35,
    "font.family":      "DejaVu Sans",
    "legend.facecolor": "#0D1117",
    "legend.edgecolor": GRID_COL,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Reset & Base ── */
    html, body, .stApp {
        background-color: #080B14 !important;
        color: #C8D6E5 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hide default Streamlit header/hamburger */
    #MainMenu, header[data-testid="stHeader"], footer { display: none !important; }

    /* Main content padding — push content down so tabs are visible */
    .block-container {
        padding-top: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #0A0F1E 100%) !important;
        border-right: 1px solid #1E2D4A !important;
        padding-top: 2rem !important;
    }
    [data-testid="stSidebar"] * { color: #C8D6E5 !important; }
    [data-testid="stSidebar"] hr { border-color: #1E2D4A !important; }

    /* Sidebar brand */
    [data-testid="stSidebar"] h2 {
        font-family: 'Space Mono', monospace !important;
        color: #F4A623 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.05em;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0D1117 !important;
        border-bottom: 1px solid #1E2D4A !important;
        border-radius: 0 !important;
        gap: 0 !important;
        padding: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5A7A9A !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        padding: 0.75rem 1.5rem !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        color: #F4A623 !important;
        border-bottom: 2px solid #F4A623 !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #C8D6E5 !important;
        background: #0F1624 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem !important;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0D1117 0%, #111827 100%) !important;
        border: 1px solid #1E2D4A !important;
        border-top: 2px solid #F4A623 !important;
        border-radius: 8px !important;
        padding: 1.2rem 1.4rem !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: #F4A623 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: #F4A623 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #5A7A9A !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* ── Headers ── */
    h1 {
        font-family: 'Space Mono', monospace !important;
        color: #F4A623 !important;
        font-size: 1.8rem !important;
        letter-spacing: -0.01em;
        border-bottom: 1px solid #1E2D4A;
        padding-bottom: 0.5rem;
    }
    h2 {
        font-family: 'Space Mono', monospace !important;
        color: #E8EDF2 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.02em;
    }
    h3 {
        color: #F4A623 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    h4, h5 { color: #C8D6E5 !important; }

    /* ── Dividers ── */
    hr { border-color: #1E2D4A !important; margin: 1.5rem 0 !important; }

    /* ── Dataframe ── */
    .stDataFrame {
        border: 1px solid #1E2D4A !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    .stDataFrame thead th {
        background: #0D1117 !important;
        color: #F4A623 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    .stDataFrame tbody tr:nth-child(even) { background: #0A0F1E !important; }
    .stDataFrame tbody td { color: #C8D6E5 !important; font-size: 0.85rem !important; }

    /* ── Selectbox / Multiselect ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #0D1117 !important;
        border: 1px solid #1E2D4A !important;
        border-radius: 6px !important;
        color: #C8D6E5 !important;
    }
    .stMultiSelect span[data-baseweb="tag"] {
        background: #1E2D4A !important;
        color: #F4A623 !important;
        border-radius: 4px !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #F4A623 !important;
        border-color: #F4A623 !important;
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] { color: #5A7A9A !important; }

    /* ── Alert / info boxes ── */
    .stAlert {
        background: #0D1117 !important;
        border: 1px solid #1E2D4A !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSuccessMessage"] {
        background: linear-gradient(135deg, #0D1A0D, #0A1A10) !important;
        border: 1px solid #2ECC71 !important;
        border-left: 4px solid #2ECC71 !important;
    }
    div[data-testid="stErrorMessage"] {
        background: linear-gradient(135deg, #1A0D0D, #1A0A0A) !important;
        border: 1px solid #E74C3C !important;
        border-left: 4px solid #E74C3C !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: transparent !important;
        color: #F4A623 !important;
        border: 1px solid #F4A623 !important;
        border-radius: 6px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #F4A623 !important;
        color: #080B14 !important;
    }

    /* ── Sidebar info pills ── */
    .sidebar-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 0;
    }

    /* ── Section card wrapper ── */
    .section-card {
        background: #0D1117;
        border: 1px solid #1E2D4A;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #F4A623 !important; }

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"

@st.cache_data(show_spinner="Loading & processing data…")
def load_data():
    # ── Sentiment ──────────────────────────────────────────────────────
    sent = pd.read_csv(DATA_DIR / "sentiment.csv")
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    sent = sent.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    def bucket(c):
        c = str(c).lower()
        if "fear"  in c: return "Fear"
        if "greed" in c: return "Greed"
        return "Neutral"

    sent["bucket"]   = sent["classification"].apply(bucket)
    sent["is_fear"]  = sent["bucket"] == "Fear"
    sent["is_greed"] = sent["bucket"] == "Greed"

    # ── Trades ─────────────────────────────────────────────────────────
    trades = pd.read_csv(DATA_DIR / "trades.csv")
    trades.columns = trades.columns.str.strip()
    trades.rename(columns={
        "Account": "account", "Coin": "coin",
        "Execution Price": "price", "Size Tokens": "size_tokens",
        "Size USD": "size_usd", "Side": "side",
        "Timestamp IST": "ts_ist", "Start Position": "start_pos",
        "Direction": "direction", "Closed PnL": "closed_pnl",
        "Fee": "fee", "Crossed": "crossed",
        "Order ID": "order_id", "Trade ID": "trade_id",
        "Transaction Hash": "tx_hash", "Timestamp": "timestamp_ms",
    }, inplace=True)

    trades["date"] = pd.to_datetime(trades["ts_ist"], dayfirst=True, errors="coerce").dt.normalize()
    trades = trades.dropna(subset=["date"])

    for col in ["price","size_tokens","size_usd","closed_pnl","fee","start_pos"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades["is_long"]  = trades["direction"].str.contains("Long|Buy",  case=False, na=False)
    trades["is_short"] = trades["direction"].str.contains("Short|Sell", case=False, na=False)

    date_min = max(sent["date"].min(), trades["date"].min())
    date_max = min(sent["date"].max(), trades["date"].max())
    trades = trades[(trades["date"] >= date_min) & (trades["date"] <= date_max)].copy()

    trades = trades.merge(
        sent[["date","classification","bucket","is_fear","is_greed"]],
        on="date", how="left"
    )
    trades["bucket"] = trades["bucket"].fillna("Neutral")

    trades["size_seg"] = pd.qcut(
        trades["size_usd"].clip(lower=0.01), q=[0,.33,.67,1.],
        labels=["Small","Medium","Large"], duplicates="drop"
    )

    # ── Daily metrics ──────────────────────────────────────────────────
    daily = (
        trades.groupby(["date","bucket","is_fear","is_greed"])
        .agg(
            n_trades   = ("closed_pnl","count"),
            total_pnl  = ("closed_pnl","sum"),
            win_rate   = ("closed_pnl", lambda x: (x[x!=0]>0).mean() if (x!=0).any() else np.nan),
            avg_size   = ("size_usd","mean"),
            long_ratio = ("is_long","mean"),
            total_fee  = ("fee","sum"),
        ).reset_index()
    )
    daily["pnl_per_trade"] = daily["total_pnl"] / daily["n_trades"].replace(0, np.nan)
    daily["net_pnl"]       = daily["total_pnl"] - daily["total_fee"]

    # ── Per-account metrics ────────────────────────────────────────────
    closed = trades[trades["closed_pnl"] != 0].copy()
    acct = (
        closed.groupby("account")
        .agg(
            n_trades  = ("closed_pnl","count"),
            total_pnl = ("closed_pnl","sum"),
            win_rate  = ("closed_pnl", lambda x: (x>0).mean()),
            pnl_std   = ("closed_pnl","std"),
            avg_size  = ("size_usd","mean"),
            total_fee = ("fee","sum"),
        ).reset_index()
    )
    acct["net_pnl"]      = acct["total_pnl"] - acct["total_fee"]
    acct["sharpe_proxy"] = acct["total_pnl"] / (acct["pnl_std"] + 1e-9)
    acct["freq_seg"]     = pd.qcut(acct["n_trades"], q=3,
                                   labels=["Infrequent","Moderate","Frequent"],
                                   duplicates="drop")
    acct["consistency"]  = acct["win_rate"].apply(
        lambda w: "Consistent Winner" if w >= 0.55
        else ("Consistent Loser" if w <= 0.45 else "Mixed")
    )

    return trades, daily, acct, closed, sent, date_min, date_max

trades, daily, acct, closed, sent, date_min, date_max = load_data()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem;
                    color: #F4A623; font-weight: 700; letter-spacing: 0.05em;'>
            HYPERLIQUID
        </div>
        <div style='font-size: 0.7rem; color: #5A7A9A; letter-spacing: 0.15em;
                    text-transform: uppercase; margin-top: 2px;'>
            × Fear/Greed Sentiment
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D4A; margin: 0.8rem 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:0.78rem; color:#5A7A9A; text-transform:uppercase;
                letter-spacing:0.08em; margin-bottom:0.6rem;'>Dataset</div>
    <div style='display:flex; flex-direction:column; gap:6px;'>
        <div style='display:flex; justify-content:space-between; font-size:0.83rem;'>
            <span style='color:#5A7A9A;'>Date range</span>
            <span style='color:#C8D6E5; font-family:monospace;'>{date_min.date()} → {date_max.date()}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.83rem;'>
            <span style='color:#5A7A9A;'>Total trades</span>
            <span style='color:#F4A623; font-family:monospace; font-weight:700;'>{len(trades):,}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.83rem;'>
            <span style='color:#5A7A9A;'>Accounts</span>
            <span style='color:#C8D6E5; font-family:monospace;'>{trades["account"].nunique()}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.83rem;'>
            <span style='color:#5A7A9A;'>Coins</span>
            <span style='color:#C8D6E5; font-family:monospace;'>{trades["coin"].nunique()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D4A; margin: 0.8rem 0;'>", unsafe_allow_html=True)

    sentiment_counts = daily.groupby("bucket")["date"].nunique()
    st.markdown("<div style='font-size:0.78rem; color:#5A7A9A; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;'>Sentiment Days</div>", unsafe_allow_html=True)
    for b in ORDER:
        color = BUCKET_COLORS[b]
        n = sentiment_counts.get(b, 0)
        pct = n / sentiment_counts.sum() * 100
        st.markdown(f"""
        <div style='margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;'>
                <span><span style='color:{color};'>▮</span> &nbsp;{b}</span>
                <span style='color:{color}; font-family:monospace; font-weight:700;'>{n}d</span>
            </div>
            <div style='background:#1E2D4A; border-radius:3px; height:3px;'>
                <div style='background:{color}; width:{pct}%; height:3px; border-radius:3px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D4A; margin: 0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem; color:#5A7A9A; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;'>Filters</div>", unsafe_allow_html=True)

    selected_buckets = st.multiselect(
        "Sentiment filter", ORDER, default=ORDER
    )
    top_n_coins = st.slider("Top N coins (heatmap)", 5, 20, 12)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Overview",
    "🔍  Behavioral Explorer",
    "🧬  Trader Clustering",
    "🤖  Predictive Model",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family: Space Mono, monospace; font-size:1.6rem;
                    color:#F4A623; font-weight:700; letter-spacing:-0.01em;'>
            Overview Dashboard
        </div>
        <div style='color:#5A7A9A; font-size:0.85rem; margin-top:4px; letter-spacing:0.03em;'>
            Key performance metrics across all sentiment regimes · 2023-05-01 to 2025-05-01
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades",     f"{len(trades):,}")
    col2.metric("Closed PnL Trades",f"{len(closed):,}")
    col3.metric("Unique Accounts",  f"{trades['account'].nunique()}")
    col4.metric("Fear Days",        f"{daily[daily['bucket']=='Fear']['date'].nunique()}")
    col5.metric("Greed Days",       f"{daily[daily['bucket']=='Greed']['date'].nunique()}")

    st.markdown("---")

    # Summary table
    st.markdown("### Summary Statistics by Sentiment")
    summary = (
        daily.groupby("bucket")[["n_trades","total_pnl","win_rate","avg_size","long_ratio","pnl_per_trade"]]
        .agg(["mean","median"])
        .reindex(ORDER)
    )
    summary.columns = [f"{m} {s}" for m,s in summary.columns]
    summary = summary.rename(columns=lambda c: c.replace("_"," ").title())
    st.dataframe(
        summary.style.format("{:,.2f}").background_gradient(cmap="RdYlGn", axis=0),
        use_container_width=True
    )

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Rolling PnL
    with c1:
        st.markdown("### 7-Day Rolling PnL Timeline")
        daily_s = daily.sort_values("date").copy()
        daily_s["roll7"] = daily_s["total_pnl"].rolling(7, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(9, 4))
        for _, row in daily_s.iterrows():
            color = BUCKET_COLORS.get(row["bucket"], "#888")
            ax.axvspan(row["date"]-pd.Timedelta(hours=12),
                       row["date"]+pd.Timedelta(hours=12),
                       alpha=0.08, color=color, linewidth=0)
        ax.plot(daily_s["date"], daily_s["roll7"], color="white", linewidth=1.8, label="7d MA")
        ax.axhline(0, color="#888", linewidth=0.7, linestyle="--")
        ax.set_ylabel("PnL (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=FEAR_COLOR,    alpha=0.5, label="Fear"),
            Patch(color=NEUTRAL_COLOR, alpha=0.5, label="Neutral"),
            Patch(color=GREED_COLOR,   alpha=0.5, label="Greed"),
            plt.Line2D([0],[0], color="white", label="7d MA"),
        ], fontsize=8, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)

    # Cumulative PnL
    with c2:
        st.markdown("### Cumulative PnL by Sentiment Regime")
        fig, ax = plt.subplots(figsize=(9, 4))
        for b in ORDER:
            sub = daily_s[daily_s["bucket"]==b].sort_values("date").copy()
            sub["cum"] = sub["total_pnl"].cumsum()
            ax.plot(sub["date"], sub["cum"], color=BUCKET_COLORS[b],
                    linewidth=1.8, label=b, alpha=0.9)
        ax.axhline(0, color="#888", linewidth=0.7, linestyle="--")
        ax.set_ylabel("Cumulative PnL (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — BEHAVIORAL EXPLORER
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Behavioral Explorer")
    st.markdown("Compare how traders behave across sentiment regimes.")
    st.markdown("---")

    filtered_daily = daily[daily["bucket"].isin(selected_buckets)]

    metric_options = {
        "Win Rate":        "win_rate",
        "Avg Trades/Day":  "n_trades",
        "Avg Trade Size":  "avg_size",
        "Long Ratio":      "long_ratio",
        "Avg PnL/Trade":   "pnl_per_trade",
        "Total Daily PnL": "total_pnl",
    }

    chosen = st.selectbox("Select metric to explore", list(metric_options.keys()))
    col = metric_options[chosen]

    agg = (
        filtered_daily.groupby("bucket")[col]
        .agg(["mean","sem","median"])
        .reindex([b for b in ORDER if b in selected_buckets])
        .reset_index()
    )

    c1, c2 = st.columns([1.3, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(
            agg["bucket"], agg["mean"],
            color=[BUCKET_COLORS[b] for b in agg["bucket"]],
            yerr=agg["sem"], capsize=7, width=0.5,
            edgecolor="black",
            error_kw={"ecolor":"white","elinewidth":1.5}
        )
        for bar, val in zip(bars, agg["mean"]):
            label = f"{val*100:.1f}%" if col in ["win_rate","long_ratio"] else f"${val:,.0f}" if "pnl" in col or "size" in col else f"{val:.0f}"
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + agg["sem"].max()*0.3,
                    label, ha="center", color="white", fontsize=11, fontweight="bold")
        ax.set_title(f"{chosen} by Sentiment", fontsize=12)
        ax.set_ylabel(chosen)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("#### Stats Table")
        display = agg.copy()
        display.columns = ["Sentiment","Mean","SEM","Median"]
        if col in ["win_rate","long_ratio"]:
            display["Mean"]   = display["Mean"].map(lambda x: f"{x*100:.2f}%")
            display["Median"] = display["Median"].map(lambda x: f"{x*100:.2f}%")
            display["SEM"]    = display["SEM"].map(lambda x: f"{x*100:.3f}%")
        else:
            display["Mean"]   = display["Mean"].map(lambda x: f"{x:,.2f}")
            display["Median"] = display["Median"].map(lambda x: f"{x:,.2f}")
            display["SEM"]    = display["SEM"].map(lambda x: f"{x:,.3f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("#### Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        for b in [b for b in ORDER if b in selected_buckets]:
            d = filtered_daily[filtered_daily["bucket"]==b][col].dropna()
            ax2.hist(d, bins=25, alpha=0.55, color=BUCKET_COLORS[b], label=b, edgecolor="black", linewidth=0.3)
        ax2.set_xlabel(chosen); ax2.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### Coin-Level PnL Heatmap")

    top_coins = closed.groupby("coin")["closed_pnl"].sum().nlargest(top_n_coins).index
    coin_sent = (
        closed[closed["coin"].isin(top_coins)]
        .groupby(["coin","bucket"])["closed_pnl"]
        .mean().unstack("bucket").reindex(columns=ORDER)
    )
    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(top_coins)*0.5)))
    sns.heatmap(coin_sent, ax=ax3, cmap="RdYlGn", center=0,
                annot=True, fmt=".0f", linewidths=0.5,
                cbar_kws={"label":"Avg PnL/Trade (USD)"},
                annot_kws={"size": 9})
    ax3.set_title(f"Avg PnL per Trade: Top {top_n_coins} Coins × Sentiment", fontsize=12, color="white")
    ax3.set_xlabel(""); ax3.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig3)

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — TRADER CLUSTERING
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Trader Clustering — Behavioral Archetypes")
    st.markdown("KMeans clustering groups traders by behavior. Each cluster represents a distinct trading archetype.")
    st.markdown("---")

    n_clusters = st.slider("Number of clusters (archetypes)", 2, 6, 4)

    cluster_features = ["n_trades","total_pnl","win_rate","pnl_std","avg_size","net_pnl","sharpe_proxy"]
    cluster_df = acct[cluster_features].fillna(0)

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(cluster_df)
    km        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    acct_c    = acct.copy()
    acct_c["Cluster"] = km.fit_predict(X_scaled).astype(str)

    # Cluster summary
    cluster_summary = (
        acct_c.groupby("Cluster")[["n_trades","total_pnl","win_rate","avg_size","sharpe_proxy"]]
        .mean().round(2)
    )
    cluster_summary.columns = ["Avg Trades","Avg Total PnL","Avg Win Rate","Avg Size","Sharpe Proxy"]

    # Auto-label archetypes
    archetype_map = {}
    for c in cluster_summary.index:
        row = cluster_summary.loc[c]
        if row["Avg Total PnL"] > cluster_summary["Avg Total PnL"].median() and row["Avg Win Rate"] > cluster_summary["Avg Win Rate"].median():
            archetype_map[c] = f"🏆 Cluster {c} — High Performer"
        elif row["Avg Trades"] > cluster_summary["Avg Trades"].median() and row["Avg Total PnL"] < cluster_summary["Avg Total PnL"].median():
            archetype_map[c] = f"🔄 Cluster {c} — Overtrader"
        elif row["Avg Size"] > cluster_summary["Avg Size"].median():
            archetype_map[c] = f"🎯 Cluster {c} — High Conviction"
        else:
            archetype_map[c] = f"💤 Cluster {c} — Passive"

    acct_c["Archetype"] = acct_c["Cluster"].map(archetype_map)

    c1, c2 = st.columns([1.4, 1])

    with c1:
        st.markdown("### Scatter: Trades vs Net PnL")
        cluster_colors = ["#E74C3C","#2ECC71","#F39C12","#3498DB","#9B59B6","#1ABC9C"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, c in enumerate(sorted(acct_c["Cluster"].unique())):
            d = acct_c[acct_c["Cluster"]==c]
            ax.scatter(d["n_trades"], d["net_pnl"],
                       color=cluster_colors[i % len(cluster_colors)],
                       alpha=0.8, s=100, edgecolors="black", linewidth=0.5,
                       label=archetype_map[c])
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Number of Trades")
        ax.set_ylabel("Net PnL after Fees (USD)")
        ax.set_title("Trader Clusters", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("### Cluster Profiles")
        st.dataframe(
            cluster_summary.style.format("{:,.2f}").background_gradient(cmap="RdYlGn", axis=0),
            use_container_width=True
        )

        st.markdown("### Archetype Labels")
        for c, label in archetype_map.items():
            n = (acct_c["Cluster"]==c).sum()
            st.markdown(f"**{label}** — {n} trader(s)")

    st.markdown("---")
    st.markdown("### Win Rate vs Sharpe Proxy by Cluster")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for i, c in enumerate(sorted(acct_c["Cluster"].unique())):
        d = acct_c[acct_c["Cluster"]==c]
        ax2.scatter(d["win_rate"]*100, d["sharpe_proxy"],
                    color=cluster_colors[i % len(cluster_colors)],
                    alpha=0.8, s=100, edgecolors="black", linewidth=0.5,
                    label=archetype_map[c])
    ax2.set_xlabel("Win Rate (%)")
    ax2.set_ylabel("Sharpe Proxy")
    ax2.set_title("Win Rate vs Risk-Adjusted Return by Cluster", fontsize=12)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

# ─────────────────────────────────────────────────────────────────────
# TAB 4 — PREDICTIVE MODEL
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Predictive Model")
    st.markdown("Predicts whether the next day will be a **profitable day** (positive platform PnL) using sentiment + behavioral features.")
    st.markdown("---")

    @st.cache_resource(show_spinner="Training model…")
    def train_model():
        model_df = daily.dropna(subset=["win_rate","long_ratio","avg_size"]).copy()
        model_df["target"]       = (model_df["total_pnl"] > 0).astype(int)
        model_df["is_fear_int"]  = model_df["is_fear"].astype(int)
        model_df["is_greed_int"] = model_df["is_greed"].astype(int)
        feat_cols = ["is_fear_int","is_greed_int","n_trades","win_rate","long_ratio","avg_size"]
        X = model_df[feat_cols].fillna(0)
        y = model_df["target"]
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
        clf.fit(X, y)
        imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
        return clf, scores, imp, feat_cols, model_df, X, y

    clf, scores, imp, feat_cols, model_df, X, y = train_model()

    # Model performance
    c1, c2, c3 = st.columns(3)
    c1.metric("5-Fold AUC",  f"{scores.mean():.3f}")
    c2.metric("AUC Std Dev", f"± {scores.std():.3f}")
    c3.metric("Training Days", f"{len(model_df)}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### Feature Importance")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [GREED_COLOR if i==0 else ACCENT if i==1 else "#3498DB" for i in range(len(imp))]
        imp.plot.barh(ax=ax, color=colors, edgecolor="black")
        ax.set_title(f"Feature Importance  (AUC = {scores.mean():.3f})", fontsize=11)
        ax.set_xlabel("Importance Score")
        for i, (val, name) in enumerate(zip(imp.values, imp.index)):
            ax.text(val + 0.002, i, f"{val:.3f}", va="center", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### AUC per Fold")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fold_colors = [GREED_COLOR if s >= scores.mean() else FEAR_COLOR for s in scores]
        ax2.bar([f"Fold {i+1}" for i in range(len(scores))], scores,
                color=fold_colors, edgecolor="black", width=0.5)
        ax2.axhline(scores.mean(), color=ACCENT, linewidth=1.5, linestyle="--",
                    label=f"Mean AUC = {scores.mean():.3f}")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("AUC Score")
        ax2.set_title("Cross-Validation AUC by Fold", fontsize=11)
        ax2.legend(fontsize=9)
        for i, s in enumerate(scores):
            ax2.text(i, s + 0.01, f"{s:.3f}", ha="center", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### 🔮 Predict Next-Day Profitability")
    st.markdown("Adjust the inputs below to simulate different market conditions:")

    p1, p2 = st.columns(2)
    with p1:
        sentiment_input = st.selectbox("Tomorrow's expected sentiment", ["Fear","Neutral","Greed"])
        n_trades_input  = st.slider("Expected trade count", 50, 2000,
                                    int(daily["n_trades"].median()), step=10)
        win_rate_input  = st.slider("Expected win rate", 0.0, 1.0,
                                    float(daily["win_rate"].median()), step=0.01)
    with p2:
        long_ratio_input = st.slider("Expected long ratio", 0.0, 1.0,
                                     float(daily["long_ratio"].median()), step=0.01)
        avg_size_input   = st.slider("Expected avg trade size (USD)", 1000, 30000,
                                     int(daily["avg_size"].median()), step=500)

    is_fear_in  = 1 if sentiment_input == "Fear"  else 0
    is_greed_in = 1 if sentiment_input == "Greed" else 0

    input_vec = np.array([[is_fear_in, is_greed_in, n_trades_input,
                           win_rate_input, long_ratio_input, avg_size_input]])
    prob      = clf.predict_proba(input_vec)[0][1]
    pred      = clf.predict(input_vec)[0]

    st.markdown("---")
    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        if pred == 1:
            st.success(f"### ✅ Profitable Day Predicted\n**Confidence: {prob*100:.1f}%**")
        else:
            st.error(f"### ❌ Unprofitable Day Predicted\n**Confidence: {(1-prob)*100:.1f}%**")

    with result_col2:
        fig3, ax3 = plt.subplots(figsize=(6, 2))
        bar_color = GREED_COLOR if pred == 1 else FEAR_COLOR
        ax3.barh(["Profitability\nProbability"], [prob], color=bar_color,
                 edgecolor="black", height=0.4)
        ax3.barh(["Profitability\nProbability"], [1-prob], left=[prob],
                 color="#333", edgecolor="black", height=0.4)
        ax3.axvline(0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability")
        ax3.text(prob/2, 0, f"{prob*100:.1f}%", ha="center", va="center",
                 color="white", fontsize=12, fontweight="bold")
        ax3.set_title("Predicted Probability of Profitable Day", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("---")
    st.markdown("""
    **Model Notes:**
    - Target: Will platform total PnL be positive on this day?
    - Features: Sentiment flags + behavioral metrics (trades, win rate, long ratio, avg size)
    - Model: Gradient Boosting Classifier, 5-fold cross-validation
    - Top predictors: win_rate, n_trades, avg_size (sentiment adds signal on top)
    """)