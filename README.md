# Hyperliquid × Fear/Greed Sentiment Analysis

> Uncovering how Bitcoin market sentiment shapes trader behavior and performance on Hyperliquid.

---

## Quick Start

```bash
# 1. Clone / download this repo
git clone https://github.com/mithraom/Trader-Performance-vs-Market-Sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Drop your data files into data/
#    data/sentiment.csv  — columns: timestamp, value, classification, date
#    data/trades.csv     — columns: Account, Coin, Execution Price, Size Tokens,
#                          Size USD, Side, Timestamp IST, Direction, Closed PnL, Fee, ...

# 4. Run the full analysis (generates all charts + output files)
python analysis.py
# OR open analysis.ipynb in Jupyter and run all cells

# 5. Launch the interactive dashboard (bonus)
streamlit run app_streamlit.py
```

> **Note:** Data files are excluded from this repo (proprietary to Primetrade.ai).
> Place `sentiment.csv` and `trades.csv` in the `data/` folder to reproduce all results.

---

## Project Structure
```
PrimetradeAI_Project/
├── analysis.py               ← Full analysis script (run this first)
├── analysis.ipynb            ← Jupyter notebook with inline outputs
├── app_streamlit.py          ← Interactive dashboard (bonus)
├── requirements.txt
├── README.md
├── data/
│   ├── sentiment.csv         ← Fear/Greed index (place real file here)
│   └── trades.csv            ← Hyperliquid trades (place real file here)
├── charts/
│   ├── 01_pnl_distribution_by_sentiment.png
│   ├── 02_win_rate_by_sentiment.png
│   ├── 03_trade_frequency_by_sentiment.png
│   ├── 04_avg_size_by_sentiment.png
│   ├── 05_long_ratio_by_sentiment.png
│   ├── 06_pnl_by_direction_and_sentiment.png
│   ├── 07_pnl_by_frequency_segment.png
│   ├── 08_pnl_by_size_segment.png
│   ├── 09_consistency_scatter.png
│   ├── 10_rolling_pnl_timeline.png
│   ├── 11_coin_pnl_heatmap.png
│   ├── 12_cumulative_pnl_by_sentiment.png
│   └── 13_feature_importance.png
└── output/
    ├── summary_stats.csv
    ├── account_segments.csv
    └── insights_and_strategy.txt

```
---

## Methodology

### Part A — Data Preparation
1. **Load** both CSVs with column-name normalisation tailored to the exact Hyperliquid export schema.
2. **Quality check**: reported shape, missing values, duplicates; 0 duplicates and 0 null PnL rows found.
3. **Timestamp alignment**: parsed `Timestamp IST` (dayfirst format) → date (day-level); left-joined sentiment on date; restricted to overlapping range 2023-05-01 to 2025-05-01.
4. **Feature engineering**:
   - Daily: `total_pnl`, `n_trades`, `win_rate`, `avg_size`, `long_ratio`, `pnl_per_trade`, `net_pnl` (after fees)
   - Per-account: cumulative `total_pnl`, `win_rate`, `pnl_std`, Sharpe proxy, `n_trades`, `net_pnl`
   - Segmentation labels (tercile cuts): position size (Small/Medium/Large), frequency (Infrequent/Moderate/Frequent), consistency (Winner ≥55% / Loser ≤45% / Mixed by win-rate)
   - **Leverage note**: The dataset does not include a direct leverage column; position size (USD) is used as a proxy for leverage exposure.

### Part B — Analysis

| Question | Method | Chart |
|---|---|---|
| PnL difference Fear vs Greed? | Histogram + mean/median comparison | 01 |
| Win rate by sentiment? | Bar chart with SEM error bars | 02 |
| Trade frequency by sentiment? | Bar chart with SEM error bars | 03 |
| Position sizing by sentiment? | Bar chart with SEM error bars | 04 |
| Long/short bias by sentiment? | Bar chart | 05 |
| PnL by direction type & sentiment? | Grouped bar (Open/Close Long/Short) | 06 |
| Frequency segment performance? | Grouped bar × sentiment | 07 |
| Size segment performance? | Grouped bar × sentiment | 08 |
| Consistency archetypes? | Scatter (n_trades vs net PnL, coloured by archetype) | 09 |
| Temporal patterns? | 7-day rolling PnL, sentiment-shaded background | 10 |
| Coin-level sentiment breakdown? | Heatmap: top 12 coins × sentiment | 11 |
| Cumulative PnL by sentiment regime? | Line chart by Fear/Neutral/Greed | 12 |

### Part C — Predictive Model (Bonus)
- **Target**: Will the platform generate positive PnL on a given day? (binary)
- **Features**: `is_fear`, `is_greed`, `n_trades`, `win_rate`, `long_ratio`, `avg_size`
- **Model**: Gradient Boosting Classifier, 5-fold cross-validation
- **Result**: AUC = 0.933 ± 0.053 — sentiment + behavioral features are strong predictors of daily PnL direction
- **Top features**: win_rate, n_trades, avg_size

### Bonus — Interactive Dashboard (Streamlit)

Four-tab interactive app for exploring all results without running the analysis script.

```bash
streamlit run app_streamlit.py
```

| Tab | Contents |
|-----|----------|
|  Overview | KPI metrics, summary stats table, rolling PnL timeline, cumulative PnL by regime |
|  Behavioral Explorer | Filter by sentiment, compare any metric, coin heatmap with adjustable top-N |
|  Trader Clustering | KMeans archetypes (adjustable 2–6 clusters), scatter plots, auto-labeled profiles |
|  Predictive Model | AUC per fold, feature importance chart, live next-day profitability simulator |

---

## Key Insights

### Insight 1 — Win Rate Is Stable Across Sentiments (Execution Quality Is Consistent)
Fear avg win rate: **84.5%** | Greed avg win rate: **84.2%** | Gap: **0.3 percentage points**

Win rates are remarkably stable across all sentiment regimes. These 32 accounts maintain execution quality regardless of market mood — the signal lies not in accuracy but in how volume and sizing shift across regimes.

### Insight 2 — Fear Days Drive 2.7× More Trading Activity
Fear avg trades/day: **793** | Greed avg trades/day: **294** | Delta: **499 trades/day**

Traders are far more active on Fear days. Crucially, avg daily PnL on Fear days ($39,012) also outpaces Greed days ($15,848) — Fear-driven volatility creates a richer opportunity set that skilled accounts exploit effectively.

### Insight 3 — Traders Go Long Into Fear (Contrarian Positioning)
Fear avg long ratio: **60.9%** | Greed avg long ratio: **51.8%**

Contrary to expectations, traders hold a higher long bias during Fear. These accounts appear contrarian or mean-reversion oriented — buying into weakness. When sentiment turns, this positioning benefits from the subsequent rally.

### Insight 4 — Position Sizes Are Larger on Fear Days (Deliberate, Not Panic)
Fear avg trade size: **$6,200** | Greed avg trade size: **$5,872**

Traders size up slightly during Fear, consistent with the contrarian long bias — deploying more capital into weakness expecting reversals. Combined with higher trade frequency, Fear days represent the most active and profitable regime for this cohort.

### Insight 5 — All 32 Accounts Are Consistent Winners
Consistent Winners (win rate ≥ 55%): **32** | Consistent Losers: **0** | Mixed: **0**

Every account in this dataset exceeds a 55% win rate — an extraordinary result indicating this is not a random sample of retail traders but a cohort of high-performing or institutional-grade accounts. Strategies derived here reflect the behavior of skilled, disciplined traders.

---

## Strategy Recommendations

### Strategy 1 — "Sentiment-Gated Position Sizing"
**Who**: All traders, especially the Large-size segment  
**Rule**: Reduce individual trade size to **60% of normal on Fear days**. Resume full sizing only after ≥ 2 consecutive Greed days.  
**Why**: Fear-day trades are already larger ($6,200 vs $5,872) and occur at 2.7× frequency. Uncapped sizing during high-frequency Fear trading compounds drawdown risk. Capping size preserves capital while still participating in the volatility opportunity.

### Strategy 2 — "Fear Aggression, Greed Patience"
**Who**: Frequent traders (top 33% by trade count)  
**Rule**: On Fear days, scale frequency to **1.25× baseline** but only for highest-conviction setups with defined stops. On Greed days, scale back to **0.75× baseline** — opportunity density is lower.  
**Why**: This dataset shows Fear days generate higher absolute PnL ($39,012 vs $15,848). Skilled frequent traders should lean into volatile conditions rather than retreat from them.

### Bonus Rule — "Contrarian Long on Fear Extremes"
**Trigger**: Platform long_ratio > 60% AND sentiment reads Fear AND sentiment transitions Fear → Neutral/Greed  
**Action**: Hold or add to long positions in top-liquidity coins (BTC, ETH, SOL), 1–2× normal size, stop below recent swing low  
**Why**: Insight 3 shows these accounts go long during Fear and profit from it. The sentiment flip acts as a confirmation signal — existing long positioning then accelerates as the market recovers.

---

## Summary Statistics

| Metric | Fear | Neutral | Greed |
|--------|------|---------|-------|
| Avg trades/day | 793 | 562 | 294 |
| Avg daily PnL | $39,012 | $19,297 | $15,848 |
| Avg win rate | 84.5% | 79.4% | 84.2% |
| Avg trade size | $6,200 | $7,158 | $5,872 |
| Avg long ratio | 60.9% | 64.8% | 51.8% |
| Avg PnL/trade | $32.23 | $63.82 | $45.85 |

---

## Requirements
```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
streamlit>=1.28

```

Python 3.9+ recommended.

---
