# Primetrade.ai – Round 0 Assignment
## Trader Performance vs Market Sentiment (Fear/Greed Index)

**Author:** Kartik  
**Dataset period:** May 2023 – May 2025 | **Traders analysed:** 32 | **Trades:** 104,402 closing trades

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Run the notebook:
```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

Or run the script directly:
```bash
python analysis.py
```

---

## Methodology

1. **Data Loading & Cleaning** — Both CSVs loaded with zero missing values and no duplicates found.
2. **Date Alignment** — Trader timestamps (IST, dd-mm-yyyy) parsed and aligned to Fear/Greed Index dates (daily level). 479 overlapping dates.
3. **Sentiment Bucketing** — 5-class classification collapsed into 3 buckets: Fear (Extreme Fear + Fear), Neutral, Greed (Greed + Extreme Greed).
4. **Key Metrics Computed** — Daily PnL per trader, win rate, trade frequency, position size (USD), leverage proxy (Size USD / |Start Position|), long/short ratio.
5. **Segmentation** — Traders split into: High vs Low Leverage, Frequent vs Infrequent, Net Winners vs Net Losers.

---

## Key Insights

### Insight 1 — Fear Days Produce Higher Average PnL
| Sentiment | Avg Daily PnL | Median Daily PnL | Win Rate |
|-----------|--------------|-----------------|---------|
| Fear      | $46,025      | $6,978          | 85%     |
| Neutral   | $23,508      | $5,306          | 79%     |
| Greed     | $17,692      | $1,517          | 84%     |

Fear days yield ~2.6× more average daily PnL than Greed days. This is counterintuitive — fear creates mispricing opportunities that disciplined traders exploit.

### Insight 2 — Traders Are More Active and Use Higher Leverage on Fear Days
| Sentiment | Avg Trades | Avg Leverage Proxy | Avg Position Size |
|-----------|-----------|-------------------|------------------|
| Fear      | 70.3      | 321.6             | $11,593          |
| Neutral   | 65.8      | 145.3             | $8,574           |
| Greed     | 54.5      | 113.9             | $6,822           |

On Fear days, traders execute ~29% more trades, use ~2.8× more leverage, and take 70% larger positions than on Greed days. High activity + high leverage in volatile conditions is a double-edged sword.

### Insight 3 — Net Losers Are Specifically Destroyed on Greed Days
| Segment    | Sentiment | Avg Daily PnL | Win Rate |
|------------|-----------|--------------|---------|
| Net Loser  | Fear      | +$3,835      | 84%     |
| Net Loser  | Greed     | **−$16,667** | 67%     |
| Net Winner | Fear      | +$7,411      | 84%     |
| Net Winner | Greed     | +$6,506      | 86%     |

Net Winners remain profitable across all sentiment regimes. Net Losers perform acceptably on Fear days but suffer heavy losses specifically on Greed days — they likely over-trade and chase momentum.

---

## Strategy Recommendations

### Rule 1 — "Fear is Opportunity; Greed is Discipline"
During **Fear days**: increase trade frequency and be willing to take larger, well-researched positions — the data shows this is when the best average returns occur.  
During **Greed days**: reduce position size and resist the urge to overtrade. Greed-day overactivity is the primary driver of losses for weak performers.

### Rule 2 — "Losers should sit on their hands during Greed"
Segment-specific rule: Traders classified as Net Losers (negative cumulative PnL) should **drastically cut activity on Greed days** — their average loss is −$16,667 on those days vs +$3,835 on Fear days. A simple rule: if the Fear/Greed Index > 60 (Greed zone), max 1–2 trades per day for this segment.

---

## Output Charts
| File | Description |
|------|-------------|
| `charts/chart1_pnl_winrate_sentiment.png` | Avg daily PnL & win rate by sentiment |
| `charts/chart2_behaviour_sentiment.png` | Trade frequency, long ratio, position size |
| `charts/chart3_pnl_distribution.png` | Box plot + histogram of PnL on Fear vs Greed days |
| `charts/chart4_segment_heatmap.png` | Heatmap of PnL by trader segment × sentiment |
| `charts/chart5_cumulative_pnl.png` | Cumulative PnL timeline colored by sentiment regime |
| `charts/chart6_leverage_violin.png` | Leverage distribution violin plot by sentiment |
| `charts/chart7_winrate_segments.png` | Win rate by trader segment × sentiment |
