# Crypto Trader Sentiment Analysis

## Overview

This project looks at how market sentiment (Fear vs Greed) affects trader performance and behavior.

The analysis uses historical trading data along with the Fear & Greed Index to see how things like PnL, trading activity, and leverage change under different market conditions.

The dataset covers May 2023 to May 2025, with over 100,000 trades across 32 traders.

---

## Setup

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Run the notebook:

jupyter notebook trader_sentiment_analysis.ipynb

Or run the script:

python analysis.py

---

## Methodology

First, both datasets were loaded and checked. There were no missing values or duplicates.

Dates from the trading data were aligned with the Fear & Greed Index so that each day could be assigned a sentiment.

The sentiment categories were simplified into three groups:
- Fear (including Extreme Fear)
- Neutral
- Greed (including Extreme Greed)

After that, key metrics were calculated such as daily PnL, win rate, number of trades, position size, and a simple leverage proxy.

Traders were also grouped into segments like net winners and net losers to compare behavior.

---

## Key Insights

Fear days show much higher average PnL compared to Greed days (around 2.6× higher). This suggests that traders perform better when the market is more volatile.

Trading activity also increases during Fear periods. Traders place more trades, use higher leverage, and take larger positions.

One important pattern is that net losing traders perform much worse during Greed periods. They tend to lose heavily in those conditions, while they do relatively better during Fear.

On the other hand, net winning traders remain consistently profitable across all sentiment types.

---

## Strategy Takeaways

Fear periods seem to offer better opportunities, but they also come with higher risk. Traders tend to be more active and aggressive during these times.

Greed periods require more discipline. Overtrading during these conditions is linked with higher losses, especially for weaker traders.

A simple takeaway is:
- Be more active and selective during Fear
- Be more cautious during Greed

---

## Outputs

The project generates multiple charts showing:
- PnL and win rate across sentiment
- Trading behavior differences
- Distribution of returns
- Segment-wise performance

All charts are saved in the charts/ folder.

---

