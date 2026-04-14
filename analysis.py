import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 130, 'font.family': 'DejaVu Sans'})

CHARTS = 'charts/'

# Load data
fg_raw  = pd.read_csv('fear_greed_index.csv')
hist_raw = pd.read_csv('historical_data.csv')

print("=" * 55)
print("PART A - DATA PREPARATION")
print("=" * 55)
print(f"\nFear/Greed  : {fg_raw.shape[0]:,} rows x {fg_raw.shape[1]} cols")
print(f"Trader Data : {hist_raw.shape[0]:,} rows x {hist_raw.shape[1]} cols")

print("\n[Fear/Greed] Missing values:\n", fg_raw.isnull().sum().to_string())
print(f"\n[Fear/Greed] Duplicate rows: {fg_raw.duplicated().sum()}")
print("\n[Trader Data] Missing values:\n", hist_raw.isnull().sum().to_string())
print(f"\n[Trader Data] Duplicate rows: {hist_raw.duplicated().sum()}")

fg = fg_raw.copy()
fg['date'] = pd.to_datetime(fg['date'])

hist = hist_raw.copy()
hist['date'] = pd.to_datetime(hist['Timestamp IST'].str[:10], format='%d-%m-%Y')

# Map detailed sentiments to broad Fear/Neutral/Greed categories
fg['sentiment'] = fg['classification'].map({
    'Extreme Fear': 'Fear', 'Fear': 'Fear',
    'Neutral': 'Neutral',
    'Greed': 'Greed', 'Extreme Greed': 'Greed'
})

trader_daily = hist.merge(fg[['date', 'classification', 'sentiment', 'value']],
                           on='date', how='inner')
print(f"\nPost-merge rows: {trader_daily.shape[0]:,}")
print(f"Date range in merged data: {trader_daily['date'].min().date()} to {trader_daily['date'].max().date()}")

# Extract closing trades and compute key metrics
closes = trader_daily[trader_daily['Closed PnL'] != 0].copy()
closes['win'] = closes['Closed PnL'] > 0
closes['is_long'] = closes['Direction'].str.contains('Long', na=False)
closes['lev_proxy'] = np.where(
    closes['Start Position'].abs() > 10,
    closes['Size USD'] / closes['Start Position'].abs(),
    np.nan
)

# Daily metrics per trader
daily_trader = closes.groupby(['date', 'Account', 'sentiment', 'classification', 'value']).agg(
    daily_pnl     = ('Closed PnL', 'sum'),
    n_trades      = ('Closed PnL', 'count'),
    win_rate      = ('win', 'mean'),
    avg_size_usd  = ('Size USD', 'mean'),
    avg_lev       = ('lev_proxy', 'mean'),
    long_ratio    = ('is_long', 'mean'),
).reset_index()

# Daily market-wide metrics
daily_mkt = closes.groupby(['date', 'sentiment', 'classification', 'value']).agg(
    total_pnl     = ('Closed PnL', 'sum'),
    n_trades      = ('Closed PnL', 'count'),
    win_rate      = ('win', 'mean'),
    avg_size_usd  = ('Size USD', 'mean'),
    avg_lev       = ('lev_proxy', 'mean'),
    long_ratio    = ('is_long', 'mean'),
).reset_index()

print(f"\nUnique traders: {closes['Account'].nunique()}")
print(f"Unique trading days: {closes['date'].nunique()}")
print(f"Total closing trades analysed: {len(closes):,}")
print("\nSentiment distribution in merged data:")
print(daily_mkt['sentiment'].value_counts().to_string())

# B1: Performance by Sentiment
sent_perf = daily_mkt.groupby('sentiment').agg(
    avg_daily_pnl = ('total_pnl', 'mean'),
    median_daily_pnl = ('total_pnl', 'median'),
    avg_win_rate  = ('win_rate', 'mean'),
    days          = ('total_pnl', 'count')
).round(2)
print("\n[B1] Performance by Sentiment:\n", sent_perf.to_string())

drawdown = daily_mkt.groupby('sentiment')['total_pnl'].min().rename('worst_day_pnl')
print("\nWorst day PnL by sentiment:\n", drawdown.to_string())

# B2: Trader Behaviour by Sentiment
bhv = daily_trader.groupby('sentiment').agg(
    avg_trades    = ('n_trades', 'mean'),
    avg_lev       = ('avg_lev', 'mean'),
    avg_long_ratio= ('long_ratio', 'mean'),
    avg_size_usd  = ('avg_size_usd', 'mean'),
).round(3)
print("\n[B2] Trader behaviour by sentiment:\n", bhv.to_string())

# B3: Trader Segmentation
trader_overall = closes.groupby('Account').agg(
    total_pnl    = ('Closed PnL', 'sum'),
    total_trades = ('Closed PnL', 'count'),
    win_rate     = ('win', 'mean'),
    avg_lev      = ('lev_proxy', 'mean'),
    avg_size     = ('Size USD', 'mean'),
).reset_index()

lev_med = trader_overall['avg_lev'].median()
trader_overall['lev_seg'] = np.where(trader_overall['avg_lev'] >= lev_med, 'High Leverage', 'Low Leverage')

freq_med = trader_overall['total_trades'].median()
trader_overall['freq_seg'] = np.where(trader_overall['total_trades'] >= freq_med, 'Frequent', 'Infrequent')

trader_overall['perf_seg'] = np.where(trader_overall['total_pnl'] > 0, 'Net Winner', 'Net Loser')

print("\n[B3] Trader segments summary:")
print(trader_overall[['lev_seg','freq_seg','perf_seg']].apply(pd.Series.value_counts))

merged_seg = daily_trader.merge(
    trader_overall[['Account','lev_seg','freq_seg','perf_seg']], on='Account')

seg_sent = merged_seg.groupby(['perf_seg','sentiment']).agg(
    avg_pnl      = ('daily_pnl', 'mean'),
    avg_win_rate = ('win_rate', 'mean'),
    avg_trades   = ('n_trades', 'mean'),
).round(3)
print("\n[B3] Net Winner/Loser vs Sentiment:\n", seg_sent.to_string())

lev_sent = merged_seg.groupby(['lev_seg','sentiment']).agg(
    avg_pnl      = ('daily_pnl', 'mean'),
    avg_win_rate = ('win_rate', 'mean'),
    avg_lev      = ('avg_lev', 'mean'),
).round(3)
print("\n[B3] Leverage Segment vs Sentiment:\n", lev_sent.to_string())

# Chart configuration
SENT_ORDER = ['Fear', 'Neutral', 'Greed']
SENT_COLORS = {'Fear': '#e74c3c', 'Neutral': '#f39c12', 'Greed': '#27ae60'}
palette = [SENT_COLORS[s] for s in SENT_ORDER]

# Chart 1: PnL and Win Rate by Sentiment
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sent_perf_ordered = sent_perf.reindex(SENT_ORDER)
bars = axes[0].bar(SENT_ORDER, sent_perf_ordered['avg_daily_pnl'], color=palette, edgecolor='white', linewidth=0.8)
axes[0].set_title('Avg Daily Total PnL by Sentiment', fontsize=13, fontweight='bold')
axes[0].set_ylabel('USD')
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
for bar, val in zip(bars, sent_perf_ordered['avg_daily_pnl']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

wr_ordered = sent_perf.reindex(SENT_ORDER)
bars2 = axes[1].bar(SENT_ORDER, wr_ordered['avg_win_rate'] * 100, color=palette, edgecolor='white', linewidth=0.8)
axes[1].set_title('Avg Win Rate by Sentiment', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 80)
for bar, val in zip(bars2, wr_ordered['avg_win_rate'] * 100):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(CHARTS + 'chart1_pnl_winrate_sentiment.png', bbox_inches='tight')
plt.close()
print("\nSaved chart 1")

# Chart 2: Behaviour metrics by Sentiment
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bhv_ordered = bhv.reindex(SENT_ORDER)

metrics = [('avg_trades', 'Avg Trades per Trader-Day', ''), 
           ('avg_long_ratio', 'Avg Long Ratio', ''),
           ('avg_size_usd', 'Avg Position Size (USD)', '$')]

for ax, (col, title, prefix) in zip(axes, metrics):
    bars = ax.bar(SENT_ORDER, bhv_ordered[col], color=palette, edgecolor='white', linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for bar, val in zip(bars, bhv_ordered[col]):
        label = f'{prefix}{val:,.2f}' if prefix == '$' else f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Trader Behaviour by Market Sentiment', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(CHARTS + 'chart2_behaviour_sentiment.png', bbox_inches='tight')
plt.close()
print("Saved chart 2")

# Chart 3: PnL Distribution Fear vs Greed
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fg_subset = daily_trader[daily_trader['sentiment'].isin(['Fear', 'Greed'])]
fear_pnl  = fg_subset[fg_subset['sentiment'] == 'Fear']['daily_pnl']
greed_pnl = fg_subset[fg_subset['sentiment'] == 'Greed']['daily_pnl']

# Clip for viz
clip = 5000
bp = axes[0].boxplot(
    [fear_pnl.clip(-clip, clip), greed_pnl.clip(-clip, clip)],
    labels=['Fear Days', 'Greed Days'],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#27ae60')
axes[0].set_title('Daily PnL Distribution: Fear vs Greed', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Daily PnL per Trader (USD, clipped +/- 5k)')
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)

# Histogram overlay
axes[1].hist(fear_pnl.clip(-clip, clip), bins=50, alpha=0.6, color='#e74c3c', label='Fear')
axes[1].hist(greed_pnl.clip(-clip, clip), bins=50, alpha=0.6, color='#27ae60', label='Greed')
axes[1].set_title('PnL Distribution Histogram', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Daily PnL per Trader (USD)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(CHARTS + 'chart3_pnl_distribution.png', bbox_inches='tight')
plt.close()
print("Saved chart 3")

# Chart 4: Segment vs Sentiment heatmaps
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Net Winner/Loser heatmap
pivot1 = merged_seg.groupby(['perf_seg','sentiment'])['daily_pnl'].mean().unstack()[SENT_ORDER]
sns.heatmap(pivot1, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=axes[0],
            linewidths=0.5, cbar_kws={'label': 'Avg Daily PnL (USD)'})
axes[0].set_title('Avg Daily PnL: Winner/Loser vs Sentiment', fontsize=11, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('')

# Leverage segment heatmap
pivot2 = merged_seg.groupby(['lev_seg','sentiment'])['daily_pnl'].mean().unstack()[SENT_ORDER]
sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=axes[1],
            linewidths=0.5, cbar_kws={'label': 'Avg Daily PnL (USD)'})
axes[1].set_title('Avg Daily PnL: Leverage Segment vs Sentiment', fontsize=11, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(CHARTS + 'chart4_segment_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved chart 4")

# Chart 5: Cumulative PnL over time
daily_cumulative = daily_mkt.sort_values('date').copy()
daily_cumulative['cum_pnl'] = daily_cumulative['total_pnl'].cumsum()

fig, ax = plt.subplots(figsize=(14, 5))
for sent, color in SENT_COLORS.items():
    mask = daily_cumulative['sentiment'] == sent
    ax.fill_between(daily_cumulative['date'],
                    daily_cumulative['cum_pnl'],
                    where=mask, alpha=0.3, color=color, label=sent)

ax.plot(daily_cumulative['date'], daily_cumulative['cum_pnl'], color='navy', linewidth=1.2, label='Cumulative PnL')
ax.set_title('Cumulative Total PnL Over Time (colored by sentiment regime)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative PnL (USD)')
ax.set_xlabel('Date')
ax.legend(loc='upper left')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig(CHARTS + 'chart5_cumulative_pnl.png', bbox_inches='tight')
plt.close()
print("Saved chart 5")

# Chart 6: Leverage distribution by sentiment
fig, ax = plt.subplots(figsize=(10, 5))
lev_data = merged_seg[merged_seg['avg_lev'].notna() & (merged_seg['avg_lev'] < 50)]
sent_order_plot = [s for s in SENT_ORDER if s in lev_data['sentiment'].unique()]
sns.violinplot(data=lev_data, x='sentiment', y='avg_lev', order=sent_order_plot,
               palette=SENT_COLORS, inner='quartile', ax=ax)
ax.set_title('Leverage Distribution by Sentiment', fontsize=13, fontweight='bold')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Avg Leverage Proxy (per trader-day)')
plt.tight_layout()
plt.savefig(CHARTS + 'chart6_leverage_violin.png', bbox_inches='tight')
plt.close()
print("Saved chart 6")

# Chart 7: Win Rate by Segment vs Sentiment
fig, ax = plt.subplots(figsize=(11, 5))
seg_wr = merged_seg.groupby(['perf_seg','sentiment'])['win_rate'].mean().reset_index()
seg_wr = seg_wr[seg_wr['sentiment'].isin(SENT_ORDER)]
x_labels = seg_wr['perf_seg'].unique()
x = np.arange(len(x_labels))
width = 0.25
for i, sent in enumerate(SENT_ORDER):
    vals = [seg_wr[(seg_wr['perf_seg']==lbl) & (seg_wr['sentiment']==sent)]['win_rate'].values[0]
            if len(seg_wr[(seg_wr['perf_seg']==lbl) & (seg_wr['sentiment']==sent)]) > 0 else 0
            for lbl in x_labels]
    ax.bar(x + i*width, [v*100 for v in vals], width, label=sent, color=SENT_COLORS[sent], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(x_labels)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Trader Segment vs Sentiment', fontsize=13, fontweight='bold')
ax.legend(title='Sentiment')
ax.set_ylim(0, 80)
plt.tight_layout()
plt.savefig(CHARTS + 'chart7_winrate_segments.png', bbox_inches='tight')
plt.close()
print("Saved chart 7")

print("\n[SUCCESS] All charts saved.")
print("\nPerformance by sentiment:")
print(sent_perf.reindex(SENT_ORDER).to_string())
print("\nBehaviour by sentiment:")
print(bhv.reindex(SENT_ORDER).to_string())
print("\nLeverage vs Sentiment:")
print(lev_sent.to_string())
print("\nWinner/Loser vs Sentiment:")
print(seg_sent.to_string())
