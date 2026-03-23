# gold-xauusd-history-com



import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
import requests
import json

# 忽略警告
warnings.filterwarnings('ignore')

# 1. 數據下載
print("正在從 Yahoo Finance 下載數據...")
ticker = "GC=F"
df = yf.download(ticker, start="1990-01-01", end="2026-03-24", progress=False)
if df.empty:
    raise ValueError(f"未能下載 {ticker} 的數據")

# 2. 數據預處理
closes = df['Close'].dropna().values
log_closes = np.log(closes)
dates = df['Close'].dropna().index
window_size = 60

def normalize(series):
    if len(series) == 0 or (np.max(series) - np.min(series) + 1e-8) == 0:
        return np.zeros_like(series)
    return (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-8)

target_log_pattern = log_closes[-window_size:]
target_norm = normalize(target_log_pattern)

# 3. 滑動視窗比對 (Pearson Correlation)
similarities = []
print(f"正在分析歷史走勢並計算相似度...")
for i in range(len(log_closes) - window_size * 2):
    historical_window = log_closes[i : i + window_size]
    historical_norm = normalize(historical_window)
    if np.std(historical_norm) > 0:
        corr, _ = pearsonr(target_norm.flatten(), historical_norm.flatten())
        similarities.append((float(corr), i))

similarities.sort(key=lambda x: x[0], reverse=True)

# 4. 繪製比對圖 (Top 3)
plt.style.use('bmh')
fig, axes = plt.subplots(4, 1, figsize=(14, 18))
axes[0].plot(dates[-window_size:], target_norm, color='#1f77b4', linewidth=2, label='Current')
axes[0].set_title(f"Current Pattern (Last {window_size} Days)", fontsize=14, fontweight='bold')

for rank in range(min(3, len(similarities))):
    corr, idx = similarities[rank]
    hist_dates = dates[idx : idx + window_size]
    hist_pattern = log_closes[idx : idx + window_size]
    axes[rank+1].plot(hist_dates, normalize(hist_pattern), color='#d62728', linewidth=2)
    axes[rank+1].set_title(f"Rank {rank+1}: {hist_dates[0].date()} to {hist_dates[-1].date()} (Similarity: {corr:.4f})", fontsize=12)
plt.tight_layout()
plt.show()

# 5. 繪製全景圖並標記背景
plt.figure(figsize=(16, 8))
plt.plot(dates, log_closes, color='gray', alpha=0.3, label='Log Price History')
colors = ['#FFD700', '#FFA500', '#FF8C00']
analysis_context = ""

for rank in range(min(3, len(similarities))):
    corr, idx = similarities[rank]
    start_dt, end_dt = dates[idx], dates[idx + window_size]
    plt.axvspan(start_dt, end_dt, color=colors[rank], alpha=0.4, label=f'Rank {rank+1} ({start_dt.year})')
    analysis_context += f"Rank {rank+1}: {start_dt.date()} to {end_dt.date()}, Similarity: {corr:.4f}\n"

plt.axvspan(dates[-window_size], dates[-1], color='cyan', alpha=0.2, label='Current')
plt.title("Gold Price Full View: Top 3 Matching Patterns Highlighted", fontsize=15)
plt.legend(loc='upper left')
plt.show()

# 6. AI 深度總結 (DeepSeek API)
print("\n正在請求 DeepSeek AI 進行深度行情與型態擬合分析...")
api_key = "sk-9f7e3b8c4d674d948427d7a66bfe25d6"
url = "https://api.deepseek.com/chat/completions"

prompt = f"""
你是一位資深的黃金市場量化分析師。根據皮爾森相關係數比對，目前的黃金對數價格走勢與以下三個歷史區間最為相似：
{analysis_context}

請針對這些結果進行專業總結：
1. 文字敘述這些區間的匹配結果。
2. 這些區間（特別是 Rank 1）在歷史上處於什麼樣的宏觀背景？
3. 從「多線方程擬合」的角度，目前的築底型態最符合哪段行情？
4. 參考歷史，通常在見底後經過幾次震盪或需要多久時間才能真正走出底部區域？
5. 給出基於歷史相似性的未來 3-6 個月趨勢展望。
"""

try:
    response = requests.post(url, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                             data=json.dumps({"model": "deepseek-chat", "messages": [{"role": "system", "content": "你是一位專精於技術分析與型態識別的金融 AI 助手。"}, {"role": "user", "content": prompt}], "stream": False}))
    print("\n--- AI 行情深度分析與擬合總結 ---\n")
    print(response.json()['choices'][0]['message']['content'])
except Exception as e:
    print(f"API 請求失敗: {e}")
