import m_src.dataloader
import m_src.dataloader_date_ver
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Load Data
data_loader = m_src.dataloader.DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

# print(len(test_data.keys())) # 1757

idx_list = list(test_data.keys())
idx = idx_list[0]
idx = 5

data_loader_with_time=m_src.dataloader_date_ver.DateLoader()
train_data = data_loader_with_time.load_train_as_dataframes(idx=[idx],padding_type="avg")

# df = train_data[idx]
# df_size = len(df)
# # 将 df 的大小调整为 24 的最大整数倍
# new_size = 24 * (df_size // 24)
# # 裁剪 DataFrame，使其大小为 24 的最大整数倍
# df = df.iloc[:new_size]
df = train_data[idx].head(24*80)

# df.plot()
# plt.show()
# Multiplicative Decomposition (! crash is value is 0)
# result_mul_h = seasonal_decompose(df['traffic_flow'], model='multiplicative', extrapolate_trend='freq')
# df_daily = result_mul_h.trend.resample('D').sum()
# result_mul_d = seasonal_decompose(df_daily, model='multiplicative', extrapolate_trend='freq')

# print(df_daily)
# Additive Decomposition
result_add_h = seasonal_decompose(df['traffic_flow'], model='additive', extrapolate_trend='freq',period=24)
result_add_d = seasonal_decompose(df['traffic_flow'], model='additive', extrapolate_trend='freq',period=24*7)

# # Plot
plt.rcParams.update({'figure.figsize': (10,10)})
# result_mul_h.plot().suptitle('Multiplicative Hourly Decompose', fontsize=22)
# result_mul_d.plot().suptitle('Multiplicative Daily Decompose', fontsize=22)
result_add_h.plot().suptitle('Additive Hourly Decompose', fontsize=22)
result_add_d.plot().suptitle('Additive Daily Decompose', fontsize=22)
plt.show()