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

idx_list = list(test_data.keys())
idx = idx_list[0]
idx = 5
data_loader_with_time=m_src.dataloader_date_ver.DateLoader()
train_data = data_loader_with_time.load_train_as_dataframes(idx=[idx],padding_type="bak")

df = train_data[idx].head(200)
# Multiplicative Decomposition 
result_mul = seasonal_decompose(df['traffic_flow'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df['traffic_flow'], model='additive', extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()