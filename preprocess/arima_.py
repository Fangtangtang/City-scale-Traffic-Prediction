from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np
import m_src.dataloader
import m_src.dataloader_date_ver
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
import json

data_loader = m_src.dataloader.DateLoader()
# test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

# print(len(test_data.keys())) # 1757

# idx_list = list(test_data.keys())
# idx = idx_list[0]
idx = '5'


path = "data/train_with_time/{}_avg.jsonl".format(idx)

train_data=[]
with open(path, "r") as f:
    for line in f:
        sample = json.loads(line)
        # data_x.append(sample["time"])
        train_data.append(sample["traffic_flow"])

# train_data = np.array(train_data[idx])
# train_data=train_data.T


time_series = pd.Series(train_data)

model = ARIMA(time_series, order=(5,1,1))
# model = pm.auto_arima(time_series, seasonal=True, stepwise=True)
# best_model = model.model

model_fit = model.fit()
# print(model_fit.summary())

pred = model_fit.predict(start=0, end=len(time_series) - 1)

print(pred)
plt.plot(time_series, label='org')
plt.plot(pred, label='fit', color='red')
plt.title('ARIMA')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.show()
