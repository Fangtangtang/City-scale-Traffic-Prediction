from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np
import m_src.dataloader
import m_src.dataloader_date_ver
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
import jsonlines
import json

data_loader = m_src.dataloader.DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

print(len(test_data.keys())) # 1757

idx_list = list(test_data.keys())

ans={}

for idx in idx_list:
    try:
        ans[idx] = []
        
        test_data_for_single = test_data[idx]


        path = "data/train_with_time/{}_avg.jsonl".format(idx)
        
        train_data=[]
        with open(path, "r") as f:
            for line in f:
                sample = json.loads(line)
                train_data.append(sample["traffic_flow"])

        time_series = pd.Series(train_data)

        model = ARIMA(time_series, order=(5,1,1))

        model_fit = model.fit()

        pred = model_fit.predict(start=1, end=17857)
        pred = pred.values.tolist()

        for k in test_data_for_single:
            ans[idx].append(pred[k[0]])

        with jsonlines.open(f"ans_arima_.json", "a") as fout:
            fout.write({idx:ans[idx]}) 

        # with jsonlines.open(f"arima_pred{idx}.json", "a") as fout:
        #     fout.write({idx:pred}) 

    except BaseException:
        with open ("error","a") as f:
                f.write(idx+'\n')