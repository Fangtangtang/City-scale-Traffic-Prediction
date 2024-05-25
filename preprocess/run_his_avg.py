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

# idx_list = list(test_data.keys())
idx_list = ['5','40']

ans={}

for idx in idx_list:
    try:
        ans[idx] = []
        
        test_data_for_single = test_data[idx]


        path = "data/train_with_time/{}_avg.jsonl".format(idx)
        
        lines_cnt=0
        total=0
        total_cnt=0

        entry_size=7*24
        flow_sum=[0] *entry_size
        valid_cnt=[0] *entry_size


        train_data=[]

        with open(path, "r") as f:
            lines_cnt+=1
            for line in f:
                sample = json.loads(line)
                entry_idx=lines_cnt%entry_size
                if sample["config"]==1:
                     total_cnt+=1
                     total+=sample["traffic_flow"]
                     flow_sum[entry_idx]+=sample["traffic_flow"]
                     valid_cnt[entry_idx]+=1
                
                if valid_cnt[entry_idx]!=0:
                    train_data.append(flow_sum[entry_idx]/valid_cnt[entry_idx])
                else:
                    train_data.append(total/total_cnt)

        # with open(path, "r") as f:
        #     for line in f:   
        #         if sample["config"]==0:

        #             train_data.append(ha)
        #         else:
        #             train_data.append(sample["traffic_flow"])

        time_series = pd.Series(train_data)


        
        # print(len(pred))
        for k in test_data_for_single:
            # print(k[0])
            ans[idx].append(train_data[k[0]])

        # with jsonlines.open(f"ha_.json", "a") as fout:
        #     fout.write({idx:ans[idx]}) 

        # with jsonlines.open(f"ha_pred{idx}.json", "a") as fout:
        #     fout.write({idx:train_data}) 

    except BaseException:
        with open ("error","a") as f:
                f.write(idx+'\n')