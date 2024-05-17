import numpy
import json
import torch
from datetime import datetime, timedelta
import os
from typing import Union
import pandas as pd
import jsonlines

class DateLoader:
    def __init__(self) -> None:
        self.train_dic_path = "data/train_with_time"
    
    def pre_train_load(self, train_path="") -> None:    # 17855 0
                                                        # 2030
        if os.path.exists("data/pre_train.json"):
            return
        train_data = {}
        max_hour = 0
        min_hour = 1 << 32
        with open(train_path, "r") as fin:
            for line in fin.readlines()[1:]:
                line = line.rstrip('\n')
                line = line.split(',')
                if len(line) != 4:
                    raise ValueError("Wrong dara format!")
                time_end = datetime.strptime(line[1], r"%Y-%m-%d %H:%M:%S")
                time_beg = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
                time_dif = time_end - time_beg
                diff_hour = int(time_dif.total_seconds() / 3600)
                max_hour = max(max_hour, diff_hour)
                min_hour = min(min_hour, diff_hour)
                idx = line[0]
                if idx not in train_data:
                    train_data[idx] = []
                train_data[idx].append([diff_hour, eval(line[3]), eval(line[2])])
        print(max_hour, min_hour)
        with open("data/pre_train.json", "w") as fout:
            for idx in train_data:
                fout.write(json.dumps({idx : train_data[idx]}) + '\n')
    
    def pre_test_load(self, test_path="") -> None:  # 17832 8784
                                                    # 1757
        if os.path.exists("data/pre_test.json"):
            return
        test_data = {}
        max_hour = 0
        min_hour = 1 << 32
        with open(test_path, "r") as fin:
            for line in fin.readlines()[1:]:
                line = line.rstrip('\n')
                line = line.split(',')
                if len(line) != 4:
                    raise ValueError("Wrong dara format!")
                time_end = datetime.strptime(line[2], r"%Y-%m-%d %H:%M:%S")
                time_beg = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
                time_dif = time_end - time_beg
                diff_hour = int(time_dif.total_seconds() / 3600)
                max_hour = max(max_hour, diff_hour)
                min_hour = min(min_hour, diff_hour)
                idx = line[1]
                if idx not in test_data:
                    test_data[idx] = []
                test_data[idx].append([diff_hour, eval(line[3])])
        print(max_hour, min_hour)
        with open("data/pre_test.json", "w") as fout:
            for idx in test_data:
                fout.write(json.dumps({idx : test_data[idx]}) + '\n')

    def load_test(self, test_path="") -> dict:
        if not os.path.exists(test_path):
            return None
        ret = {}
        with open(test_path) as fw:
            for line in fw:
                line = json.loads(line)
                for k in line:
                    ret[k] = line[k]
        return ret
    
    def load_train(self, train_path="", idx=[]) -> dict:
        ret = {}
        for i in idx:
            t_path = os.path.join(train_path, f"{i}.jsonl")
            if not os.path.exists(t_path):
                return None
            with open(t_path, "r") as fw:
                for line in fw:
                    line = json.loads(line)
                    for k in line:
                        ret[k] = line[k]
        return ret
    
    def create_train_data_with_padding(self,org_path,padding_type,id):
        time_beg = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
        current_time=time_beg
        t_path = os.path.join(self.train_dic_path, f"{id}_{padding_type}.jsonl")
        
        entry_size=7*24
        flow_sum=[0] *entry_size
        valid_cnt=[0] *entry_size

        # avg in week
        larger_flow_sum=[0]*24
        larger_valid_cnt=[0] *24
        avg_hourly_flow=[0]*24

        if padding_type=="avg" or padding_type=="avg2":
            with open(org_path, "r") as fin:
                cnt=0
                for line in fin:
                    line = json.loads(line)
                    time= datetime.strptime(line['time'], r"%Y-%m-%d %H:%M:%S")
                    while current_time != time:
                        current_time = current_time + timedelta(hours=1)
                        cnt+=1

                    entry_idx=cnt%entry_size
                    flow_sum[entry_idx]+=line["traffic_flow"]
                    valid_cnt[entry_idx]+=1
                    # print(line['time'],'\t',cnt%24,'\t',cnt%entry_size)
                    current_time = current_time + timedelta(hours=1)
                    cnt+=1
            if padding_type=="avg":
                for i in range(entry_size):
                    larger_flow_sum[i%24]+=flow_sum[i]
                    larger_valid_cnt[i%24]+=valid_cnt[i]
                
                total_avg=sum(larger_flow_sum)/sum(larger_valid_cnt)
                for i in range(24):
                    if larger_valid_cnt[i]!=0:
                        avg_hourly_flow[i]=larger_flow_sum[i]/larger_valid_cnt[i]
                    else:
                        avg_hourly_flow[i]=total_avg
    

        current_time=time_beg
        with open(org_path, "r") as fin:
            cnt=0
            with jsonlines.open(t_path,"a") as f_out:
                for line in fin:
                    line = json.loads(line)
                    line["config"] = 1
                    time= datetime.strptime(line['time'], r"%Y-%m-%d %H:%M:%S")
                    while current_time != time:
                        dict={}
                        if padding_type=="zero":
                            dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":0.1, "config":0}
                        if padding_type=="bak":
                            dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":line["traffic_flow"], "config":0}
                        if padding_type=="avg":
                            entry_idx=cnt%entry_size
                            avg_flow=0
                            if valid_cnt[entry_idx]!=0:
                                avg_flow = flow_sum[entry_idx]/valid_cnt[entry_idx]
                            else:
                                print(current_time.strftime('%Y-%m-%d %H:%M:%S')+"\t24*7 failed.")
                                avg_flow = avg_hourly_flow[entry_idx%24]
                            if avg_flow < 0.1:
                                avg_flow = 0.1
                            dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":avg_flow, "config":0}
                        f_out.write(dict)
                        current_time = current_time + timedelta(hours=1)
                        cnt+=1
                    if line['traffic_flow'] < 0.1:
                        line['traffic_flow'] = 0.1
                    f_out.write(line)
                    # print(line['time'],'\t',cnt%24,'\t',cnt%entry_size)
                    current_time = current_time + timedelta(hours=1)
                    cnt+=1
                endtime = datetime(year=2024,month=1,day=15,hour=0,minute=0,second=0)
                while current_time != endtime:
                    dict={}
                    if padding_type=="zero":
                        dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":0.1, "config":0}
                    if padding_type=="bak":
                        dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":line["traffic_flow"], "config":0}
                    if padding_type=="avg":
                        entry_idx=cnt%entry_size
                        avg_flow=0
                        if valid_cnt[entry_idx]!=0:
                            avg_flow = flow_sum[entry_idx]/valid_cnt[entry_idx]
                        else:
                            print(current_time.strftime('%Y-%m-%d %H:%M:%S')+"\t24*7 failed.")
                            avg_flow = avg_hourly_flow[entry_idx%24]
                        if avg_flow < 0.1:
                            avg_flow = 0.1
                        dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":avg_flow, "config":0}
                    f_out.write(dict)
                    current_time = current_time + timedelta(hours=1)
                    cnt+=1
            

    def load_train_as_dataframes(self,padding_type="", idx=[]) -> dict:
        '''
            padding_type:
                - ""
                - "zero": use 0 for padding
                - "bak": use the first data after it for padding
                - "avg": average from existing data (24*7 entries)
        '''
        ret = {}
        for i in idx:
            t_path = os.path.join(self.train_dic_path, f"{i}_{padding_type}.jsonl")

            if not os.path.exists(t_path):
                self.create_train_data_with_padding(
                    os.path.join(self.train_dic_path, f"{i}_origin.jsonl"),
                    padding_type,
                    id=i
                )
                
            df = pd.read_json(path_or_buf=t_path, 
                                lines=True,
                                convert_dates={'time': lambda x: pd.to_datetime(x, format='%Y-%m-%d %H', errors='coerce')}
                                )
            # 设置 'time' 列为索引， 以hour为粒度
            begin_time = df.loc[0, "time"]
            df.set_index('time', inplace=True)
            df.index = pd.DatetimeIndex(df.index, freq='H')

            ret[i]=[df, begin_time]

        return ret

MAX_ITERATION = 17855 - 0 + 1

def format_train_data(data:list) -> Union[int, torch.tensor, torch.tensor]:
    ret = [[0.0, 0, 3] for _ in range(MAX_ITERATION)]
    mem = [[0.0, 0] for _ in range(7*24)]
    for line in data:
        ret[line[0]][0], ret[line[0]][1], ret[line[0]][2] = line[1], 1, line[2]
        mem[line[0] % (7*24)][0] = (mem[line[0] % (7*24)][0] * mem[line[0] % (7*24)][1] + line[1]) / (mem[line[0] % (7*24)][1] + 1)
        mem[line[0] % (7*24)][1] += 1
    for i in range(MAX_ITERATION):
        line = ret[i]
        if line[1] == 0:
            line[0] = mem[i % (7*24)][0]
    return data[0][0], torch.tensor(ret, dtype=torch.float32), torch.tensor(data)[:, 1]

if __name__ == "__main__":
    dl = DateLoader()
    # dl.pre_test_load(test_path="data/pre_test.json")
    # dl.pre_train_load(train_path="data/pre_train.json")
    with open("data/pre_train.json", "r") as fin:
        for line in fin:
            line = json.loads(line)
            for k in line:
                with open(f"data/train/{k}.jsonl", "a") as fout:
                    fout.write(json.dumps(line) + '\n')