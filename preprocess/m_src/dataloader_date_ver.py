import json
import jsonlines
import torch
from datetime import datetime, timedelta
import os
from typing import Union
import platform
import pandas as pd


def path_convert(path)->str:
    os_name = platform.system()

    if os_name == 'Windows':
        return path.replace("/", "\\");
    elif os_name == 'Linux':
        return path.replace("\\", "/");
    else:
        print("Only support windows or linux currently, please check path format manually.")

class DateLoader:
    def __init__(self) -> None:
        self.pre_train_path=path_convert("data/pre_train_with_date.jsonl")
        self.pre_test_path=path_convert("data/pre_test_with_date.jsonl")
        self.train_dic_path=path_convert("data/train_with_time/")

    def pre_train_load(self, train_path="") -> None:    # 17855 0
                                                        # 2030
        train_path=path_convert(train_path)
        if os.path.exists(self.pre_train_path):
            return
        with open(train_path, "r") as fin:
            for line in fin.readlines()[1:]:
                line = line.rstrip('\n')
                line = line.split(',')
                if len(line) != 4:
                    raise ValueError("Wrong dara format!")
            
                dict={"sensor_id":line[0],"time":line[1],"condition":eval(line[2]),"traffic_flow":eval(line[3])}
                with jsonlines.open(self.pre_train_path,"a") as fout:
                    fout.write(dict)

    
    def pre_test_load(self, test_path="") -> None:  # 17832 8784
                                                    # 1757
        test_path=path_convert(test_path)
        if os.path.exists(self.pre_test_path):
            return
    
        with open(test_path, "r") as fin:
            with jsonlines.open(self.pre_test_path,"a") as fout:
                for line in fin.readlines()[1:]:
                    line = line.rstrip('\n')
                    line = line.split(',')
                    if len(line) != 4:
                        raise ValueError("Wrong dara format!")
                
                    dict={"task_id":line[0],"sensor_id":line[1],"time":line[2],"condition":eval(line[3])}
                    fout.write(dict)

    def split_train_data(self) -> None:
        with open(self.pre_train_path, "r") as fin:
            for line in fin:
                line = json.loads(line)
                id=line["sensor_id"]
                path = os.path.join(self.train_dic_path, f"{id}.jsonl")
                with open(path, "a") as fout:
                    fout.write(json.dumps(line) + '\n')

    def creat_train_data_with_padding(self,org_path,padding_type,id):
        time_beg = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
        current_time=time_beg
        t_path = os.path.join(self.train_dic_path, f"{id}_{padding_type}.jsonl")
        
        with open(org_path, "r") as fin:
            with jsonlines.open(t_path,"a") as fout:
                for line in fin:
                    line = json.loads(line)
                    time= datetime.strptime(line['time'], r"%Y-%m-%d %H:%M:%S")
                    while current_time != time:
                        dict={}
                        if padding_type=="zero":
                            dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":0}
                        if padding_type=="bak":
                            dict={"sensor_id":f"{id}","time":current_time.strftime('%Y-%m-%d %H:%M:%S'),"condition":0,"traffic_flow":line["traffic_flow"]}
                        fout.write(dict)
                        current_time = current_time + timedelta(hours=1)
                    fout.write(line)
                    current_time = current_time + timedelta(hours=1)
        
                
            

    def load_train_as_dataframes(self,padding_type="", idx=[]) -> dict:
        '''
            padding_type:
                - ""
                - "zero": use 0 for padding
                - "bak": use the first data after it for padding
        '''
        ret = {}
        for i in idx:
            t_path = os.path.join(self.train_dic_path, f"{i}_{padding_type}.jsonl")

            if not os.path.exists(t_path):
                self.creat_train_data_with_padding(
                    os.path.join(self.train_dic_path, f"{i}.jsonl"),
                    padding_type,
                    id=i
                )
                
            df = pd.read_json(path_or_buf=t_path, 
                                lines=True,
                                convert_dates={'time': lambda x: pd.to_datetime(x, format='%Y-%m-%d %H', errors='coerce')}
                                )
            # 设置 'time' 列为索引， 以hour为粒度
            df.set_index('time', inplace=True)
            df.index = pd.DatetimeIndex(df.index, freq='H')
            ret[i]=df

        return ret
    
if __name__ == "__main__":
    dl = DateLoader()
    dl.pre_test_load(test_path="data\\raw\\loop_sensor_test_x.csv")
    # dl.pre_train_load(train_path="data\\raw\\loop_sensor_train.csv")
    # dl.split_train_data()