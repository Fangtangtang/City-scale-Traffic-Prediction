import numpy
import json
import torch
from datetime import datetime, timedelta
import os
from typing import Union

class DateLoader:
    def __init__(self) -> None:
        pass
    
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
    

MAX_ITERATION = 17855 - 0 + 1

def format_train_data(data:list) -> Union[int, torch.tensor, torch.tensor]:
    ret = [[0.0, 0, 3] for _ in range(MAX_ITERATION)]
    for line in data:
        ret[line[0]][0], ret[line[0]][1], ret[line[0]][2] = line[1], 1, line[2]
    return data[0][0], torch.tensor(ret, dtype=torch.float32), torch.tensor(data)[:, 1]

if __name__ == "__main__":
    dl = DateLoader()
    dl.pre_test_load(test_path="")
    dl.pre_train_load(train_path="")
    with open("pre_train.jsonl", "r") as fin:
        for line in fin:
            line = json.loads(line)
            for k in line:
                with open(f"train/{k}.jsonl", "a") as fout:
                    fout.write(json.dumps(line) + '\n')