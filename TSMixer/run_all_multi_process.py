import multiprocessing
import time
import json
import sys
import torch
from exp import Exp
import argparse
import matplotlib.pyplot as plt
from dataloader import DateLoader
import jsonlines

######################################################
use_gpu = True
######################################################
data_loader = DateLoader()
test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

with open('data/work_list/large_work_list0.json', 'r') as file:
    loaded_list = json.load(file)
idx_list = list(loaded_list)
# idx_list = list(test_data.keys())

args = argparse.Namespace()


# configs
args.use_gpu = True if torch.cuda.is_available() and use_gpu else False
args.use_multi_gpu = False
args.gpu = 0
args.checkpoints = "./checkpoints_/"

# training paras
args.train_epochs = 100
args.learning_rate = 0.0001

# model super paras
args.decomposition = 0
args.encoder_input_size = 1
args.patch_len = 24
args.target_window = 1
args.padding_patch = "end"
args.batch_size = 3600
args.seq_len = args.patch_len
args.label_len = 0
args.pred_len = args.target_window
args.e_layers = 1

def worker(idx,file_name):
# for idx in idx_list:
    ans = {}
    ans[idx] = []
    test_data_for_single = test_data[idx]
    args.sensor_id = [idx]
    exp = Exp(args)  # set experiments
    setting = "ts_setting_" + idx

    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    exp.train(setting)
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    result_list = exp.predict(setting, 0)[idx]
    
    for k in test_data_for_single:
        ans[idx].append(result_list[k[0]])

    with jsonlines.open(file_name, "a") as fout:
        fout.write({idx:ans[idx]}) 

if __name__ == '__main__':
    # 要处理的任务参数列表，每个元素是一个元组
    task_params = idx_list


    # 创建一个包含两个进程的进程池
    with multiprocessing.Pool(processes=2) as pool:
        # 为每个任务生成单独的文件名
        results = [pool.apply_async(worker, args=(task, f'test_ans_{i}.json')) for i, task in enumerate(task_params)]

        # 等待所有任务完成
        [result.get() for result in results]
