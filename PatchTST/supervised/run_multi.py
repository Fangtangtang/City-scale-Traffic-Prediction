import sys
import torch
from exp import Exp
import argparse
import jsonlines
from dataloader import DateLoader

######################################################
use_gpu=True
######################################################

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


data_loader = DateLoader()
test_data = data_loader.load_test(test_path="../data/pre_test.json")

idx_lists = chunk_list(list(test_data.keys()),8)

ans = {}
for idx_list in idx_lists:
    args = argparse.Namespace()

    args.sensor_id = idx_list

    # configs
    args.use_gpu = True if torch.cuda.is_available() and use_gpu else False
    args.use_multi_gpu = False
    args.gpu = 0
    args.checkpoints = "./checkpoints/"

    # training paras
    args.train_epochs = 200
    args.learning_rate = 0.0001

    # model super paras
    args.decomposition = 0
    args.encoder_input_size = len(idx_list) # ! we want to use every dimension as a sensor channel
    args.patch_len = 24
    args.stride = 4
    args.target_window = 1
    args.padding_patch = "end"
    args.batch_size = 9000
    args.seq_len = args.patch_len
    args.label_len = 0
    args.pred_len = args.target_window

    exp = Exp(args)  # set experiments

    setting = "setting" 

    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    exp.train(setting)

    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    result_list = exp.predict(setting, 0)

    for idx in args.sensor_id:
        test_data_for_single = test_data[idx]

        for k in test_data_for_single:
            ans[idx].append(result_list[idx][k[0]].item())

        with jsonlines.open(f"ans/ans{idx}.json", "a") as fout:
            fout.write({idx:ans[idx]}) 
