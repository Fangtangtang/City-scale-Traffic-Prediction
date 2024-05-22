import sys
import torch
import jsonlines
from exp import Exp
import argparse
import matplotlib.pyplot as plt
# import  preprocess.m_src as m_src

######################################################
use_gpu=True
######################################################

# data_loader = m_src.dataloader.DateLoader()
# test_data = data_loader.load_test(test_path="data/pre_test.jsonl")

# idx_list = list(test_data.keys())
idx_list = ['5']
ans={}

args = argparse.Namespace()

# configs
args.use_gpu = True if torch.cuda.is_available() and use_gpu else False
args.use_multi_gpu = False
args.gpu = 0
args.checkpoints = "./checkpoints/"

# training paras
args.train_epochs = 160
args.learning_rate = 0.0001

# model super paras
args.decomposition = 0
args.encoder_input_size = 1
args.patch_len = 7 
args.stride = 4 
args.target_window = 1
args.padding_patch = "end"
args.batch_size = 64
args.seq_len = args.patch_len
args.label_len = 0
args.pred_len = args.target_window

for idx in idx_list:
    ans[idx] = []
    # test_data_for_single = test_data[idx]

    args.sensor_id = str(idx)
    exp = Exp(args)  # set experiments

    setting = "setting_" + args.sensor_id

    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    exp.train(setting)
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    result_list = exp.predict(setting, 0)

    # for k in test_data_for_single:
    #     ans[idx].append(result_list[k[0]].item())

    # with jsonlines.open(f"ans/{idx}_.json", "a") as fout:
    #     fout.write({idx:ans[idx]}) 
    #     fout.flush()
