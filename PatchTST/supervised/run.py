import sys
import torch
import jsonlines
import numpy as np
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
idx_list = ['40']
ans={}

args = argparse.Namespace()

# configs
args.use_gpu = True if torch.cuda.is_available() and use_gpu else False
args.use_multi_gpu = False
args.gpu = 0
args.checkpoints = "./checkpoints/"

# training paras
args.train_epochs = 300
args.learning_rate = 0.0001

# model super paras
args.decomposition = 0
args.encoder_input_size = 1
args.patch_len = 24 
args.context_window = args.patch_len
args.stride = 4 
args.target_window = 1
args.padding_patch = "end"
args.batch_size = 9000
args.seq_len = args.patch_len
args.label_len = 0
args.pred_len = args.target_window

for idx in idx_list:
    ans[idx] = []
    # test_data_for_single = test_data[idx]

    args.sensor_id = [idx]
    exp = Exp(args)  # set experiments

    setting = "setting_" + idx

    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    exp.train(setting)
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    # result_list = exp.predict(setting, 0)[idx]
    results = exp.predict(setting, 0)

    # np.save("tst_pred40.npy", np.array(result_list))

    # for k in test_data_for_single:
    #     ans[idx].append(result_list[k[0]].item())

    with jsonlines.open(f"tst_pred{idx}.json", "a") as fout:
        fout.write(results) 
        # fout.flush()
