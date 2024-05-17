import sys
import torch
from exp import Exp
import argparse

args = argparse.Namespace()


# configs
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = False
args.checkpoints ="./checkpoints/"

# training paras
args.train_epochs = 2
args.learning_rate= 0.0001

# model super paras
args.decomposition = 0
args.encoder_input_size = 1
args.patch_len = 16
args.stride = 8
args.context_window = 96
args.target_window = 16
args.padding_patch = "end"
args.batch_size=128
args.seq_len= 16
args.label_len= 4
args.pred_len= 8

exp = Exp(args)  # set experiments

setting="setting"

print(
    ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
)
exp.train(setting)

print(
    ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
)
exp.test(setting,test_model_from_path=1)