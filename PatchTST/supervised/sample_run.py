import sys
import torch
from exp import Exp
import argparse
import matplotlib.pyplot as plt

######################################################
use_gpu=True
######################################################

args = argparse.Namespace()

args.sensor_id = ["1","2"]

# configs
args.use_gpu = True if torch.cuda.is_available() and use_gpu else False
args.use_multi_gpu = False
args.gpu = 0
args.checkpoints = "./checkpoints/"

# training paras
args.train_epochs = 5
args.learning_rate = 0.0001

# model super paras
args.decomposition = 0
args.encoder_input_size = 2 # ! we want to use every dimension as a sensor channel
args.patch_len = 7 * 24
args.stride = 4 * 24
args.target_window = 24
args.padding_patch = "end"
args.batch_size = 64
args.seq_len = 7 * 24
args.label_len = 0
args.pred_len = 24

exp = Exp(args)  # set experiments

setting = "setting" 

print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
# exp.train(setting)

print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
pred_result, data = exp.predict(setting, 1)
raw_data_list=data.get_raw()
# print(pred_result)
print(pred_result.keys())

for idx in args.sensor_id:
    raw_data=raw_data_list[f"{idx}_flow"]
    result=pred_result[idx]
    keys = list(result.keys())
    values = list(result.values())
    print(int(keys[0]))
    print(keys[-1])

    result_list = []
    x_values = range(data.raw_data_length())

    for i in x_values:
        if f"{i+1}" in result:
            result_list.append(result[f"{i+1}"])
        else:
            result_list.append(raw_data[i])

    # print(result_list)
    # 创建折线图
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, result_list, marker="o", label="Result")
    plt.plot(
        x_values, raw_data, marker="s", label="Data"
    )

    # 添加标题和标签
    plt.title("Two Lines Plot")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 添加图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()
