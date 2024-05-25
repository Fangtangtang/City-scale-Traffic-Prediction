import sys
import torch
from exp import Exp
import argparse
import matplotlib.pyplot as plt
import jsonlines
######################################################
use_gpu=True
######################################################

args = argparse.Namespace()

args.sensor_id = ["5"]

# configs
args.use_gpu = True if torch.cuda.is_available() and use_gpu else False
args.use_multi_gpu = False
args.gpu = 0
args.checkpoints = "./checkpoints/"

# training paras
args.train_epochs = 2000
args.learning_rate = 0.0001

# model super paras
args.decomposition = 0
args.encoder_input_size = 1 
args.patch_len = 24
args.target_window = 1
args.padding_patch = "end"
args.batch_size = 9000
args.seq_len = args.patch_len
args.label_len = 0
args.pred_len = args.target_window
args.e_layers=3
exp = Exp(args)  # set experiments

setting = "setting" 

print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
exp.train(setting)

print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))

results = exp.predict(setting, 0)

    # np.save("tst_pred40.npy", np.array(result_list))

    # for k in test_data_for_single:
    #     ans[idx].append(result_list[k[0]].item())
idx=args.sensor_id[0]

with jsonlines.open(f"tm_mixer_pred{idx}.json", "a") as fout:
    fout.write(results) 

# pred_result = exp.predict(setting,0)
# raw_data_list=data.get_raw()
# # print(pred_result)
# print(pred_result.keys())

# for idx in args.sensor_id:
#     raw_data=raw_data_list[f"{idx}_flow"]
#     result=pred_result[idx]
#     keys = list(result.keys())
#     values = list(result.values())
#     print(int(keys[0]))
#     print(keys[-1])

#     result_list = []
#     x_values = range(data.raw_data_length())

#     for i in x_values:
#         if f"{i}" in result:
#             result_list.append(result[f"{i}"])
#         else:
#             result_list.append(raw_data[i])

#     with jsonlines.open(f"tm_mixer_pred{idx}.json", "a") as fout:
#         fout.write({idx:result_list}) 
#     # # print(result_list)
#     # predictions = torch.tensor(result_list, dtype=torch.float32)
#     # # 真实值
#     # targets = torch.tensor(raw_data, dtype=torch.float32)
#     # mse_loss = torch.nn.MSELoss()

#     # loss = mse_loss(predictions, targets)
#     # print(loss.item())
#     # # 创建折线图
#     # plt.figure(figsize=(10, 5))
#     # plt.plot(x_values, result_list, marker="o", label="Result")
#     # plt.plot(
#     #     x_values, raw_data, marker="s", label="Data"
#     # )

#     # # 添加标题和标签
#     # plt.title("Two Lines Plot")
#     # plt.xlabel("X")
#     # plt.ylabel("Y")

#     # # 添加图例
#     # plt.legend()

#     # # 显示图形
#     # plt.grid(True)
#     # plt.show()
