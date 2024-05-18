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
    model, loss_list  = exp.train(setting)

    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    pred_result, data = exp.predict(setting, 1)
    raw_data_list=data.get_raw()
    # print(pred_result)
    print(pred_result.keys())

    with jsonlines.open("ans_multi/loss_list.jsonl", "a") as f_out:
        dict={
            "id":args.sensor_id,
            "loss_list":loss_list,
            }
        f_out.write(dict)

    for idx in args.sensor_id:
        raw_data=raw_data_list[f"{idx}_flow"]
        result=pred_result[idx]
        keys = list(result.keys())
        values = list(result.values())

        result_list = []
        x_values = range(data.raw_data_length())

        for i in x_values:
            if f"{i+1}" in result:
                result_list.append(result[f"{i+1}"])
            else:
                result_list.append(raw_data[i])

        ans[idx] = []
        test_data_for_single = test_data[idx]
        for k in test_data_for_single:
            ans[idx].append(result_list[k[0]].item())

        with jsonlines.open("ans_multi/ans.json", "a") as fout:
            fout.write({idx:ans[idx]})

    print("1!!!!")
    break    