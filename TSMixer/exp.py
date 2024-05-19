import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import TSMixer_model
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import json
from collections import defaultdict


class Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.channel_list = self.args.sensor_id
        self.data_set = DataSet(
            self.args.sensor_id,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
        )
        
        self.args.context_window = len(self.data_set)

        self.model = self._build_model().to(self.device)

        print(len(self.data_set))
        self.data_loader = DataLoader(
            self.data_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )

    def _build_model(self):
        model = TSMixer_model.Model(
            encoder_input_size=self.args.encoder_input_size,
            seq_len=self.args.seq_len,
            e_layers=self.args.e_layers,
            pred_len=self.args.pred_len,
        )
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self, flag):
        return self.data_set, self.data_loader

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        # 不跟踪梯度，加快计算，减少内存消耗
        with torch.no_grad():
            for i, (batch_x, batch_y, stamp_x, stamp_y) in enumerate(vali_loader):
                # put data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # apply
                outputs = self.model(batch_x)

                # get predicted feature of given length
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # calculate loss
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        # switch back to train mode
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        loss_list = []
        for epoch in range(self.args.train_epochs):
            if epoch % 10 == 0:
                print(
                    ">>>>>>>>>>>>>>>>>>>> Epoch {} <<<<<<<<<<<<<<<<<<<<<".format(epoch)
                )

            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, stamp_x, stamp_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
              
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
              

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            print(np.average(train_loss))
            loss_list.append(np.average(train_loss))

            if epoch % 10 == 0:
                loss_list.append(np.average(train_loss))

            if epoch > 0 and epoch % 50 == 0:
                torch.save(
                    self.model.state_dict(),
                    path + "/" + "checkpoint{}.pth".format(epoch / 100),
                )

        print(loss_list)
        torch.save(self.model.state_dict(), path + "/" + "checkpoint.pth")
        return self.model

    def test(self, setting, test_model_from_path=0):
        test_data, test_loader = self._get_data(flag="test")

        if test_model_from_path:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        inputx = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # 不跟踪梯度，加快计算，减少内存消耗
        with torch.no_grad():
            for i, (batch_x, batch_y, stamp_x, stamp_y) in enumerate(test_loader):
                # put data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # apply
                outputs = self.model(batch_x)

                # get predicted feature of given length
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "pred.npy", preds)

        return

    def predict(self, setting, load_model_from_path=0):
        pred_data, pred_loader = self._get_data(flag="pred")

        self.answer_list = {}
        ans = {}
        cnt = {}
        for idx in self.channel_list:
            self.answer_list[idx] = []

        for i in range(len(self.channel_list)):
            ans[i] = defaultdict(float)
            cnt[i] = defaultdict(int)

        if load_model_from_path:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, stamp_x, stamp_y) in enumerate(pred_loader):
                # put data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # apply
                outputs = self.model(batch_x)

                # get predicted feature of given length
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()  # .squeeze()

            
                for i in range(batch_y.shape[0]):
                    for j in range(batch_y.shape[1]):
                        for k in range(batch_y.shape[2]):
                            # for channel in range(batch_y.shape[2]):
                            cnt[k][f"{stamp_y[i][j][0]}"] += 1
                            ans[k][f"{stamp_y[i][j][0]}"] += pred[i][j][k]

                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        for channel in range(len(self.channel_list)):
            averages = {label: ans[channel][label] / cnt[channel][label] for label in ans[channel]}
            self.answer_list[self.channel_list[channel]]=averages

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return self.answer_list, pred_data


class DataSet(Dataset):

    def __init__(self, sensor_id_list, flag="train", size=None, features="S"):

        if size == None:
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        data={}
        
        data_y = []
        for idx in sensor_id_list:
            path = "data/train_with_time/{}_avg.jsonl".format(idx)

            # data_x = []
            data_y = []
            with open(path, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    # data_x.append(sample["time"])
                    data_y.append(sample["traffic_flow"])

            data["{}_flow".format(idx)] =  data_y

        self.raw_data=data

        self.data_len=len(data_y)
        # data["stamp"] =  range(self.data_len)

        df = pd.DataFrame(data)
        # df.set_index('stamp', inplace=True)
        # cols_data = df.columns[1:]
        cols_data = df.columns
        print(cols_data)
        self.data = df[cols_data].values
        data = {
            "time": range(self.data_len),
        }
        df = pd.DataFrame(data)
        cols_data = df.columns
        print(cols_data)
        self.stamp = df[cols_data].values

    def __len__(self):
        return self.data_len - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        
        return (
            self.data[s_begin:s_end],
            self.data[r_begin:r_end],
            self.stamp[s_begin:s_end],
            self.stamp[r_begin:r_end],
        )

    def raw_data_length(self):
        return self.data_len
    
    def get_raw(self):
        return self.raw_data
